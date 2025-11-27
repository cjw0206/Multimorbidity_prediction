import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import sys
import math
import json
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Literal

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import dgl
from dgl import DGLGraph

# GNN 백본 import
from . import GCN, GAT, GIN, HGT

#from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
#import optuna
#from optuna.trial import TrialState
from dataclasses import replace

import warnings
warnings.filterwarnings('ignore')

# -----------------------------
# 설정
# -----------------------------

@dataclass
class Settings:
    # --- 파일 경로  ---
    CSV_FILE = r"./imputed_data_with_knn_person_wave_v3.csv"
    PERSON_SIM_FILE = r"./graph/similar_person_th_0.7.csv"
    DISEASE_SIM_FILE = r"./graph/disease_edgelist.csv"
    HAS_DISEASE_FILE = r"./graph/has_disease_edgelist_long_v3.csv"
    RESULTS_FILE = r"./models/edge_p/[test2]multimorbidity_results.txt"
    EDGE_DIST_LOG_FILE = r"./[test2]edge_distribution_log.txt"

    # --- 실험 설정 (고정값) ---
    waves: List[int] = field(default_factory=lambda: [10])
    disease_init: Literal["onehot", "embed"] = "onehot"
    disease_emb_dim: int = 16
    neg_pos_ratio: float = 2.0 #정답 샘플 1개당 오답을 몇 개 사용할지
    num_classes: int = 10 # 클래스 정답 개수 9개질병+질병없음
    n_folds: int = 5 # 최종 Test를 위한 Fold 수
    #n_train_folds: int = 5 # 하이퍼파라미터 튜닝(학습)을 위한 Fold 수
    in_dim_pad: Optional[int] = None # GNN 입력 차원 패딩
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 2025
    #validation_seed: int = 42 # 왜 이렇게 하는지 이해가 안 간다..비상
    pred_threshold: float = 0.5 # 예측 임계값
    test_sampling_ratio: float = 0.2 # test로 보낼 edge 비율 조정


# --- 변수 리스트 (질병 변수 제외) ---
DISEASE_LIST = ["hibp", "diab", "cancr", "lung", "heart", "strok", "psych", "arthr", "memory"]
DISEASE_NAME_MAP = {
    "hypertension": "hibp",
    "diabetes": "diab",
    "cancer": "cancr",
    "stroke": "strok",
    "mental": "psych",
    "lung": "lung",
    "heart": "heart",
    "arthr": "arthr",
    "memory": "memory"
}

NUMERIC_VARS = ["doctim", "oopmd", "cesd", "imrc", "dlrc", "ser7", "bmi", "height", "weight", "bpsys", "bpdia", "bppuls", "puff", "shltc"]
BINARY_VARS = ["hosp", "nrshom", "doctor", "homcar", "drugs", "outpt", "dentst", "spcfac", "depres", "effort", "sleepr", "whappy", "flone", "fsad", "going", "drink", "smokev", "smoken"]
CATEGORICAL_VARS = ["shlt", "hltc3", "slfmem", "bwc20"]

DISEASE_TO_IDX: Dict[str, int] = {d: i for i, d in enumerate(DISEASE_LIST)}
IDX_TO_DISEASE: Dict[int, str] = {i: d for d, i in DISEASE_TO_IDX.items()}
NONE_CLASS_IDX = 9  # 9+1에서 'none' 클래스 인덱스

# -----------------------------
# 유틸/헬퍼
# -----------------------------

def ensure_dir(path: str): #디렉터리 폴더가 존재하지 않으면 자동으로 생성
    os.makedirs(os.path.dirname(path), exist_ok=True)

def set_seed(seed: int = 2025): #실험 결과를 다시 재현하기 위한
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def zscore(col: pd.Series, name: str = "") -> pd.Series: # 데이터의 칼럼별로 정규화(제트스코어)
    x = col.astype(float)
    mu = x.mean()
    std = x.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros(len(x)), index=col.index)
    return (x - mu) / std

def pad_right_with_zeros(X: np.ndarray, out_dim: int) -> np.ndarray: # 입력 데이터 차원 수 맞추기 패딩
    if X.shape[1] == out_dim:
        return X
    pad = out_dim - X.shape[1]
    return np.pad(X, ((0, 0), (0, pad)), mode="constant")

def torch_long(arr):
    # 입력이 pandas Series나 DataFrame이면 .values (.to_numpy())로 numpy 배열을 추출합니다.
    if hasattr(arr, 'values'):
        arr = arr.values
    return torch.as_tensor(arr, dtype=torch.long)

def torch_float(arr):
    return torch.as_tensor(arr, dtype=torch.float32)

def log_edge_distribution(log_file_path: str, wave: int, fixed_disease_name: str, fold_num: int, total_pos: int, total_neg: int, per_disease_counts: Dict[str, Tuple[int, int]]):
    """
    주어진 wave, fixed disease, fold에 대한 Test 엣지 분포를 파일에 기록합니다.
    """
    # ensure_dir 함수가 main 함수에서 이미 상위 디렉토리를 생성하므로 여기서는 생략 가능
    with open(log_file_path, 'a', encoding='utf-8') as f:
        f.write(f"--- Wave: {wave} | Fixed Disease: {fixed_disease_name} | Fold: {fold_num} ---\n")
        total_edges = total_pos + total_neg
        f.write(f"  - Total Test Edges: {total_edges} (Positive: {total_pos}, Negative: {total_neg})\n")
        f.write("  - Per-Disease Distribution:\n")
        if not per_disease_counts:
            f.write("    - No diseases with edges in the test set.\n")
        else:
            # 질병 이름 순으로 정렬하여 기록
            for disease_name, (pos, neg) in sorted(per_disease_counts.items()):
                f.write(f"    - Disease '{disease_name}': {pos} Positive, {neg} Negative\n")
        f.write("-" * 55 + "\n\n")

# === A1. 전역 vs 로컬 우주 비교 ===
def debug_print_disease_universe(disease_id_map, IDX_TO_DISEASE):
    # 전역 리스트(고정 순서)
    global_list = [IDX_TO_DISEASE[i] for i in sorted(IDX_TO_DISEASE.keys())]
    print("\n[UNIVERSE] Global disease order:", global_list)

    # 로컬 리스트(해당 wave에서 sorted된 타입드 ID 순서)
    inv_local = {tid: name for name, tid in disease_id_map.items()}
    local_list = [inv_local[i] for i in range(len(inv_local))]
    print("[UNIVERSE] Local  disease order:", local_list)

    # 전역 idx → 이름 → 로컬 typed id 매핑 테이블
    print("[UNIVERSE] GlobalIdx  Name   -> LocalTypedID")
    for g_idx in sorted(IDX_TO_DISEASE.keys()):
        name = IDX_TO_DISEASE[g_idx]
        print(f"           {g_idx:>3}      {name:<10} -> {disease_id_map.get(name, None)}")

    print("[UNIVERSE] Global order == Local order ? ->", global_list == local_list)

# === B1. split 단계 전역/로컬 비교 결과를 나란히 출력 ===
def debug_check_fixed_in_split(pd_with_ids, disease_id_map, fixed_disease_idx, IDX_TO_DISEASE):
    fixed_name  = IDX_TO_DISEASE[fixed_disease_idx]
    fixed_local = disease_id_map.get(fixed_name, None)

    print(f"\n[SPLIT] fixed_global={fixed_disease_idx} ({fixed_name}), fixed_local={fixed_local}")

    mask_global_eq = (pd_with_ids["disease_typed_id"] == fixed_disease_idx)  # (진단용) 전역으로 비교
    mask_local_eq  = (pd_with_ids["disease_typed_id"] == fixed_local)        # (정상) 로컬로 비교

    print("[SPLIT] count(disease_typed_id == fixed_global) =", int(mask_global_eq.sum()))
    print("[SPLIT] count(disease_typed_id == fixed_local ) =", int(mask_local_eq.sum()))

    diff = mask_global_eq ^ mask_local_eq
    if diff.any():
        print("[SPLIT] MISMATCH rows (head):")
        cols = ["person_wave_id","person_typed_id","disease","disease_typed_id"]
        print(pd_with_ids.loc[diff, cols].head())
    else:
        print("[SPLIT] No mismatch between global-vs-local equality masks ✔")


# -----------------------------
# 데이터 로딩
# -----------------------------

def load_person_sim_edges(path: str, wave: int) -> pd.DataFrame: #사람-사람
    df = pd.read_csv(path)
    def is_wave_id(x: str) -> bool:
        return isinstance(x, str) and x.endswith(f"_{wave}")
    mask = df["source"].apply(is_wave_id) & df["target"].apply(is_wave_id)
    dfw = df[mask].copy()
    dfw_rev = dfw.rename(columns={"source": "target", "target": "source"})
    df_all = pd.concat([dfw, dfw_rev], axis=0, ignore_index=True)
    return df_all[["source", "target", "weight"]]

def load_disease_sim_edges(path: str) -> pd.DataFrame: #질병-질병
    df = pd.read_csv(path)
    df["source"] = df["source"].str.strip()
    df["target"] = df["target"].str.strip()

    # 질병 이름을 내부 이름으로 바꾸기
    df["source"] = df["source"].replace(DISEASE_NAME_MAP)
    df["target"] = df["target"].replace(DISEASE_NAME_MAP)

    df = df[df["source"].isin(DISEASE_LIST) & df["target"].isin(DISEASE_LIST)].copy()
    df_rev = df.rename(columns={"source": "target", "target": "source"})
    df_all = pd.concat([df, df_rev], axis=0, ignore_index=True)
    df_all = df_all.drop_duplicates(subset=["source", "target"])
    return df_all[["source", "target", "weight"]]

def load_has_disease_edges(path: str, wave: int) -> pd.DataFrame: #사람-질병
    df = pd.read_csv(path)
    if "disease" in df.columns:
        df["disease"] = df["disease"].str.strip()
        # 질병 이름을 내부 이름으로 바꾸기
        df["disease"] = df["disease"].replace(DISEASE_NAME_MAP)
    df = df[df["person_wave_id"].apply(lambda x: isinstance(x, str) and x.endswith(f"_{wave}"))].copy()
    df = df[df["disease"].isin(DISEASE_LIST)]
    return df[["person_wave_id", "disease"]]

def build_node_id_maps(pp_df: pd.DataFrame, pd_df: pd.DataFrame, dd_df: pd.DataFrame) -> Tuple[Dict[str, int], Dict[str, int]]:
    #문자열 ID를 컴퓨터가 다룰 수 있는 0, 1, 2... 와 같은 숫자로 변환 (ID 맵핑이라 부르는듯)
    person_ids = set(pp_df["source"]).union(set(pp_df["target"])).union(set(pd_df["person_wave_id"]))
    person_id_map = {pid: i for i, pid in enumerate(sorted(person_ids))}
    disease_ids = set(dd_df["source"]).union(set(dd_df["target"])).union(set(pd_df["disease"]))
    disease_id_map = {did: i for i, did in enumerate(sorted(disease_ids))}
    return person_id_map, disease_id_map

def build_hetero_graph_for_wave(settings: Settings, wave: int):
    # pp_df (person–person 유사도 엣지)
    # dd_df (disease–disease 연관 엣지)
    # pd_df (person–disease 보유 엣지)
    # person_id_map (문자열 → 정수 typed id)
    # disease_id_map (문자열 → 정수 typed id)
    pp_df = load_person_sim_edges(settings.PERSON_SIM_FILE, wave)
    dd_df = load_disease_sim_edges(settings.DISEASE_SIM_FILE)
    pd_df = load_has_disease_edges(settings.HAS_DISEASE_FILE, wave)
    person_id_map, disease_id_map = build_node_id_maps(pp_df, pd_df, dd_df) #build_node_id_maps 함수를 통해
    pp_src = [person_id_map[s] for s in pp_df["source"]]
    pp_dst = [person_id_map[t] for t in pp_df["target"]]
    dd_src = [disease_id_map[s] for s in dd_df["source"]]
    dd_dst = [disease_id_map[t] for t in dd_df["target"]]
    pd_src = [person_id_map[p] for p in pd_df["person_wave_id"]]
    pd_dst = [disease_id_map[d] for d in pd_df["disease"]]
    num_person = len(person_id_map)
    num_disease = len(disease_id_map)
    data_dict = {
        ("person", "similar_to", "person"): (torch_long(pp_src), torch_long(pp_dst)),
        ("disease", "related_to", "disease"): (torch_long(dd_src), torch_long(dd_dst)),
        ("person", "has_disease", "disease"): (torch_long(pd_src), torch_long(pd_dst)),
    }
    g_hetero = dgl.heterograph(
        data_dict,
        num_nodes_dict={"person": num_person, "disease": num_disease},
    )
    # person_id_map/disease_id_map으로 로컬 ID(typed id)
    return g_hetero, pp_df, dd_df, pd_df, person_id_map, disease_id_map

# -----------------------------
# 피처 생성 (wave별 동적)
# -----------------------------

def ensure_person_wave_id_column(df: pd.DataFrame):
    if "person_wave_id" not in df.columns:
        if "hhidpn" in df.columns and "wave" in df.columns:
            df["person_wave_id"] = df["hhidpn"].astype(str) + "_" + df["wave"].astype(str)
        else:
            raise ValueError("CSV_FILE must contain 'person_wave_id' or both 'hhidpn' and 'wave'.")

def build_features_person_dynamic(csv_path: str, wave: int, person_id_map: Dict[str, int]) -> np.ndarray:
    df = pd.read_csv(csv_path)
    ensure_person_wave_id_column(df)
    df = df[df["person_wave_id"].isin(person_id_map.keys())].copy()
    df = df.set_index("person_wave_id")
    feat_parts = []
    for var in NUMERIC_VARS:
        col = next((c for c in [f"r{wave}{var}", var] if c in df.columns), None)
        if col is None: continue
        s = zscore(df[col], name=col).fillna(0.0)
        feat_parts.append(s.to_frame(name=col))
    for var in BINARY_VARS:
        col = next((c for c in [f"r{wave}{var}", var] if c in df.columns), None)
        if col is None: continue
        s = (pd.to_numeric(df[col], errors="coerce").fillna(0.0) > 0.5).astype(float)
        feat_parts.append(s.to_frame(name=col))
    for var in CATEGORICAL_VARS:
        col = next((c for c in [f"r{wave}{var}", var] if c in df.columns), None)
        if col is None: continue
        dummies = pd.get_dummies(df[col].astype(str).fillna("NA"), prefix=col)
        feat_parts.append(dummies)
    if not feat_parts:
        return np.zeros((len(person_id_map), 1), dtype=np.float32)
    feats_df = pd.concat(feat_parts, axis=1)
    feats_df = feats_df.reindex(index=sorted(person_id_map, key=person_id_map.get), fill_value=0.0)
    return feats_df.values.astype(np.float32)

def build_features_disease(disease_id_map: Dict[str, int], mode: str = "onehot", emb_dim: int = 16) -> np.ndarray:
    Nd = len(disease_id_map)
    if mode == "onehot":
        return np.eye(Nd, dtype=np.float32)
    else:
        rng = np.random.default_rng(7)
        return rng.normal(0, 0.01, size=(Nd, emb_dim)).astype(np.float32)

# homo-graph
# (수정 제안) 패딩을 수행하지 않고 원본 피처를 그대로 붙이는 함수
def attach_raw_features_and_to_homo(g_hetero, person_X, disease_X):
    # 이제 차원을 맞추지 않습니다.
    g_hetero.nodes["person"].data["feat"] = torch_float(person_X)
    g_hetero.nodes["disease"].data["feat"] = torch_float(disease_X)

    # 이종 그래프를 동종 그래프로 변환
    # to_homogeneous는 다른 타입의 피처 크기가 달라도 ndata에 딕셔너리로 저장해줍니다.
    g_homo = dgl.to_homogeneous(g_hetero, ndata=["feat"], store_type=True)
    
    # g_homo.ndata['feat']는 이제 딕셔너리 형태 {'person': (Np, Dp), 'disease': (Nd, Dd)}
    # 이를 하나의 텐서로 합쳐주는 작업이 모델 forward에서 필요합니다.
    
    g_homo = dgl.add_self_loop(g_homo)
    
    # 각 피처의 원래 차원도 반환해줍니다.
    person_dim = person_X.shape[1]
    disease_dim = disease_X.shape[1]
    
    return g_homo, person_dim, disease_dim

# # hetero-graph
# def attach_features_hetero(g_hetero, person_X, disease_X, in_dim_pad=None):
#     P_dim, D_dim = person_X.shape[1], disease_X.shape[1]
#     in_dim = max(P_dim, D_dim) if in_dim_pad is None else max(in_dim_pad, P_dim, D_dim)
#     # 두 타입 입력 차원 맞추기(제로 패딩)
#     person_X = pad_right_with_zeros(person_X, in_dim)
#     disease_X = pad_right_with_zeros(disease_X, in_dim)
#     g_hetero.nodes["person"].data["feat"]  = torch_float(person_X)
#     g_hetero.nodes["disease"].data["feat"] = torch_float(disease_X)
#     return g_hetero, in_dim

def build_homo_nid_maps(g_homo: dgl.DGLGraph,
                        g_hetero: dgl.DGLHeteroGraph) -> Dict[str, torch.Tensor]:
    # g_homo.ndata["_TYPE"] = 노드가 원래 어떤 ntype인지 나타내는 정수 라벨
    # g_homo.ndata["_ID"] = 위에 type id가 원래 타입을 나타낸다면 로컬 id는 그 타입에서 몇 번 노드였는지
    type_ids, orig_ids = g_homo.ndata["_TYPE"], g_homo.ndata["_ID"]
    homo_ids = torch.arange(g_homo.num_nodes(), dtype=torch.long)
    maps = {}
    for i, ntype in enumerate(g_hetero.ntypes):
        mask = (type_ids == i)
        table = torch.empty(g_hetero.num_nodes(ntype), dtype=torch.long)
        table[orig_ids[mask]] = homo_ids[mask]
        maps[ntype] = table
    return maps

# -----------------------------
# 라벨셋(9+1) 구성
# -----------------------------

def attach_typed_ids_to_pd(pd_df: pd.DataFrame,
                           person_id_map: Dict[str, int],
                           disease_id_map: Dict[str, int]) -> pd.DataFrame:
    return pd.DataFrame({
        "person_wave_id": pd_df["person_wave_id"], "disease": pd_df["disease"],
        "person_typed_id": pd_df["person_wave_id"].map(person_id_map),
        "disease_typed_id": pd_df["disease"].map(disease_id_map),
    })

def load_data_for_wave(settings: Settings, wave: int):
    """특정 wave의 데이터와 그래프를 미리 로드하는 함수 (프로젝션 모델용)"""
    print(f"Loading and preprocessing data for wave {wave}...")
    g_hetero, _, _, pd_df, person_id_map, disease_id_map = build_hetero_graph_for_wave(settings, wave)

    # 1. 원본 피처 생성 (패딩 X)
    person_X = build_features_person_dynamic(settings.CSV_FILE, wave, person_id_map)
    disease_X = build_features_disease(disease_id_map, mode=settings.disease_init, emb_dim=settings.disease_emb_dim)

    # 2. 그래프의 '구조'만 먼저 동종 그래프로 변환. (ndata 인자 제거)
    g_homo = dgl.to_homogeneous(g_hetero)
    g_homo = dgl.add_self_loop(g_homo)

    # 3. HeteroProjectionGNN에 입력할 통합 피처 텐서를 '수동으로' 생성.
    num_nodes = g_homo.num_nodes()
    ntypes = g_homo.ndata['_TYPE']
    nids = g_homo.ndata['_ID']
    
    unified_dim = max(person_X.shape[1], disease_X.shape[1])
    hg_features = torch.zeros(num_nodes, unified_dim, dtype=torch.float32)

    # --------------------------------------------------------------------------
    # 각 노드 타입에 해당하는 피처를 가져와 hg_features 텐서에 채워 넣습니다.
    # 이 과정에서 g_hetero의 ntypes 순서와 g_homo의 _TYPE ID 순서가 일치해야 합니다.
    for i, ntype in enumerate(g_hetero.ntypes):
        mask = (ntypes == i)
        orig_ids = nids[mask]
        
        # 타입에 맞는 원본 피처 텐서 선택
        original_features = torch_float(person_X) if ntype == 'person' else torch_float(disease_X)
        
        # 통합 텐서의 앞부분에 피처를 채워넣기
        hg_features[mask, :original_features.shape[1]] = original_features[orig_ids]
    
    # 4. 수동으로 만든 통합 피처 텐서를 동종 그래프에 'feat'라는 이름으로 붙여줍니다.
    g_homo.ndata['feat'] = hg_features

    maps = build_homo_nid_maps(g_homo, g_hetero)
    pd_with_ids = attach_typed_ids_to_pd(pd_df, person_id_map, disease_id_map)
    
    preprocessed_data = {
        "g_homo": g_homo,
        "g_hetero": g_hetero,
        "features": hg_features, # 모델에 입력될 통합 피처 텐서
        "person_dim": person_X.shape[1],
        "disease_dim": disease_X.shape[1],
        "maps": maps,
        "pd_with_ids": pd_with_ids,
        "disease_id_map": disease_id_map
    }
    return preprocessed_data

    # --------------------------------------------------------------------------

# 특정 질병을 고정하고 Train/Test 셋을 나누는 함수
def split_data_by_fixed_disease(
    pd_with_ids: pd.DataFrame,
    person_h2h: torch.Tensor,
    disease_h2h: torch.Tensor,
    fixed_disease_idx: int,
    settings: Settings,
    disease_id_map: Dict[str, int]          # <<< 추가
) -> Tuple[Tuple, torch.Tensor, Tuple, torch.Tensor, Dict]:
    
    # 특정 '고정 질병'을 가진 복합질환자를 대상으로 데이터 분할
    # 고정 질병을 가진 복합질환자의 경우:
    # '고정 질병' 엣지는 Train Positive로 사용하지 않음
    # 나머지 다른 질병 엣지들은 Test Positive로 사용
    # 그 외 모든 Positive 엣지들은 학습을 위한 구조만 남김
    print(f"Splitting data, fixing disease_idx: {fixed_disease_idx} ({IDX_TO_DISEASE[fixed_disease_idx]})")

    # --- 1. 복합질환자(multimorbidity) 정의 ---
    person_disease_counts = pd_with_ids.groupby("person_typed_id").size()
    multimorbid_persons_ids = person_disease_counts[person_disease_counts >= 2].index # 질병 2개 이상자의 ID

    #debug_check_fixed_in_split(pd_with_ids, disease_id_map, fixed_disease_idx, IDX_TO_DISEASE)

    # --- 2. '고정 질병'을 가진 복합질환자 찾기 ---
    # '고정 질병'을 엣지 중 하나로 가진 사람들의 ID
    fixed_name  = IDX_TO_DISEASE[fixed_disease_idx]
    fixed_local = disease_id_map[fixed_name]       # ← NEW

    persons_with_fixed_disease = set(
        pd_with_ids[pd_with_ids["disease_typed_id"] == fixed_local]['person_typed_id']
    )

    # 두 조건을 모두 만족(교집합)하는 최종 타겟 그룹 (고정 질병을 가진 복합질환자)
    target_multimorbid_ids = multimorbid_persons_ids.intersection(persons_with_fixed_disease)

    # --- 3. Test Positive 엣지 구성 ---
    # 타겟 그룹이 가진 '고정 질병 외 다른 질병'들을 모두 후보군으로 선정
    potential_test_pos_mask = (
        pd_with_ids['person_typed_id'].isin(target_multimorbid_ids) &
        (pd_with_ids['disease_typed_id'] != fixed_local)     # ← 여기
    )
    potential_test_pos_df = pd_with_ids[potential_test_pos_mask] # 고정 질병과 함께 있는 질병들만 (고정질병 제외)

    # 3-1 ) 사람 노드 별로 비율을 맞춰서 샘플링 -> edge 비율이 딱 맞지 않을 수 있음
    # 각 환자별로 '고정 질병 외 다른 질병'들 중에서 설정된 비율(test_sampling_ratio)만큼 랜덤 샘플링하여 최종 Test Edge로 확정
    # 만약 환자가 가진 다른 질병이 1개뿐이라 샘플링이 불가능한 경우, 그 1개를 그대로 사용 (min(1, ...))
    # test_pos_df = potential_test_pos_df.groupby('person_typed_id').sample(
    #     frac=settings.test_sampling_ratio,
    #     random_state=settings.seed
    # ).reset_index(drop=True)
    
    # 3-2 ) 전체 edge 통합에서 비율을 맞춰서 샘플링 -> 특정 사람에게 몰릴 수 있음
    # potential_test_pos_df 전체에서 설정된 비율(test_sampling_ratio)만큼 한 번에 랜덤 샘플링
    # test_pos_df = potential_test_pos_df.sample(
    #     frac=settings.test_sampling_ratio,
    #     random_state=settings.seed
    # )
    # test_pos_df 안에는 ratio에 맞춰진 고정 질병과 함께 있는 질병이 들어감(고정 질병 제외)

    # 3-3 ) 절충안 : 전체 edge로 나누지만 모든 환자별로 최소 1개의 test edge를 갖도록
    # 각 환자별로 1개의 엣지를 무작위로 추출하여 '최소 보장' Test 엣지 생성
    guaranteed_edges = potential_test_pos_df.groupby('person_typed_id').sample(
        n=1, random_state=settings.seed
    )
    # 전체 비율에 따라 필요한 총 Test 엣지 개수 계산
    total_required_test_size = int(len(potential_test_pos_df) * settings.test_sampling_ratio)
    # 추가로 샘플링해야 할 엣지 개수 계산
    num_additional_samples = max(0, total_required_test_size - len(guaranteed_edges))
    # '최소 보장' 엣지를 제외한 나머지 엣지 풀(pool) 생성
    remaining_pool = potential_test_pos_df.drop(guaranteed_edges.index)
    # 나머지 풀에서 추가 엣지를 무작위 샘플링
    additional_edges = remaining_pool.sample(
        n=min(num_additional_samples, len(remaining_pool)), 
        random_state=settings.seed
    )
    # '최소 보장' 엣지와 '추가' 엣지를 합쳐 최종 Test set 구성
    test_pos_df = pd.concat([guaranteed_edges, additional_edges])

    # --- 4. Train Positive 엣지 구성 ---
    # Test 엣지로 사용된 것을 제외한 나머지를 Train 엣지로 사용
    train_pos_df = potential_test_pos_df.drop(test_pos_df.index)

    # ===== 여기서 sanity check 두 줄 삽입 =====
    assert (test_pos_df['disease_typed_id'] != fixed_local).all(), \
        "test_pos_df에 fixed 질병(고정 질병)이 섞였습니다."
    assert (train_pos_df['disease_typed_id'] != fixed_local).all(), \
        "train_pos_df에 fixed 질병(고정 질병)이 섞였습니다."

    # --- 5. 평가를 위한 '기저질환' 맵 생성 (Train 엣지 기준) ---
    known_disease_map = train_pos_df.groupby('person_typed_id')['disease_typed_id'].apply(set).to_dict()

    # --- 6. Negative 엣지 생성 및 분할 ---
    all_diseases_tids = np.arange(disease_h2h.shape[0])
    target_persons = np.array(sorted(list(target_multimorbid_ids)))

    all_possible_pairs = pd.MultiIndex.from_product(
        [target_persons, all_diseases_tids],
        names=['person_typed_id', 'disease_typed_id']
    )
    all_possible_df = pd.DataFrame(index=all_possible_pairs).reset_index()

    # 6-1. Negative 후보군 만듦
    merged_neg = all_possible_df.merge(pd_with_ids[['person_typed_id', 'disease_typed_id']], how='left', indicator=True)
    neg_candidates_df = merged_neg.loc[merged_neg['_merge'] == 'left_only', ['person_typed_id', 'disease_typed_id']].copy()
    
    # 6-2. 필요한 Negative 엣지 개수 계산
    num_train_neg = int(len(train_pos_df) * settings.neg_pos_ratio)
    num_test_neg = int(len(test_pos_df) * settings.neg_pos_ratio)
    
    # 6-3. 샘플링 가능 개수 조정
    if num_train_neg + num_test_neg > len(neg_candidates_df):
        print(f"Warning: Not enough negative samples available. Adjusting sample counts.")
        total_pos_edges = len(train_pos_df) + len(test_pos_df)
        train_ratio = len(train_pos_df) / total_pos_edges if total_pos_edges > 0 else 0.5
        num_train_neg = int(len(neg_candidates_df) * train_ratio)
        num_test_neg = len(neg_candidates_df) - num_train_neg

    num_train_neg = min(num_train_neg, len(neg_candidates_df))

    # 6-4. Train/Test Negative 엣지 샘플링
    train_neg_df = neg_candidates_df.sample(n=num_train_neg, random_state=settings.seed)

    # 먼저 정의
    remaining_candidates = neg_candidates_df.drop(train_neg_df.index)

    # 그 다음 필터/샘플링
    test_neg_candidates = remaining_candidates[
        remaining_candidates['disease_typed_id'] != fixed_local
    ].copy()

    remaining_candidates = remaining_candidates[
        remaining_candidates['disease_typed_id'] != fixed_local
        ]
    
    # ==================================================================================
    # ### 방법 1: 질병별 균형 샘플링  ##
    # ==================================================================================
    '''
    # 질병별로 Positive 엣지 수에 비례하여 Negative 엣지를 샘플링
    print("DEBUG: Using Method 1 (Per-Disease Balanced Sampling) for Test Negatives.")
    sampled_test_negs_list = []
    test_pos_counts_per_disease = test_pos_df['disease_typed_id'].value_counts()
    
    for disease_id, pos_count in test_pos_counts_per_disease.items():
        # 해당 질병에 필요한 Negative 샘플 수 계산
        num_neg_to_sample = int(pos_count * settings.neg_pos_ratio)
        
        # 해당 질병의 Negative 후보군 추출
        candidates_for_disease = test_neg_candidates[test_neg_candidates['disease_typed_id'] == disease_id]
        
        # 샘플링할 개수가 후보군보다 많지 않도록 조정
        num_neg_to_sample = min(num_neg_to_sample, len(candidates_for_disease))
        
        if num_neg_to_sample > 0:
            sampled = candidates_for_disease.sample(n=num_neg_to_sample, random_state=settings.seed)
            sampled_test_negs_list.append(sampled)
            
    # 샘플링된 Negative 엣지들을 하나로 합침
    if sampled_test_negs_list:
        test_neg_df = pd.concat(sampled_test_negs_list, ignore_index=True)
    else:
        # 샘플링된 엣지가 없는 경우를 대비한 빈 데이터프레임
        test_neg_df = pd.DataFrame(columns=['person_typed_id', 'disease_typed_id'])
    '''

    # ==================================================================================
    # ### 방법 2: 전체 통합 샘플링 ###
    # ==================================================================================
    print("Using Method 2 (Overall Pooled Sampling) for Test Negatives.")
    # 1. 필요한 Test Negative 엣지 총 개수를 계산합니다.
    #    (이 값은 이미 # 6-2. 에서 num_test_neg로 계산되었습니다.)

    # 2. 샘플링할 개수가 실제 후보군 개수를 넘지 않도록 최종 조정합니다.
    num_neg_to_sample = min(num_test_neg, len(test_neg_candidates))

    # 3. 전체 Test Negative 후보군에서 질병 구분 없이 한번에 랜덤 샘플링합니다.
    if num_neg_to_sample > 0:
        test_neg_df = test_neg_candidates.sample(n=num_neg_to_sample, random_state=settings.seed)
    else:
        test_neg_df = pd.DataFrame(columns=['person_typed_id', 'disease_typed_id'])

    # --- 7. 최종 Train/Test 데이터셋 구성 ---
    # Train Tensors
    train_u_tids = torch_long(np.concatenate([train_pos_df['person_typed_id'], train_neg_df['person_typed_id']]))
    train_v_tids = torch_long(np.concatenate([train_pos_df['disease_typed_id'], train_neg_df['disease_typed_id']]))
    train_u, train_v = person_h2h[train_u_tids], disease_h2h[train_v_tids]
    train_y = torch.cat([torch.ones(len(train_pos_df)), torch.zeros(len(train_neg_df))])

    # Test Tensors
    test_u_tids = torch_long(np.concatenate([test_pos_df['person_typed_id'], test_neg_df['person_typed_id']]))
    test_v_tids = torch_long(np.concatenate([test_pos_df['disease_typed_id'], test_neg_df['disease_typed_id']]))
    test_u, test_v = person_h2h[test_u_tids], disease_h2h[test_v_tids]
    test_y = torch.cat([torch.ones(len(test_pos_df)), torch.zeros(len(test_neg_df))])

    # --- 8. 평가 정보 반환 ---
    eval_info = {
        'test_person_tids': test_u_tids.numpy(),
        'test_disease_tids': test_v_tids.numpy(),
        'known_disease_map': known_disease_map
    }
    # # 생성된 Train/Test 엣지 개수 출력
    # print(f"    Train Edges: {len(train_y)} (Pos: {len(train_pos_df)}, Neg: {len(train_neg_df)})")
    # print(f"    Test Edges : {len(test_y)} (Pos: {len(test_pos_df)}, Neg: {len(test_neg_df)})")
    # print("-" * 30)

    # # Test Positive Edge의 질병별 분포 출력
    # print("    Positive Test Edge Counts by Disease:")
    # disease_counts = test_pos_df['disease_typed_id'].value_counts()
    # if not disease_counts.empty:
    #     for disease_id, count in disease_counts.sort_index().items():
    #         disease_name = IDX_TO_DISEASE.get(disease_id, "Unknown")
    #         print(f"        - {disease_name:<10}: {count} edges")
    # else:
    #     print("        - No positive test edges found.")
    
    # print("-" * 30)


    return (train_u, train_v), train_y, (test_u, test_v), test_y, eval_info

# -----------------------------
# Predictor (엣지 분류기)
# -----------------------------

class MLPPredictor(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, out_dim),
        )
    # def forward(self, g: DGLGraph, z: torch.Tensor, eids: Optional[torch.Tensor] = None, uv=None):
    #     u, v = g.find_edges(eids) if uv is None else uv
    #     return self.mlp(torch.cat([z[u], z[v]], dim=-1))
        # fixed_nid와 기타 키워드는 받아서 그냥 무시
    def forward(self, g: DGLGraph, z: torch.Tensor, eids: Optional[torch.Tensor] = None, uv=None,
                fixed_nid: Optional[int] = None, **kwargs):
        u, v = g.find_edges(eids) if uv is None else uv
        return self.mlp(torch.cat([z[u], z[v]], dim=-1))

class HeteroProjectionGNN(nn.Module):
    """
    각 노드 타입(person, disease)의 피처를 타입별 프로젝션 레이어를 통해
    공통된 hidden_dim으로 맞춘 후, GNN 백본에 전달하는 모델
    """
    def __init__(self, person_in_dim: int, disease_in_dim: int, params: Dict, g_hetero: dgl.DGLHeteroGraph):
        super().__init__()
        model_type = params['model_type']
        hidden_dim = params['hidden_dim']
        
        # 1. 타입별 프로젝션 레이어를 ModuleDict로 관리
        # 각 타입을 hidden_dim으로 매핑합니다.
        self.projectors = nn.ModuleDict({
            'person': nn.Linear(person_in_dim, hidden_dim),
            'disease': nn.Linear(disease_in_dim, hidden_dim)
        })
        
        # 2. GNN 백본은 이제 hidden_dim을 입력으로 받습니다.
        activation, tasks, causal = F.relu, [], False
        if model_type == "gcn":
            self.gnn = GCN(hidden_dim, hidden_dim, hidden_dim, params['n_layers'], activation, params['dropout'], tasks, causal)
        elif model_type == "gat":
            self.gnn = GAT(params['n_layers'], hidden_dim, hidden_dim, hidden_dim, params['gat_heads'], activation, params['dropout'], params['gat_attn_drop'], params['gat_neg_slope'], True, tasks, causal)
        elif model_type == "gin":
            self.gnn = GIN(hidden_dim, hidden_dim, hidden_dim, params['n_layers'], params['gin_mlp_layers'], params['dropout'], tasks, causal, "mean", True)
        else:
            raise ValueError(f"Unknown model_type for HeteroProjectionGNN: {model_type}")
            
        # 3. 노드 타입 이름과 DGL이 부여한 정수 ID를 매핑
        self.ntype_map = {ntype: i for i, ntype in enumerate(g_hetero.ntypes)}
        self.in_dim = hidden_dim # GNN의 입력 차원은 hidden_dim

    def forward(self, g: dgl.DGLGraph, features: torch.Tensor) -> torch.Tensor:
        # 최종 GNN에 입력될, 프로젝션이 완료된 피처 텐서
        projected_feats = torch.zeros(g.num_nodes(), self.in_dim, device=g.device)

        # 2. 타입별로 순회하며 각각의 프로젝터를 적용
        for ntype, projector in self.projectors.items():
            type_id = self.ntype_map[ntype]
            mask = (g.ndata['_TYPE'] == type_id)
            
            # 원본 피처 슬라이싱 (제로 패딩 부분 제거)
            original_dim = projector.in_features
            original_h = features[mask, :original_dim]
            
            # 해당 타입의 프로젝터 통과
            projected_h = projector(original_h)
            
            # 결과 텐서에 저장
            projected_feats[mask] = projected_h
            
        # 3. GNN 백본에 통과시켜 최종 노드 임베딩(z)을 얻음
        z = self.gnn(g, projected_feats)
        return z
    
    def forward_edges(self, g: dgl.DGLGraph, features: torch.Tensor, predictor: nn.Module, uv: Tuple) -> torch.Tensor:
        """ 학습 루프와의 호환성을 위한 엣지 포워드 메소드 """
        z = self.forward(g, features)
        return predictor(g, z, uv=uv)

# -----------------------------
# 모델 생성
# -----------------------------

# def create_model_and_predictor(params: Dict, settings: Settings,
#                                person_in_dim: int, disease_in_dim: int, # 차원 정보를 받도록 수정
#                                g_hetero: dgl.DGLHeteroGraph):
#     activation, tasks, causal = F.relu, [], False
#     if params['model_type'] == "gcn":
#         model = GCN(in_dim, params['hidden_dim'], params['hidden_dim'], params['n_layers'], activation, params['dropout'], tasks, causal)
#     elif params['model_type'] == "gat":
#         model = GAT(params['n_layers'], in_dim, params['hidden_dim'], params['hidden_dim'], params['gat_heads'], activation, params['dropout'], params['gat_attn_drop'], params['gat_neg_slope'], True, tasks, causal)
#     elif params['model_type'] == "gin":
#         model = GIN(in_dim, params['hidden_dim'], params['hidden_dim'], params['n_layers'], params['gin_mlp_layers'], params['dropout'], tasks, causal, "mean", True)
#     elif params['model_type'] == "hgt":
#         raise ValueError(f"The HeteroProjectionGNN is not configured for HGT. Please use gcn, gat, or gin.")
#     else:
#         raise ValueError(f"Unknown model_type: {params['model_type']}")
    
#     # 1. 새로운 HeteroProjectionGNN 모델을 생성
#     model = HeteroProjectionGNN(person_in_dim, disease_in_dim, params, g_hetero)
    
#     # 2. Predictor는 그대로 사용 (입력 차원은 GNN의 hidden_dim)
#     predictor = MLPPredictor(params['hidden_dim'], params['pred_hidden'], 1, params['pred_dropout'])
#     return model, predictor

def create_model_and_predictor(params: Dict, settings: Settings,
                               person_in_dim: int, disease_in_dim: int,
                               g_hetero: dgl.DGLHeteroGraph):
    
    # HGT는 이 설계와 맞지 않으므로 실행되지 않도록 처리
    if params['model_type'] == "hgt":
        raise ValueError(f"The HeteroProjectionGNN is not configured for HGT. Please use gcn, gat, or gin.")

    # 1. 래퍼 모델을 바로 생성합니다.
    model = HeteroProjectionGNN(person_in_dim, disease_in_dim, params, g_hetero)
    
    # 2. Predictor를 생성합니다.
    predictor = MLPPredictor(params['hidden_dim'], params['pred_hidden'], 1, params['pred_dropout'])
    
    return model, predictor

# -----------------------------
# 학습/평가 루프
# -----------------------------

def evaluate_metrics_binary(
    y_true: np.ndarray,
    y_score: np.ndarray,                 # model logit or prob
    threshold: float = 0.5,
    disease_ids: Optional[np.ndarray] = None
) -> Dict[str, float]:
    out = {}
    
    # 1. 전체 성능 지표(Micro) 이름 변경 및 계산
    # 전체 샘플에 대해 계산하는 것은 Micro 평균과 동일합니다.
    try:
        out["auroc_micro"] = float(roc_auc_score(y_true, y_score))
    except Exception:
        out["auroc_micro"] = float("nan")
    try:
        out["auprc_micro"] = float(average_precision_score(y_true, y_score))
    except Exception:
        out["auprc_micro"] = float("nan")

    y_pred = (y_score >= threshold).astype(int)
    out["precision_micro"] = float(precision_score(y_true, y_pred, zero_division=0))
    out["recall_micro"]    = float(recall_score   (y_true, y_pred, zero_division=0))
    out["f1_micro"]        = float(f1_score       (y_true, y_pred, zero_division=0))
    out["accuracy"]        = float(accuracy_score (y_true, y_pred)) # Accuracy는 개념상 하나만 존재

    # 2. 질병별 성능을 평균 낸 Macro 지표 계산
    if disease_ids is not None:
        uniq_diseases = np.unique(disease_ids)
        per_cls_auroc, per_cls_auprc, per_cls_rec, per_cls_f1, per_cls_prec = [], [], [], [], []
        
        for d_id in uniq_diseases:
            mask = (disease_ids == d_id)
            if mask.sum() < 2 or len(np.unique(y_true[mask])) < 2: # 샘플이 적거나 한 클래스만 있으면 스킵
                continue
            
            y_t, y_s = y_true[mask], y_score[mask]
            y_p = (y_s >= threshold).astype(int)
            
            try: per_cls_auroc.append(roc_auc_score(y_t, y_s))
            except Exception: pass
            try: per_cls_auprc.append(average_precision_score(y_t, y_s))
            except Exception: pass
            
            per_cls_prec.append(precision_score(y_t, y_p, zero_division=0))
            per_cls_rec.append(recall_score(y_t, y_p, zero_division=0))
            per_cls_f1.append(f1_score(y_t, y_p, zero_division=0))

        if per_cls_auroc: out["auroc_macro"] = float(np.mean(per_cls_auroc))
        if per_cls_auprc: out["auprc_macro"] = float(np.mean(per_cls_auprc))
        if per_cls_prec: out["precision_macro"] = float(np.mean(per_cls_prec))
        if per_cls_rec: out["recall_macro"] = float(np.mean(per_cls_rec))
        if per_cls_f1: out["f1_macro"] = float(np.mean(per_cls_f1))

    return out

def train_one_fold(g: DGLGraph, feats: torch.Tensor,
                   train_uv: Tuple[torch.Tensor, torch.Tensor], train_y: torch.Tensor,
                   test_uv: Tuple[torch.Tensor, torch.Tensor], test_y: torch.Tensor,
                   params: Dict, settings: Settings, g_hetero,
                   person_dim: int, disease_dim: int, # << 추가
                   progress_desc: Optional[str] = None,
                   eval_info: Optional[Dict] = None) -> Dict[str, Dict[str, float]]:
    device = settings.device
    g = g.to(device)
    feats = feats.to(device)

    # HGT 관련 로직 제거 (이제 불필요)
    # in_dim 계산 로직 제거 (이제 불필요)
    
    train_u, train_v = train_uv[0].to(device), train_uv[1].to(device)
    test_u, test_v = test_uv[0].to(device), test_uv[1].to(device)
    train_y, test_y = train_y.to(device), test_y.to(device)

    # 2. 모델 생성 부분 수정
    model, predictor = create_model_and_predictor(params, settings,
                                                  person_in_dim=person_dim,
                                                  disease_in_dim=disease_dim,
                                                  g_hetero=g_hetero)
    
    model.to(device); predictor.to(device)
    
    opt = torch.optim.AdamW(list(model.parameters()) + list(predictor.parameters()), lr=params['lr'], weight_decay=params['weight_decay'])
    loss_fn = nn.BCEWithLogitsLoss()

    epoch_losses = []
    
    # 미리 CPU로 옮겨두어 반복 계산 방지
    test_y_cpu = test_y.cpu().numpy()

    epoch_iter = tqdm(range(params['max_epochs']), desc=f"Epoch | {progress_desc}", ncols=80, leave=False) if progress_desc else range(params['max_epochs'])

    for epoch in epoch_iter:
        # --- 1. 학습 ---
        model.train(); predictor.train()
        logits = model.forward_edges(g, feats, predictor, uv=(train_u, train_v)).squeeze(-1)
        loss = loss_fn(logits, train_y.float())
        opt.zero_grad(); loss.backward(); opt.step()
        
        epoch_losses.append(loss.item())
        
        # --- 2. 매 에폭마다 테스트 성능 측정 (모니터링용) ---
        model.eval(); predictor.eval()
        with torch.no_grad():
            test_logits = model.forward_edges(g, feats, predictor, uv=(test_u, test_v)).squeeze(-1)
            test_scores = torch.sigmoid(test_logits).cpu().numpy()
            
        try:
            test_auroc = roc_auc_score(test_y_cpu, test_scores)
        except ValueError:
            test_auroc = 0.0

        # --- 3. 진행률 표시줄 업데이트 ---
        if hasattr(epoch_iter, "set_postfix"):
            epoch_iter.set_postfix(loss=loss.item(), test_auroc=test_auroc)

    if hasattr(epoch_iter, "close"): epoch_iter.close()

    # --- 최종 평가는 마지막 에폭 모델로 한 번만 수행 ---
    model.eval(); predictor.eval()
    with torch.no_grad():
        test_logits = model.forward_edges(g, feats, predictor, uv=(test_u, test_v)).squeeze(-1)
        test_scores = test_logits.cpu().numpy()
        
        p = settings.pred_threshold
        eps = 1e-12
        p = min(max(p, eps), 1 - eps)
        logit_thr = math.log(p / (1 - p))
        
        if eval_info:
            final_test_metrics = evaluate_multimorbidity_metrics(
                test_y_cpu, test_scores,
                threshold=logit_thr,
                eval_info=eval_info,
                idx_to_disease_map=IDX_TO_DISEASE
            )
        else:
            final_test_metrics = evaluate_metrics_binary(
                test_y_cpu, test_scores,
                threshold=logit_thr,
                disease_ids=None
            )

        train_logits = model.forward_edges(g, feats, predictor, uv=(train_u, train_v)).squeeze(-1)
        train_scores = train_logits.cpu().numpy()
        
        final_train_metrics = evaluate_metrics_binary(
            train_y.cpu().numpy(), train_scores,
            threshold=logit_thr,
            disease_ids=None
        )
        
    return {"train": final_train_metrics, "test": final_test_metrics, "loss_history": epoch_losses}

def evaluate_metrics_binary(
    y_true: np.ndarray,
    y_score: np.ndarray,                 # model logit or prob
    threshold: float = 0.5,
    disease_ids: Optional[np.ndarray] = None
) -> Dict[str, float]:
    #y_score는 로짓이어도 되고, 확률이어도 됨(roc/auprc는 score만 필요).
    # 전체 스칼라 지표
    # - AUROC/AUPRC는 score 사용(로짓 OK)
    out = {}
    try:
        out["auroc"] = float(roc_auc_score(y_true, y_score))
    except Exception:
        out["auroc"] = float("nan")
    try:
        out["auprc"] = float(average_precision_score(y_true, y_score))
    except Exception:
        out["auprc"] = float("nan")

    y_pred = (y_score >= threshold).astype(int)
    out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    out["recall"]    = float(recall_score   (y_true, y_pred, zero_division=0))
    out["f1"]        = float(f1_score       (y_true, y_pred, zero_division=0))
    out["accuracy"]  = float(accuracy_score (y_true, y_pred))

    # 질병별 집계
    if disease_ids is not None:
        # micro: 모든 샘플 통합 → 이미 계산된 것과 동일(precision/recall/f1/auprc/auroc)
        # macro: 질병별로 산출 후 평균
        uniq = np.unique(disease_ids)
        per_cls_auroc, per_cls_auprc, per_cls_rec, per_cls_f1 = [], [], [], []
        for d in uniq:
            mask = (disease_ids == d)
            if mask.sum() == 0:
                continue
            y_t, y_s = y_true[mask], y_score[mask]
            y_p = (y_s >= threshold).astype(int)
            try:
                per_cls_auroc.append(roc_auc_score(y_t, y_s))
            except Exception:
                pass
            try:
                per_cls_auprc.append(average_precision_score(y_t, y_s))
            except Exception:
                pass
            per_cls_rec.append(recall_score(y_t, y_p, zero_division=0))
            per_cls_f1.append(f1_score(y_t, y_p, zero_division=0))

        if per_cls_auroc:
            out["macro_auroc"] = float(np.mean(per_cls_auroc))
        if per_cls_auprc:
            out["macro_auprc"] = float(np.mean(per_cls_auprc))
        if per_cls_rec:
            out["macro_recall"] = float(np.mean(per_cls_rec))
        if per_cls_f1:
            out["f1_macro"] = float(np.mean(per_cls_f1))

    return out

def evaluate_multimorbidity_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    eval_info: Dict,
    idx_to_disease_map: Dict[int, str]
) -> Dict[str, float]:
    # test_person_tids = eval_info['test_person_tids']
    # known_disease_map = eval_info['known_disease_map']
    
    # Macro 계산을 위해 질병 ID를 evaluate_metrics_binary에 전달
    test_disease_tids = eval_info.get('test_disease_tids') # eval_info에서 질병 ID 가져오기

    # 1. 전체 성능 계산 (Micro와 Macro 모두 포함)
    results = evaluate_metrics_binary(y_true, y_score, threshold=threshold, disease_ids=test_disease_tids)

    # # 2. 기저질환별 성능 계산 (기존 로직 유지)
    # for disease_idx, disease_name in idx_to_disease_map.items():
    #     persons_with_base_disease = {
    #         p_id for p_id, diseases in known_disease_map.items() if disease_idx in diseases
    #     }

    #     if not persons_with_base_disease:
    #         continue

    #     mask = np.isin(test_person_tids, list(persons_with_base_disease))
        
    #     if np.sum(mask) < 2 or len(np.unique(y_true[mask])) < 2:
    #         continue

    #     y_true_subset, y_score_subset = y_true[mask], y_score[mask]

    #     try:
    #         results[f"auroc_given_{disease_name}"] = float(roc_auc_score(y_true_subset, y_score_subset))
    #     except ValueError:
    #         results[f"auroc_given_{disease_name}"] = float("nan")
        
    #     y_pred_subset = (y_score_subset >= threshold).astype(int)
    #     results[f"f1_given_{disease_name}"] = float(f1_score(y_true_subset, y_pred_subset, zero_division=0))
    #     results[f"recall_given_{disease_name}"] = float(recall_score(y_true_subset, y_pred_subset, zero_division=0))

    return results
# -----------------------------
# K-Fold 교차 검증 실행
# -----------------------------
def run_cross_validation(params: Dict, settings: Settings, wave: int, fixed_disease_idx: int, preprocessed_data: Dict) -> Dict[str, float]:
    #주어진 하이퍼파라미터로 K-Fold 교차 검증을 수행하고 결과를 집계합니다.

    all_fold_train_metrics = [] # 성능을 저장할 리스트
    all_fold_test_metrics = []

    # 각 Fold의 Loss 기록을 저장할 리스트
    all_fold_loss_histories = []

    disease_name = IDX_TO_DISEASE.get(fixed_disease_idx, "Unknown")
    print(f"\nRunning {settings.n_folds}-Fold Cross-Validation for Fixed Disease: {disease_name.upper()}...")

    for fold in range(settings.n_folds):
        # 각 fold마다 다른 random seed를 사용하여 데이터 분할을 다르게 합니다.
        fold_seed = settings.seed + fold
        
        # replace를 사용하여 settings 객체의 복사본을 만들고 seed 값만 교체
        fold_settings = replace(settings, seed=fold_seed)
        
        # 이제 metrics_dict는 train과 test 키를 모두 가짐
        metrics_dict = run_experiment(params, fold_settings, wave=wave, fixed_disease_idx=fixed_disease_idx, preprocessed_data=preprocessed_data, fold_num=fold + 1)
        
        if "status" not in metrics_dict.get("test", {}):
            # train_one_fold에서 loss_history를 반환하도록 수정했다면 아래 코드를 활성화
            # if "loss_history" in metrics_dict:
            #     all_fold_loss_histories.append(metrics_dict["loss_history"])
            all_fold_train_metrics.append(metrics_dict["train"]) # Train 결과 저장
            all_fold_test_metrics.append(metrics_dict["test"])
    
    if not all_fold_test_metrics:
        return {"status": "all_folds_skipped"}
    
    # 각 fold에서 얻은 결과들을 집계
    def aggregate(fold_metrics):
        aggregated = {}
        if not fold_metrics: return aggregated
        # metric_keys = fold_metrics[0].keys() # 원본
        metric_keys = {k for m in fold_metrics for k in m.keys()} # 수정
        for key in metric_keys:
            values = [m[key] for m in fold_metrics if key in m and not np.isnan(m[key])]
            if values:
                aggregated[f"{key}_mean"] = np.mean(values)
                aggregated[f"{key}_std"] = np.std(values)
        return aggregated

    aggregated_train_metrics = aggregate(all_fold_train_metrics)
    aggregated_test_metrics = aggregate(all_fold_test_metrics)

    # --- Loss 그래프 생성 및 저장 로직 ---
    if all_fold_loss_histories:
        plt.figure(figsize=(10, 6))
        for i, loss_hist in enumerate(all_fold_loss_histories):
            plt.plot(loss_hist, label=f'Fold {i+1} Loss')
        
        # 평균 Loss 곡선 추가
        if len(all_fold_loss_histories) > 1:
            mean_losses = np.mean(np.array(all_fold_loss_histories), axis=0)
            plt.plot(mean_losses, label='Average Loss', color='black', linestyle='--', linewidth=2)

        plt.title(f"Training Loss Curve for {params['model_type'].upper()} (Fixed Disease: {disease_name.upper()})")
        plt.xlabel("Epoch")
        plt.ylabel("BCEWithLogitsLoss")
        plt.legend()
        plt.grid(True)
        
        # 그래프를 이미지 파일로 저장
        plot_filename = f"loss_curve_{params['model_type']}_{disease_name}_wave{wave}.png"
        plt.savefig(plot_filename)
        plt.close() # 메모리 해제를 위해 그래프 창을 닫음
        print(f"  - Saved loss curve plot to: {plot_filename}")
    # ---------------------------------------------
    
    print(f"\n--- CV Average Results for {disease_name.upper()} ---")
    
    # Train 성능은 AUROC만 간결하게 출력
    train_auroc = aggregated_train_metrics.get('auroc_mean', float('nan'))
    print(f"  - Train AUROC (for diagnostics): {train_auroc:.4f}")
            
    # Test 성능은 모든 지표를 자세히 출력
    print("\n  [Test Set Performance]")
    mean_keys = sorted([k for k in aggregated_test_metrics if k.endswith("_mean")])
    for key in mean_keys:
        print(f"    - {key:<30}: {aggregated_test_metrics[key]:.4f}")
    
    # 최종 파일 저장을 위해 Test 결과 전체만 반환
    return aggregated_test_metrics

# -----------------------------
# 실험/CV 실행 (wave별)
# -----------------------------

def run_experiment(params: Dict, settings: Settings, wave: int, fixed_disease_idx: int, preprocessed_data: Dict, fold_num: int) -> Dict[str, Dict[str, float]]:
    # 각 Fold에서 호출될 때마다 새로운 시드를 설정
    set_seed(settings.seed)
    disease_name = IDX_TO_DISEASE.get(fixed_disease_idx, "Unknown")
    
    # 더 이상 데이터를 직접 로드하지 않고, 미리 처리된 데이터를 사용
    g_homo = preprocessed_data["g_homo"]
    maps = preprocessed_data["maps"]
    pd_with_ids = preprocessed_data["pd_with_ids"]
    person_h2h, disease_h2h = maps["person"], maps["disease"]
    g_hetero = preprocessed_data["g_hetero"] # 이종 그래프 가져오기
    
    # 주어진 fixed_disease_idx로 데이터 분할
    train_uv, train_y, test_uv, test_y, eval_info = split_data_by_fixed_disease(
        pd_with_ids, person_h2h, disease_h2h,
        fixed_disease_idx=fixed_disease_idx,
        settings=settings,
        disease_id_map=preprocessed_data["disease_id_map"]
    )
    
    # 1. 엣지 분포 계산 및 화면 출력
    print(f"--- [Fold {fold_num}] Test Set Edge Distribution ---")
    labels = test_y.numpy()
    disease_ids = eval_info['test_disease_tids']
    
    total_pos = int(np.sum(labels == 1))
    total_neg = int(np.sum(labels == 0))
    print(f"  - Total Edges: {len(labels)} (Positive: {total_pos}, Negative: {total_neg})")

    # 2. 질병별 분포 딕셔너리 생성 (한 번에)
    inv_disease_map = {v: k for k, v in preprocessed_data["disease_id_map"].items()}
    per_disease_counts = {}

    unique_diseases = np.unique(disease_ids)
    for tid in sorted(unique_diseases):
        mask = (disease_ids == tid)
        pos = int(np.sum(labels[mask] == 1))
        neg = int(np.sum(labels[mask] == 0))
        name = inv_disease_map.get(int(tid), f"tid:{int(tid)}")
        per_disease_counts[name] = (pos, neg)

    # 3. 화면 출력
    for name in sorted(per_disease_counts.keys()):
        pos, neg = per_disease_counts[name]
        print(f"    - Disease '{name}': {pos} Positive, {neg} Negative")

    print("-" * 45)

    # 3. 위에서 만든 로그 함수 호출하여 파일에 저장
    log_edge_distribution(
        log_file_path=settings.EDGE_DIST_LOG_FILE,
        wave=wave,
        fixed_disease_name=disease_name,
        fold_num=fold_num,
        total_pos=total_pos,
        total_neg=total_neg,
        per_disease_counts=per_disease_counts
    )

    if (test_y > 0.5).sum() == 0:
        print(f"--- Skipping fold for fixed disease '{disease_name}': No positive edges in test set. ---")
        return {"status": "skipped_no_test_positives"}

    # Transductive 설정: 테스트 엣지를 동종 그래프 구조에서 제거
    pos_mask_in_test = (test_y > 0.5)
    pos_test_u = test_uv[0][pos_mask_in_test]
    pos_test_v = test_uv[1][pos_mask_in_test]
    g_train_homo = dgl.remove_edges(g_homo, g_homo.edge_ids(pos_test_u, pos_test_v)) if len(pos_test_u) > 0 else g_homo

    # 이제 모델 타입에 관계없이 동일한 그래프와 피처를 사용
    graph_to_train = g_train_homo
    # 1. preprocessed_data에서 통합 피처 텐서를 가져옴
    features_to_train = preprocessed_data["features"]

    # 2. train_one_fold 호출 시 person_dim과 disease_dim을 추가로 전달
    metrics_dict = train_one_fold(
        graph_to_train, features_to_train,
        train_uv, train_y, test_uv, test_y,
        params, settings,
        g_hetero=g_hetero,
        person_dim=preprocessed_data["person_dim"],   # << 추가
        disease_dim=preprocessed_data["disease_dim"], # << 추가
        progress_desc=f"Fold Execution",
        eval_info=eval_info
    )

    return metrics_dict
# -----------------------------
# Optuna Objective 함수
# -----------------------------


# -----------------------------
# 메인
# -----------------------------
def main():
    settings = Settings()
    ensure_dir(settings.RESULTS_FILE)

    if os.path.exists(settings.EDGE_DIST_LOG_FILE):
        os.remove(settings.EDGE_DIST_LOG_FILE)

    set_seed(settings.seed)

    FIXED_HYPERPARAMS = {
        "gcn": {"model_type": "gcn", "lr": 0.002, "hidden_dim": 128, "n_layers": 3, "dropout": 0.3, "weight_decay": 5e-4, "pred_hidden": 128, "pred_dropout": 0.3, "max_epochs": 150},
        "gat": {"model_type": "gat", "lr": 0.002, "hidden_dim": 128, "n_layers": 2, "dropout": 0.3, "weight_decay": 5e-4, "pred_hidden": 128, "pred_dropout": 0.3, "max_epochs": 150, "gat_heads": [4, 1], "gat_attn_drop": 0.1, "gat_neg_slope": 0.2},
        "gin": {"model_type": "gin", "lr": 0.002, "hidden_dim": 128, "n_layers": 3, "dropout": 0.3, "weight_decay": 5e-3, "pred_hidden": 128, "pred_dropout": 0.3, "max_epochs": 150, "gin_mlp_layers": 2},
    }

    all_results = {}
    model_types_to_run = ["gcn", "gat", "gin"] # 실행할 모델 선택

    for wave in settings.waves:
        print(f"\n{'#'*20} STARTING EXPERIMENTS FOR WAVE: {wave} {'#'*20}")
        
        # Wave별 데이터 미리 로딩
        preprocessed_data = load_data_for_wave(settings, wave)
        all_results[f"wave_{wave}"] = {} # wave별 결과를 저장할 딕셔너리 생성

        for model_type in model_types_to_run:
            all_results[f"wave_{wave}"][model_type] = {}

            params = FIXED_HYPERPARAMS[model_type]
            
            for disease_idx, disease_name in IDX_TO_DISEASE.items():
                print(f"\n\n{'='*20} {model_type.upper()} | Fixed Disease: {disease_name.upper()} | Wave: {wave} {'='*20}")
                print("Using fixed hyperparameters:")
                for p_name, p_val in params.items():
                    print(f"  - {p_name}: {p_val}")

                cv_metrics = run_cross_validation(
                    params, settings, 
                    wave=wave, 
                    fixed_disease_idx=disease_idx,
                    preprocessed_data=preprocessed_data
                )

                all_results[f"wave_{wave}"][model_type][disease_name] = {
                    "fixed_params": params, # 'best_params' 대신 'fixed_params'로 저장
                    "results": cv_metrics
                }

    print("\n--- Saving results to a text file ---")
    with open(settings.RESULTS_FILE, "w", encoding="utf-8") as f:
        for wave_key, wave_data in all_results.items():
            f.write(f"\n\n{'#'*40}\n")
            f.write(f"### RESULTS FOR {wave_key.upper()} ###\n")
            f.write(f"{'#'*40}\n")
            for model_type, disease_data in wave_data.items():
                f.write(f"\n{'='*30}\n")
                f.write(f"   MODEL: {model_type.upper()}\n")
                f.write(f"{'='*30}\n\n")

                for disease_name, data in disease_data.items():
                    f.write(f"--- Scenario: Fixed Disease '{disease_name.upper()}' ---\n")
                    f.write("  - Fixed Hyperparameters Used:\n")
                    for param, value in data['fixed_params'].items(): # 'best_params' -> 'fixed_params'
                        f.write(f"    - {param:<15}: {value}\n")
                    
                    f.write("\n  - Final CV Test Metrics (Mean ± Std Dev):\n")
                    metrics = data['results']
                    if "status" in metrics:
                        f.write(f"    - Status: {metrics['status']}\n")
                    else:
                        mean_keys = sorted([k for k in metrics.keys() if k.endswith("_mean") and "_given_" not in k])
                        for key in mean_keys:
                            mean_val = metrics[key]
                            std_key = key.replace("_mean", "_std")
                            std_val = metrics.get(std_key, 0.0)
                            metric_name = key[:-5] 
                            f.write(f"    - {metric_name:<25}: {mean_val:.4f} ± {std_val:.4f}\n")
                    f.write("\n" + "-"*50 + "\n\n")

    print(f"Saved all model-specific and disease-specific results to: {settings.RESULTS_FILE}")

if __name__ == "__main__":
    main()