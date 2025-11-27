from typing import Tuple, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl import DGLGraph

from core.config import (
    Settings, DISEASE_LIST, DISEASE_NAME_MAP,
    NUMERIC_VARS, BINARY_VARS, CATEGORICAL_VARS
)
from core.utils import (
    zscore, pad_right_with_zeros, torch_long, torch_float
)

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
# 라벨셋 구성
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
