import os
import random
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

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

# === split 단계 전역/로컬 비교 결과를 나란히 출력 ===
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