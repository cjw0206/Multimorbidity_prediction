from typing import Tuple, Dict

import numpy as np
import pandas as pd
import torch

from core.config import Settings, IDX_TO_DISEASE
from core.utils import torch_long

# -----------------------------
# train/test, positive/negative 분리
# -----------------------------

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