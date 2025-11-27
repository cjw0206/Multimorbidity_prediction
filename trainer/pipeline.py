from typing import Dict

import numpy as np
import dgl

import matplotlib.pyplot as plt
from dataclasses import replace

# from dataloader.splits import split_data_by_fixed_disease
# from dataloader.loader import load_data_for_wave
from dataloader import split_data_by_fixed_disease
from dataloader import load_data_for_wave

from core.config import Settings, IDX_TO_DISEASE
from core.utils import set_seed, log_edge_distribution


# from .loops import train_one_fold
from .loops_best_epoch import train_one_fold

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
    # print(f"--- [Fold {fold_num}] Test Set Edge Distribution ---")
    labels = test_y.numpy()
    disease_ids = eval_info['test_disease_tids']
    
    total_pos = int(np.sum(labels == 1))
    total_neg = int(np.sum(labels == 0))
    # print(f"  - Total Edges: {len(labels)} (Positive: {total_pos}, Negative: {total_neg})")

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
    # for name in sorted(per_disease_counts.keys()):
    #     pos, neg = per_disease_counts[name]
    #     print(f"    - Disease '{name}': {pos} Positive, {neg} Negative")

    # print("-" * 45)

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