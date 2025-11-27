import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import itertools
import json
import warnings
warnings.filterwarnings("ignore")

from core.config import Settings, IDX_TO_DISEASE
from core.utils import ensure_dir
from dataloader.loader import load_data_for_wave
from trainer import run_cross_validation  # __init__에서 export 가정

# ===========================================
# 고정 하이퍼파라미터 (기본 설정)
# ===========================================
FIXED_HYPERPARAMS = {
    "gcn": {"model_type": "gcn", "lr": 0.002, "hidden_dim": 128, "n_layers": 3, "dropout": 0.3,
            "weight_decay": 5e-4, "pred_hidden": 128, "pred_dropout": 0.3,
            "max_epochs": 150, "using_moe": False},

    "gat": {"model_type": "gat", "lr": 0.002, "hidden_dim": 128, "n_layers": 2, "dropout": 0.3,
            "weight_decay": 5e-4, "pred_hidden": 128, "pred_dropout": 0.3,
            "max_epochs": 150, "num_heads": 4, "gat_attn_drop": 0.1,
            "gat_neg_slope": 0.2, "using_moe": False},

    "gin": {"model_type": "gin", "lr": 0.002, "hidden_dim": 128, "n_layers": 3, "dropout": 0.3,
            "weight_decay": 5e-3, "pred_hidden": 128, "pred_dropout": 0.3,
            "max_epochs": 150, "gin_mlp_layers": 2, "using_moe": False},

    "gin_moe": {"model_type": "gin_moe", "lr": 0.002, "hidden_dim": 128, "n_layers": 3, "dropout": 0.3,
                "weight_decay": 5e-3, "pred_hidden": 128, "pred_dropout": 0.3,
                "max_epochs": 150, "gin_mlp_layers": 2, "using_moe": True},
    
    "gat_moe": {"model_type": "gat_moe", "lr": 0.002, "hidden_dim": 128, "n_layers": 2, "dropout": 0.3,
            "weight_decay": 5e-4, "pred_hidden": 128, "pred_dropout": 0.3,
            "max_epochs": 150, "num_heads": 4, "gat_attn_drop": 0.1,
            "gat_neg_slope": 0.2, "using_moe": True},

    "gcn_moe": {"model_type": "gcn_moe", "lr": 0.002, "hidden_dim": 128, "n_layers": 3, "dropout": 0.3,
                "weight_decay": 5e-4, "pred_hidden": 128, "pred_dropout": 0.3,
                "max_epochs": 150, "using_moe": True},

    "multi_graph": {"model_type": "multi_graph", "lr": 0.002, "hidden_dim": 128, "n_layers": 3, "dropout": 0.3,
                    "weight_decay": 5e-4, "pred_hidden": 128, "pred_dropout": 0.3,
                    "max_epochs": 150, "num_heads": 4, "gat_attn_drop": 0.1,
                    "gat_neg_slope": 0.2, "gin_mlp_layers": 2, "using_moe": True},
    "multi_graph_pred_moe": {"model_type": "multi_graph_pred_moe", "lr": 0.002, "hidden_dim": 128, "n_layers": 3, "dropout": 0.3,
                    "weight_decay": 5e-4, "pred_hidden": 128, "pred_dropout": 0.3,
                    "max_epochs": 150, "num_heads": 4, "gat_attn_drop": 0.1,
                    "gat_neg_slope": 0.2, "gin_mlp_layers": 2, "using_moe": True, "top_k": 1},

    "multi_graph_moe_fuse": {"model_type": "multi_graph_moe_fuse", "lr": 0.002, "hidden_dim": 128, "n_layers": 3, "dropout": 0.3,
                    "weight_decay": 5e-4, "pred_hidden": 128, "pred_dropout": 0.3,
                    "max_epochs": 150, "num_heads": 4, "gat_attn_drop": 0.1,
                    "gat_neg_slope": 0.2, "gin_mlp_layers": 2, "using_moe": True},

    "graphormer": {"model_type": "graphormer", "batch_size": 512, "lr": 0.002, "hidden_dim": 32, "n_layers": 1,
                    "dropout": 0.3, "weight_decay": 5e-4, "pred_hidden": 128, "pred_dropout": 0.3,
                    "max_epochs": 150, "num_heads": 4, "using_moe": False},

    "graphormer_moe": {"model_type": "graphormer_moe", "batch_size": 512, "lr": 0.002, "hidden_dim": 32, "n_layers": 2,
                        "dropout": 0.3, "weight_decay": 5e-4, "pred_hidden": 128, "pred_dropout": 0.3,
                        "max_epochs": 150, "num_heads": 4, "using_moe": True}
}

# ===========================================
# 하이퍼파라미터 튜닝 범위 설정
# ===========================================
TUNING_GRID = {
    "gcn": {
        "lr": [0.001, 0.002, 0.005],
        "dropout": [0.2, 0.3],
        "n_layers": [2, 3],
        "weight_decay": [5e-4, 1e-3]
    },
    "gat": {
        "lr": [0.001, 0.002],
        "dropout": [0.3, 0.4],
        "n_layers": [2, 3],
        "weight_decay": [5e-4, 5e-3]
    },
    "gin": {
        "lr": [0.001, 0.002, 0.004],
        "dropout": [0.3, 0.4],
        "n_layers": [2, 3, 4],
        "weight_decay": [1e-3, 5e-3]
    },
    "gin_moe": {
        "lr": [0.001],
        "dropout": [0.2],
        "n_layers": [2],
        "weight_decay": [1e-3]
    },
    "gat_moe": {
        "lr": [1e-4, 5e-3, 1e-3],
        "dropout": [0.1, 0.3],
        "n_layers": [2, 3],
        "num_heads": [4, 8, 16],
        "weight_decay": [5e-4, 1e-3]
    },
    "gcn_moe": {
        "lr": [0.001, 0.002],
        "dropout": [0.2, 0.3],
        "n_layers": [2, 3],
        "weight_decay": [5e-4, 1e-3]
    },
    # "multi_graph": {
    #     "lr": [5e-3],
    #     "dropout": [0.3],
    #     "n_layers": [2, 3, 4],
    #     "num_heads": [4, 8, 16],
    #     "weight_decay": [5e-4],
    #     "hidden_dim": [128,256,512]
    # },
    "multi_graph": {
        "lr": [1e-3],
        "dropout": [0.3],
        "n_layers": [3],
        "num_heads": [16],
        "weight_decay": [1e-3]
    },
    "multi_graph_pred_moe": {
        "lr": [2e-5],
        "dropout": [0.3, 0.5],
        "n_layers": [3],
        "hidden_dim": [128, 256,512],
        "weight_decay": [1e-3],
        "pred_hidden": [128, 256],
        "pred_dropout": [0.1, 0.3],
        "top_k": [1,2,3,4]
    },
    "multi_graph_moe_fuse": {
        "lr": [1e-3],
        "dropout": [0.3],
        "n_layers": [3],
        "num_heads": [16],
        "weight_decay": [1e-3]
    },
    "graphormer": {
        "lr": [0.001, 0.002, 0.004],
        "dropout": [0.3, 0.4],
        "n_layers": [1, 2],
        "weight_decay": [5e-4, 1e-3]
    },
    "graphormer_moe": {
    "lr": [0.001, 0.002],
    "dropout": [0.0, 0.1, 0.3],
    "n_layers": [1, 2],
    "hidden_dim": [16],
    "weight_decay": [0, 1e-4, 5e-4]
    }

}

# ===========================================
# 결과 파일 정렬 함수 (AUROC 내림차순)
# ===========================================
def sort_results_file(result_file):
    with open(result_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 데이터 라인만 추출
    data_lines = [
        line for line in lines
        if "\t" in line and "Disease" not in line and "MEAN" not in line
    ]

    parsed = []
    for line in data_lines:
        parts = line.strip().split("\t")
        if len(parts) == 3:
            try:
                parsed.append((parts[0], float(parts[1]), float(parts[2])))
            except:
                continue

    # AUROC 내림차순 정렬
    parsed.sort(key=lambda x: x[1], reverse=True)

    # 정렬된 결과 다시 저장 (헤더 포함)
    with open(result_file, "w", encoding="utf-8") as f:
        f.write("### HYPERPARAMETER TUNING RESULTS (AUTO-SAVED PER COMBINATION, SORTED) ###\n\n")
        f.write("PARAMS\tAUROC\tAUPRC\n")

        for params, auc, auprc in parsed:
            f.write(f"{params}\t{auc:.4f}\t{auprc:.4f}\n")


# ===========================================
# 메인 실행부 (매 조합마다 txt 파일 즉시 저장 + 정렬)
# ===========================================
def main():
    settings = Settings()

    ensure_dir(settings.RESULTS_FILE)
    ensure_dir(settings.EDGE_DIST_LOG_FILE)
    print(f"---------- result file name : {settings.RESULTS_FILE} ----------")

    waves = settings.waves

    # 파일 초기화
    # with open(settings.RESULTS_FILE, "w", encoding="utf-8") as f:
    #     f.write("### HYPERPARAMETER TUNING RESULTS (AUTO-SAVED PER COMBINATION) ###\n\n")

    if not os.path.exists(settings.RESULTS_FILE):
        with open(settings.RESULTS_FILE, "w", encoding="utf-8") as f:
            f.write("### HYPERPARAMETER TUNING RESULTS (AUTO-SAVED PER COMBINATION) ###\n\n")

    for wave in waves:
        pre = load_data_for_wave(settings, wave)
        wave_key = f"wave{wave}"

        # for model_key in ["multi_graph"]:
        for model_key in ["multi_graph_pred_moe"]:
        # for model_key in ["multi_graph_moe_fuse"]:
            base_params = FIXED_HYPERPARAMS[model_key]

            param_grid = TUNING_GRID[model_key]
            keys, values = zip(*param_grid.items())
            param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

            print(f"[Wave {wave}] Start tuning {model_key} ({len(param_combinations)} combinations)\n")

            for combo_idx, combo in enumerate(param_combinations):
                combo_name = "_".join(f"{k}={v}" for k, v in combo.items())
                tuned_params = {**base_params, **combo}

                print(f"Current Combination {combo_idx + 1} / {len(param_combinations)}")
                print(f"[Wave {wave}] {model_key.upper()} - Params: {combo_name}")

                aurocs, auprcs = [], []

                for idx, disease_name in IDX_TO_DISEASE.items():
                    metrics = run_cross_validation(tuned_params, settings, wave, idx, pre)

                    # mean 값 추출
                    mean_keys = sorted(
                        [k for k in metrics.keys() if k.endswith("_mean") and "_given_" not in k]
                    )
                    auroc_val = None
                    auprc_val = None

                    for key in mean_keys:
                        metric_name = key[:-5]
                        if metric_name.lower() == "auroc":
                            auroc_val = metrics.get(key)
                        elif metric_name.lower() == "auprc":
                            auprc_val = metrics.get(key)

                    if auroc_val is not None: aurocs.append(auroc_val)
                    if auprc_val is not None: auprcs.append(auprc_val)

                # 조합별 mean 계산
                if aurocs and auprcs:
                    mean_auroc = sum(aurocs) / len(aurocs)
                    mean_auprc = sum(auprcs) / len(auprcs)

                    # 1) Append
                    with open(settings.RESULTS_FILE, "a", encoding="utf-8") as f:
                        f.write(f"{combo_name}\t{mean_auroc:.4f}\t{mean_auprc:.4f}\n")

                    # 2) Append 후 즉시 파일 전체 정렬
                    sort_results_file(settings.RESULTS_FILE)

            print(f"Completed tuning for {model_key} ({len(param_combinations)} configs).")

    print(f"\nSaved continuously-updated (sorted) results to: {settings.RESULTS_FILE}")



# ===========================================
# 실행 시작점
# ===========================================
if __name__ == "__main__":
    main()