import os; os.environ["PYTORCH_CUDA_ALLOC_CONF"]="max_split_size_mb:128"

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import json
import warnings
warnings.filterwarnings("ignore")

from core.config import Settings, IDX_TO_DISEASE
from core.utils import ensure_dir     
from dataloader.loader import load_data_for_wave
from trainer import run_cross_validation  # __init__에서 export 가정

FIXED_HYPERPARAMS = {
        "gcn": {"model_type": "gcn", "lr": 0.002, "hidden_dim": 128, "n_layers": 3, "dropout": 0.3, "weight_decay": 5e-4, "pred_hidden": 128, "pred_dropout": 0.3, "max_epochs": 150, "using_moe":False},
        "gat": {"model_type": "gat", "lr": 0.0001, "hidden_dim": 128, "n_layers": 2, "dropout": 0.3, "weight_decay": 5e-4, "pred_hidden": 128, "pred_dropout": 0.3, "max_epochs": 150, "num_heads": 4, "gat_attn_drop": 0.1, "gat_neg_slope": 0.2, "using_moe":False},
        "gin": {"model_type": "gin", "lr": 0.002, "hidden_dim": 128, "n_layers": 3, "dropout": 0.3, "weight_decay": 5e-3, "pred_hidden": 128, "pred_dropout": 0.3, "max_epochs": 150, "gin_mlp_layers": 2, "using_moe":False},
        "gin_moe": {"model_type": "gin_moe", "lr": 0.0001, "hidden_dim": 256, "n_layers": 3, "dropout": 0.2, "weight_decay": 1e-3, "pred_hidden": 128, "pred_dropout": 0.3, "max_epochs": 150, "gin_mlp_layers": 2, "using_moe":True},
        "gcn_moe": {"model_type": "gcn_moe", "lr": 0.002, "hidden_dim": 128, "n_layers": 3, "dropout": 0.3, "weight_decay": 5e-4, "pred_hidden": 128, "pred_dropout": 0.3, "max_epochs": 150, "using_moe":True},
        "multi_graph": {"model_type": "multi_graph", "lr": 0.002, "hidden_dim": 128, "n_layers": 3, "dropout": 0.3, "weight_decay": 5e-4, "pred_hidden": 128, "pred_dropout": 0.3, "max_epochs": 150, "num_heads": 4, "gat_attn_drop": 0.1, "gat_neg_slope": 0.2, "gin_mlp_layers": 2, "using_moe":False},
        "multi_graph_pred_moe": {"model_type": "multi_graph_pred_moe", "lr": 2e-5, "hidden_dim": 512, "n_layers": 3, "dropout": 0.3,
                    "weight_decay": 1e-3, "pred_hidden": 256, "pred_dropout": 0.1,
                    "max_epochs": 150, "num_heads": 16, "gat_attn_drop": 0.1,
                    "gat_neg_slope": 0.2, "gin_mlp_layers": 2, "using_moe": True, "top_k": 3},
        "graphormer": {"model_type": "graphormer", "batch_size": 512 ,"lr": 0.002, "hidden_dim": 32, "n_layers": 2, "dropout": 0.3, "weight_decay": 5e-4, "pred_hidden": 128, "pred_dropout": 0.3, "max_epochs": 150, "num_heads" : 4, "using_moe":False},
        "graphormer_moe": {"model_type": "graphormer_moe", "batch_size": 512 ,"lr": 0.002, "hidden_dim": 32, "n_layers": 1, "dropout": 0.3, "weight_decay": 5e-4, "pred_hidden": 128, "pred_dropout": 0.3, "max_epochs": 150, "num_heads" : 4, "using_moe":True}
    }

FIXED = FIXED_HYPERPARAMS

all_results = {}

def main():
    settings = Settings()
    all_results = {}
    ensure_dir(settings.RESULTS_FILE)
    ensure_dir(settings.EDGE_DIST_LOG_FILE)
    print(f"---------- result file name : {settings.RESULTS_FILE} ----------")
    
    waves = settings.waves            # config에서 조정
    for wave in waves:
        pre = load_data_for_wave(settings, wave)
        wave_key = f"wave{wave}"                 
        all_results[wave_key] = {} 

        # for model_key in ["gin_moe"]:
        # for model_key in ["multi_graph"]:
        # for model_key in ["graphormer_moe"]:
        # for model_key in ["gin"]: 
        for model_key in ["multi_graph_pred_moe"]:
            params = FIXED[model_key]
            all_results[wave_key][model_key] = {}

            for idx, name in IDX_TO_DISEASE.items():
                metrics = run_cross_validation(params, settings, wave, idx, pre)
                all_results[wave_key][model_key][name] = {
                    "fixed_params": params,
                    "results": metrics
                }

            # print(json.dumps({f"{model_key}_{wave_key}": {
            #     k: v["results"] for k, v in all_results[wave_key][model_key].items()
            # }}, ensure_ascii=False, indent=2))

    # ---- 저장 ----
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
                ############## 모델 파라미터 ##############
                f.write("  - Fixed Hyperparameters Used:\n")
                # for param, value in data['fixed_params'].items():
                for param, value in disease_data['hibp']['fixed_params'].items():
                    f.write(f"    - {param:<15}: {value}\n")
                f.write(f"\n")
                ############## 모델 파라미터 ##############

                # ✅ 리스트 초기화 (각 질병별 AUROC, AUPRC 수집용)
                aurocs, auprcs = [], []

                for disease_name, data in disease_data.items():
                    f.write(f"--- Scenario: Fixed Disease '{disease_name.upper()}' ---\n")
                    f.write("\n  - Final CV Test Metrics (Mean ± Std Dev):\n")
                    metrics = data['results']
                    if "status" in metrics:
                        f.write(f"    - Status: {metrics['status']}\n")
                    else:
                        # mean_keys = sorted([k for k in metrics.keys() if k.endswith('_mean') and '_given_' not in k])
                        # for key in mean_keys:
                        #     std_key = key.replace('_mean', '_std')
                        #     mean_val = metrics.get(key, float('nan'))
                        #     std_val = metrics.get(std_key, 0.0)
                        #     metric_name = key[:-5]
                        #     f.write(f"    - {metric_name:<25}: {mean_val:.4f} ± {std_val:.4f}\n")
                        #     # ✅ AUROC/AUPRC 값 저장
                        #     if metric_name.lower() == "auroc":
                        #         aurocs.append(mean_val)
                        #     elif metric_name.lower() == "auprc":
                        #         auprcs.append(mean_val)

                        #-------------auroc, auprc만 저장-------------------
                        mean_keys = sorted(
                            [k for k in metrics.keys() 
                             if k.endswith('_mean') and '_given_' not in k]
                        )

                        # 🔹 1) AUROC / AUPRC만 출력
                        for key in mean_keys:
                            std_key = key.replace('_mean', '_std')
                            mean_val = metrics.get(key, float('nan'))
                            std_val = metrics.get(std_key, 0.0)
                            metric_name = key[:-5]  # "_mean" 떼기

                            # auroc / auprc 이외의 metric은 스킵
                            if metric_name.lower() not in ("auroc", "auprc"):
                                continue

                            f.write(f"    - {metric_name:<25}: {mean_val:.4f} ± {std_val:.4f}\n")

                            # ✅ AUROC/AUPRC 값은 요약 표를 위해 따로 저장
                            if metric_name.lower() == "auroc":
                                aurocs.append(mean_val)
                            elif metric_name.lower() == "auprc":
                                auprcs.append(mean_val)

                        # 🔹 2) AUROC 기준 best epoch도 같이 출력
                        be_mean = metrics.get("best_epoch_by_auroc_mean", None)
                        be_std  = metrics.get("best_epoch_by_auroc_std", None)
                        if be_mean is not None and be_std is not None:
                            f.write(f"    - {'best_epoch_by_auroc':<25}: {be_mean:.1f} ± {be_std:.1f}\n")        

                    f.write("\n" + "-"*50 + "\n\n")
                # -----------------------------------------------------------------------------------------------

                # ✅ 각 모델의 질병별 AUROC/AUPRC 요약 + 평균 추가
                if aurocs and auprcs:
                    f.write("\nSUMMARY OF AUROC & AUPRC BY DISEASE\n")
                    f.write("-"*50 + "\n")
                    f.write(f"{'Disease':<25}{'AUROC':>10}{'AUPRC':>10}\n")
                    f.write("-"*50 + "\n")
                    for i, (disease_name, data) in enumerate(disease_data.items()):
                        try:
                            auroc_val = aurocs[i]
                            auprc_val = auprcs[i]
                            f.write(f"{disease_name:<25}{auroc_val:>10.4f}{auprc_val:>10.4f}\n")
                        except IndexError:
                            continue
                    # f.write("-"*50 + "\n")
                    f.write(f"{'Mean':<25}{sum(aurocs)/len(aurocs):>10.4f}{sum(auprcs)/len(auprcs):>10.4f}\n")
                    f.write("="*50 + "\n\n")

    print(f"Saved all model/disease results to: {settings.RESULTS_FILE}")

    # p=argparse.ArgumentParser()
    # p.add_argument("--wave", type=int, default=10)
    # p.add_argument("--model", choices=list(FIXED.keys()), default="gcn")
    # p.add_argument("--disease", type=str)
    # args=p.parse_args()

    # settings = Settings()
    # pre = load_data_for_wave(settings, args.wave)
    # params = FIXED[args.model]

    # if args.disease:
    #     inv = {v:k for k,v in IDX_TO_DISEASE.items()}
    #     idx = inv[args.disease]
    #     out = run_cross_validation(params, settings, args.wave, idx, pre)
    #     print(json.dumps({args.disease: out}, indent=2, ensure_ascii=False))
    # else:
    #     results={}
    #     for idx, name in IDX_TO_DISEASE.items():
    #         results[name]=run_cross_validation(params, settings, args.wave, idx, pre)
    #     print(json.dumps(results, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()













'''
    # ---- 저장 ----
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

                # ✅ 리스트 초기화 (각 질병별 AUROC, AUPRC 수집용)
                aurocs, auprcs = [], []

                for disease_name, data in disease_data.items():
                    f.write(f"--- Scenario: Fixed Disease '{disease_name.upper()}' ---\n")
                    f.write("  - Fixed Hyperparameters Used:\n")
                    for param, value in data['fixed_params'].items():
                        f.write(f"    - {param:<15}: {value}\n")
                    f.write("\n  - Final CV Test Metrics (Mean ± Std Dev):\n")
                    metrics = data['results']
                    if "status" in metrics:
                        f.write(f"    - Status: {metrics['status']}\n")
                    else:
                        mean_keys = sorted([k for k in metrics.keys() if k.endswith('_mean') and '_given_' not in k])
                        for key in mean_keys:
                            std_key = key.replace('_mean', '_std')
                            mean_val = metrics.get(key, float('nan'))
                            std_val = metrics.get(std_key, 0.0)
                            metric_name = key[:-5]
                            f.write(f"    - {metric_name:<25}: {mean_val:.4f} ± {std_val:.4f}\n")
                            # ✅ AUROC/AUPRC 값 저장
                            if metric_name.lower() == "auroc":
                                aurocs.append(mean_val)
                            elif metric_name.lower() == "auprc":
                                auprcs.append(mean_val)

                    f.write("\n" + "-"*50 + "\n\n")

                # ✅ 각 모델의 질병별 AUROC/AUPRC 요약 + 평균 추가
                if aurocs and auprcs:
                    f.write("\nSUMMARY OF AUROC & AUPRC BY DISEASE\n")
                    f.write("-"*50 + "\n")
                    f.write(f"{'Disease':<25}{'AUROC':>10}{'AUPRC':>10}\n")
                    f.write("-"*50 + "\n")
                    for i, (disease_name, data) in enumerate(disease_data.items()):
                        try:
                            auroc_val = aurocs[i]
                            auprc_val = auprcs[i]
                            f.write(f"{disease_name:<25}{auroc_val:>10.4f}{auprc_val:>10.4f}\n")
                        except IndexError:
                            continue
                    f.write("-"*50 + "\n")
                    f.write(f"{'Mean':<25}{sum(aurocs)/len(aurocs):>10.4f}{sum(auprcs)/len(auprcs):>10.4f}\n")
                    f.write("="*50 + "\n\n")

    print(f"Saved all model/disease results to: {settings.RESULTS_FILE}")


    # p=argparse.ArgumentParser()
    # p.add_argument("--wave", type=int, default=10)
    # p.add_argument("--model", choices=list(FIXED.keys()), default="gcn")
    # p.add_argument("--disease", type=str)
    # args=p.parse_args()

    # settings = Settings()
    # pre = load_data_for_wave(settings, args.wave)
    # params = FIXED[args.model]

    # if args.disease:
    #     inv = {v:k for k,v in IDX_TO_DISEASE.items()}
    #     idx = inv[args.disease]
    #     out = run_cross_validation(params, settings, args.wave, idx, pre)
    #     print(json.dumps({args.disease: out}, indent=2, ensure_ascii=False))
    # else:
    #     results={}
    #     for idx, name in IDX_TO_DISEASE.items():
    #         results[name]=run_cross_validation(params, settings, args.wave, idx, pre)
    #     print(json.dumps(results, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
'''