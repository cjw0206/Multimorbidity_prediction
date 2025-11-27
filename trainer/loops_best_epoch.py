import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from dgl import DGLGraph  # ← 타입힌트 쓰면 꼭 필요
from core.config import Settings, IDX_TO_DISEASE  # ← 타입/상수 둘 다 사용

from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score, accuracy_score
)
from tqdm import tqdm

from .builder import create_model_and_predictor

# -----------------------------
# 학습/평가 루프
# -----------------------------
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

    # 🔹 AUROC 기준 best epoch 추적용 변수
    best_test_auroc = -float("inf")
    best_test_epoch = -1          # 나중에 1-based로 저장
    best_test_scores = None       # best epoch에서의 test logits

    epoch_iter = tqdm(range(params['max_epochs']), desc=f"Epoch | {progress_desc}", ncols=80, leave=False) if progress_desc else range(params['max_epochs'])

    for epoch in epoch_iter:
        # --- 1. 학습 ---
        model.train(); predictor.train()

        if not params["using_moe"]:
            logits = model.forward_edges(g, feats, predictor, uv=(train_u, train_v)).squeeze(-1)
            loss = loss_fn(logits, train_y.float())
        else:
            logits, moe_loss = model.forward_edges(g, feats, predictor, uv=(train_u, train_v))
            logits = logits.squeeze(-1)
            loss = loss_fn(logits, train_y.float())
            loss += moe_loss * 0.01               # 3 losses 사용 시 이 부분 주석해야 함.

        opt.zero_grad(); loss.backward(); opt.step()
        epoch_losses.append(loss.item())
        
        # --- 2. 매 에폭마다 테스트 성능 측정 (모니터링용) ---
        model.eval(); predictor.eval()
        with torch.no_grad():
            if not params["using_moe"]:
                test_logits = model.forward_edges(g, feats, predictor, uv=(test_u, test_v)).squeeze(-1)
            #test_scores = torch.sigmoid(test_logits).cpu().numpy()
            else:
                test_logits, moe_loss = model.forward_edges(g, feats, predictor, uv=(test_u, test_v))
                test_logits = test_logits.squeeze(-1)

            test_scores_epoch = test_logits.cpu().numpy()
            
        try:
            test_auroc = roc_auc_score(test_y_cpu, test_scores_epoch)
        except ValueError:
            test_auroc = 0.0

        # 베스트 auroc를 만날 경우 갱신
        if test_auroc >= best_test_auroc:   # 동일하면 나중 epoch을 best로
            best_test_auroc = test_auroc
            best_test_epoch = epoch + 1     # 사람이 보기 좋게 1-based
            best_test_scores = test_scores_epoch

        #     torch.save(
        #     {
        #         "model": model.state_dict(),
        #         "predictor": predictor.state_dict()
        #     },
        #     f"../saved_models/best_model{settings.seed}.pth"
        # )
        
        # --- 3. 진행률 표시줄 업데이트 ---
        if hasattr(epoch_iter, "set_postfix"):
            epoch_iter.set_postfix(loss=loss.item(), 
                                   test_auroc=test_auroc,
                                   best_auroc=best_test_auroc,
                                   best_epoch=best_test_epoch
                                   )

    if hasattr(epoch_iter, "close"): epoch_iter.close()

    assert best_test_scores is not None, "best_test_scores가 설정되지 않았습니다. train loop를 확인하세요."

    test_scores_for_eval = best_test_scores

    # --- 최종 평가는 최고 auc 성능 에폭 모델로 한 번만 수행 ---
    model.eval(); predictor.eval()
    with torch.no_grad():
        # test_logits = model.forward_edges(g, feats, predictor, uv=(test_u, test_v)).squeeze(-1)
        # test_scores = test_logits.cpu().numpy()
        
        p = settings.pred_threshold
        eps = 1e-12
        p = min(max(p, eps), 1 - eps)
        logit_thr = math.log(p / (1 - p))
        
        if eval_info:
            final_test_metrics = evaluate_multimorbidity_metrics(
                test_y_cpu, 
                #test_scores,
                test_scores_for_eval,
                threshold=logit_thr,
                eval_info=eval_info,
                idx_to_disease_map=IDX_TO_DISEASE
            )
        else:
            final_test_metrics = evaluate_metrics_binary(
                test_y_cpu, 
                #test_scores,
                test_scores_for_eval,
                threshold=logit_thr,
                disease_ids=None
            )

        if not params["using_moe"]:
            train_logits = model.forward_edges(g, feats, predictor, uv=(train_u, train_v)).squeeze(-1)
        else:
            logits, moe_loss = model.forward_edges(g, feats, predictor, uv=(train_u, train_v))
            train_logits = logits.squeeze(-1)

        train_scores = train_logits.cpu().numpy()
        
        final_train_metrics = evaluate_metrics_binary(
            train_y.cpu().numpy(), train_scores,
            threshold=logit_thr,
            disease_ids=None
        )

    # 🔹 best epoch / best auroc 정보도 같이 넣기
    #   -> run_cross_validation에서 fold별로 평균/표준편차를 낼 수 있음
    final_test_metrics["best_epoch_by_auroc"] = float(best_test_epoch)
    final_test_metrics["best_auroc_during_train"] = float(best_test_auroc)   

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