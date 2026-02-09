from dataclasses import dataclass, field
from typing import List, Dict, Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path

# -----------------------------
# 설정
# -----------------------------

@dataclass
class Settings:
    # --- 파일 경로  ---
    # CSV_FILE = r"../imputed_data_with_knn_person_wave_v3.csv"
    # CSV_FILE = r"../hrs_p_select.csv"
    CSV_FILE = r"../hrs_p_select_no_leak.csv"
    # PERSON_SIM_FILE = r"../graph/similar_person_th_0.8.csv"
    # PERSON_SIM_FILE = r"../graph/similar_person_th_0.8_no_leak.csv"
    PERSON_SIM_FILE = r"../graph/similar_person_th_0.8_small.csv"
    # PERSON_SIM_FILE = r"../graph/similar_person_th_0.7_small.csv"
    DISEASE_SIM_FILE = r"../graph/disease_edgelist.csv"
    HAS_DISEASE_FILE = r"../graph/has_disease_edgelist_long_v3.csv"

    RESULTS_FILE = r"../result/[test]multimorbidity_results.txt"
    # RESULTS_FILE = r"../result/[GAT_ontology]multimorbidity_results.txt"
    # RESULTS_FILE = r"../result/[GraphFormer]_(hd32,l2,h4)_new_data_ontology_results.txt"
    # RESULTS_FILE = r"../result/[GraphFormer_MoE]_(hd32,l1,h4)_results.txt"
    # RESULTS_FILE = r"../result/[2weighted_sum]multimorbidity_results.txt"
    # RESULTS_FILE = r"../result/[GIN_predictor_moe_fusion3]multimorbidity_results.txt"
    # RESULTS_FILE = r"../result/[GIN_LLM_GPT_feature32]multimorbidity_results.txt"
    # RESULTS_FILE = r"../result/[GIN_MoE2]multimorbidity_results.txt"
    # RESULTS_FILE = r"../result/[GIN_predictor_moe_LLM_GPT_feature16_new]multimorbidity_results.txt"
    # RESULTS_FILE = r"../result/[GIN_predictor_moe_Ontology_mean_feature64_IEA_ND_new]multimorbidity_results.txt"
    # RESULTS_FILE = r"../result/[Ablation_woMoE_onehot_no_leak2]multimorbidity_results.txt"
    # RESULTS_FILE = r"../result/[MoE_usage_ratio_lambda_0]multimorbidity_results.txt"

    # RESULTS_FILE = r"../result/[Ablation_(expert1[X]_expert2[O]_expert3[O]_expert4[O])]multimorbidity_results6.txt"
    
    # RESULTS_FILE = r"../result/[GIN_predictor_moe_new_onehot_no_leak]multimorbidity_results.txt"

    # RESULTS_FILE = r"../result/[Graphormer_MoE_2]hyperparam_results.txt"
    # RESULTS_FILE = r"../result/[GAT_MoE_3loss]hyperparam_results.txt"
    # RESULTS_FILE = r"../result/[GAT_GIN_mul_MoE]hyperparam_results.txt"
    # RESULTS_FILE = r"../result/[GAT_GIN_gated_mul_MoE]hyperparam_results.txt"
    # RESULTS_FILE = r"../result/[GAT_GIN_moe_fusion]hyperparam_results.txt"
    # RESULTS_FILE = r"../result/[GIN_predictor_moe_LLM_GPT_feature16]hyperparam_results.txt"
    # RESULTS_FILE = r"../result/[GIN_predictor_moe_BioBERT_feature768]hyperparam_results.txt"
    # RESULTS_FILE = r"../result/[GIN_predictor_moe_Ontology_mean_feature64_IEA_ND]hyperparam_results.txt"
    # RESULTS_FILE = r"../result/[GIN_predictor_moe_LLM_Onto_feature80]hyperparam_results.txt"
    # RESULTS_FILE = r"../result/[GIN_predictor_moe_new_onehot_no_leak]hyperparam_results.txt"




    EDGE_DIST_LOG_FILE = r"../result/distribution/[GIN]edge_distribution_log.txt"

    # --- 실험 설정 (고정값) ---
    waves: List[int] = field(default_factory=lambda: [10])
    disease_init: Literal["onehot", "embed", "LLM", "BioBERT","Ontology", "LLM_Onto"] = "onehot"
    
    disease_emb_dim: int = 16
    # neg_pos_ratio: float = 2.0 #정답 샘플 1개당 오답을 몇 개 사용할지
    neg_pos_ratio: float = 1.0 #
    num_classes: int = 10 # 클래스 정답 개수 9개질병+질병없음
    n_folds: int = 5 # 최종 Test를 위한 Fold 수
    kfold_split_seed: int = 42 # K-Fold 분할용 고정 시드
    #n_train_folds: int = 5 # 하이퍼파라미터 튜닝(학습)을 위한 Fold 수
    in_dim_pad: Optional[int] = None # GNN 입력 차원 패딩
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # seed: int = 2025
    seed: int = 42
    
    #validation_seed: int = 42 # 왜 이렇게 하는지 이해가 안 간다..비상
    pred_threshold: float = 0.5 # 예측 임계값
    test_sampling_ratio: float = 0.2 # test로 보낼 edge 비율 조정


# --- 변수 리스트 (질병 변수 제외) ---
DISEASE_LIST = ["hibp", "diab", "cancr", "lung", "heart", "strok", "psych", "arthr", "memory"]
# DISEASE_LIST = ["strok"]
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

# NUMERIC_VARS = ["doctim", "oopmd", "cesd", "imrc", "dlrc", "ser7", "bmi", "height", "weight", "bpsys", "bpdia", "bppuls", "puff", "shltc"]
# BINARY_VARS = ["hosp", "nrshom", "doctor", "homcar", "drugs", "outpt", "dentst", "spcfac", "depres", "effort", "sleepr", "whappy", "flone", "fsad", "going", "drink", "smokev", "smoken"]
# CATEGORICAL_VARS = ["shlt", "hltc3", "slfmem", "bwc20"]


########################### new data ################################
NUMERIC_VARS = [
    "iearn", "agey_b", "agey_m", "dcbal1", "agey_e",
    "agem_b", "agem_m", "agem_e", "isret", "oopmdo",
    "liv75p", "oopmd", "dcbal2", "drinkn", "cog27",
    "issdi", "bmi", "iirawy1", "iiraw", "igxfr",
    "jyears", "jlten"
]
BINARY_VARS = [
    "work", "hosp", "walksa", "hearte2", "hearte",
    "climsa", "arthre2", "stoopa", "higov", "arthre",
    "walk1a", "pusha", "lifta", "clim1a", "unionf",
    "chaira", "hibpe2", "hibpe", "lunge2", "lunge",
    "shoph", "batha", "shopa", "mealsh", "armsa",
    "stroke2", "stroke", "mealsa", "walkra", "fsizef",
    "dressa", "bathh", "walkrh", "medsh", "dressh",
    "moneyh", "medsa", "eata", "phonea", "phoneh",
    "toilta", "moneya", "work2", "govmr", "psyche2",
    "eath", "diabe2", "sleepe", "doctor", "sleep",
    "diabe", "cancre2", "psyche", "demene", "bedh",
    "drink", "toilth", "sita", "oopmdof", "cancre",
    "beda", "hearts", "demene2", "stroks", "oopmdf",
    "demens", "lungs", "smoken", "alzhee2", "work70af",
    "alzhee", "psychs", "sleepr"
]
CATEGORICAL_VARS = [
    "conde", "lbrf", "lbrfh", "shlt", "mobila",
    "arthrq", "sayret", "walks", "lgmusa", "heartq",
    "clims", "stoop", "strokq", "grossa", "psychq",
    "walk1", "push", "hibpq", "lift", "clim1",
    "mdactx", "iadl5h", "shop", "earrt", "adl6a",
    "meals", "chair", "ltactx", "finea", "adl5a",
    "lungq", "iadl5a", "diabq", "vgactx", "adl5h",
    "demenq", "adl6h", "arms", "alzheq", "money",
    "drinkd", "meds", "phone", "mstath", "mstat",
    "imrc", "dlrc", "cancrq", "eyefr", "sleepq",
    "sit", "sleepsn", "urbrur", "eyert", "dlrcp",
    "cendiv", "imrcp", "sleepfal", "cesd", "shltc",
    "mstatf", "livbro", "ser7", "bwc20", "eyenr",
    "sleepwkn", "condef", "arthrf", "livpar", "livsis",
    "ser7p", "jltenf", "psychf", "lungf", "sleepwke",
    "alzhef", "cenreg"
]
########################### new data ################################

DISEASE_TO_IDX: Dict[str, int] = {d: i for i, d in enumerate(DISEASE_LIST)}
IDX_TO_DISEASE: Dict[int, str] = {i: d for d, i in DISEASE_TO_IDX.items()}
# IDX_TO_DISEASE: Dict[int, str] = {5:'strok'}
NONE_CLASS_IDX = 9  # 9+1에서 'none' 클래스 인덱스
