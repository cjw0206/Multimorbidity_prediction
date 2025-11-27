# import pandas as pd
# import numpy as np
# import networkx as nx
# import re
# import gc
# from tqdm import tqdm
# from collections import defaultdict
# import json
# import matplotlib.pyplot as plt
# import seaborn as sns

# # ==============================================================================
# # 1. Wide → Long 변환 함수
# # ==============================================================================
# def reshape_in_memory(df_raw):
#     id_col = 'hhidpn'
#     all_cols = list(df_raw.columns)
#     pattern = re.compile(r'^r(\d+)([a-zA-Z0-9_]+)$')

#     wave_columns = {}
#     for col in all_cols:
#         if col == id_col:
#             continue
#         m = pattern.match(col)
#         if m:
#             wave_str = m.group(1)
#             var_str  = m.group(2)
#             wave_int = int(wave_str)
#             if wave_int not in wave_columns:
#                 wave_columns[wave_int] = {}
#             wave_columns[wave_int][var_str] = col

#     df_list = []
#     waves_sorted = sorted(wave_columns.keys())

#     for w in tqdm(waves_sorted, desc="[reshape_in_memory] Reshaping data"):
#         var_map = wave_columns[w]
#         use_cols = [id_col] + list(var_map.values())
#         sub = df_raw[use_cols].copy()
#         rename_dict = { widecol: varname for varname, widecol in var_map.items() }
#         sub.rename(columns=rename_dict, inplace=True)
#         sub['wave'] = w
#         df_list.append(sub)
#     df_reshaped = pd.concat(df_list, ignore_index=True)
#     return df_reshaped

# # ==============================================================================
# # 2. Vectorized Gower 유사도 계산 함수
# # ==============================================================================
# def compute_gower_similarity_matrix_vectorized(df_wave, numeric_vars, binary_vars, categorical_vars):
#     n = df_wave.shape[0]
#     total_diff = np.zeros((n, n))
#     var_count = len(numeric_vars) + len(binary_vars) + len(categorical_vars)

#     if var_count == 0:
#         return np.zeros((n, n))

#     # tqdm을 제거하여 루프 내 출력 간소화
#     for col in numeric_vars:
#         arr = df_wave[col].to_numpy(dtype=float)
#         rng = np.nanmax(arr) - np.nanmin(arr)
#         if rng == 0: rng = 1.0
#         diff = np.abs(arr[:, None] - arr[None, :]) / rng
#         total_diff += np.nan_to_num(diff)

#     for col in binary_vars:
#         arr = df_wave[col].to_numpy(dtype=float)
#         diff = (arr[:, None] != arr[None, :]).astype(float)
#         total_diff += np.nan_to_num(diff)

#     for col in categorical_vars:
#         arr = df_wave[col].to_numpy()
#         diff = (arr[:, None] != arr[None, :]).astype(float)
#         total_diff += np.nan_to_num(diff)

#     avg_diff = total_diff / var_count
#     sim_matrix = 1 - avg_diff
#     return sim_matrix

# # ==============================================================================
# # 3. 그래프 생성 함수
# # ==============================================================================
# def build_intra_layer_graph_gower_no_topk(df, numeric_vars, binary_vars, categorical_vars, threshold=0.8):
#     G_intra = nx.Graph()
#     waves = sorted(df['wave'].unique())

#     # 바깥쪽 tqdm: Wave별 진행률
#     for w in tqdm(waves, desc="[build_graph] Processing Waves", leave=False):
#         sub = df[df['wave'] == w].copy().reset_index(drop=True)
#         person_ids = sub['hhidpn'].tolist()

#         for pid in person_ids:
#             node = ("person", pid, w)
#             G_intra.add_node(node, node_type="person", wave=w)

#         if len(sub) < 2:
#             continue

#         filtered_numeric_vars = [var for var in numeric_vars if var in sub.columns and sub[var].notna().any()]
#         filtered_binary_vars = [var for var in binary_vars if var in sub.columns and sub[var].notna().any()]
#         filtered_categorical_vars = [var for var in categorical_vars if var in sub.columns and sub[var].notna().any()]

#         if not (filtered_numeric_vars or filtered_binary_vars or filtered_categorical_vars):
#             continue

#         sim_matrix = compute_gower_similarity_matrix_vectorized(
#             sub, filtered_numeric_vars, filtered_binary_vars, filtered_categorical_vars
#         )
#         n = len(sub)

#         for i in tqdm(range(n), desc=f"  - Comparing Pairs (Wave {w})", leave=False):
#             for j in range(i + 1, n):
#                 sim = sim_matrix[i, j]
#                 if sim >= threshold:
#                     node_i = ("person", person_ids[i], w)
#                     node_j = ("person", person_ids[j], w)
#                     G_intra.add_edge(node_i, node_j, edge_type="intra-layer", weight=sim)

#         del sim_matrix
#         gc.collect()

#     return G_intra

# # ==============================================================================
# # 4. 그래프를 저장하는 함수
# # ==============================================================================
# # def save_graph_as_adjacency_list_json(G, output_file):
# #     adjacency_list = defaultdict(list)
# #     for u, v, data in G.edges(data=True):
# #         u_key = f"{u[1]}_{u[2]}"
# #         v_key = f"{v[1]}_{v[2]}"
# #         weight = data.get("weight", 0.0)
# #         adjacency_list[u_key].append([v_key, weight])
# #         adjacency_list[v_key].append([u_key, weight])

# #     for key in adjacency_list:
# #         adjacency_list[key] = sorted(adjacency_list[key], key=lambda x: x[1], reverse=True)

# #     try:
# #         with open(output_file, 'w', encoding='utf-8') as f:
# #             json.dump(adjacency_list, f, indent=4, ensure_ascii=False)
# #         print(f"Graph saved to '{output_file}'")
# #     except Exception as e:
# #         print(f"Error saving JSON file: {e}")

# def save_graph_as_edgelist_csv(G, output_file):
#     try:
#         with open(output_file, 'w', encoding='utf-8') as f:
#             # CSV 헤더(첫 줄) 작성
#             f.write("source,target,weight\n")
#             # 모든 엣지를 순회하며 한 줄씩 파일에 쓰기
#             for u, v, data in G.edges(data=True):
#                 u_key = f"{u[1]}_{u[2]}"
#                 v_key = f"{v[1]}_{v[2]}"
#                 weight = data.get("weight", 0.0)
#                 f.write(f"{u_key},{v_key},{weight}\n")
#         print(f"Graph saved to '{output_file}' (CSV format)")
#     except Exception as e:
#         print(f"Error saving CSV file: {e}")

# # ==============================================================================
# # 5. 변수 리스트 설정 
# # ==============================================================================
# numeric_vars_example = [
#     "doctim", "oopmd", "cesd", "imrc", "dlrc", "ser7",
#     "bmi", "height", "weight", "bpsys", "bpdia", "bppuls", "puff", "shltc"
# ]
# original_binary_vars = [
#     "hosp", "nrshom", "doctor", "homcar", "drugs", "outpt", "dentst",
#     "spcfac", "depres", "effort", "sleepr", "whappy", "flone", "fsad",
#     "going", "hibp", "diab", "cancr", "lung", "heart", "strok",
#     "psych", "arthr", "drink", "smokev", "smoken", "memory"
# ]
# disease_vars = ["hibp", "diab", "cancr", "lung", "heart", "strok", "psych", "arthr", "memory"]
# binary_vars_example = [var for var in original_binary_vars if var not in disease_vars]
# categorical_vars_example = ["shlt", "hltc3", "slfmem", "bwc20"]


# # ==============================================================================
# # 6. 메인 실행 블록 (CSV 저장으로 수정됨)
# # ==============================================================================
# if __name__ == "__main__":
#     # matplotlib 백엔드 설정 (서버 환경용)
#     import matplotlib
#     matplotlib.use('Agg')
#     import matplotlib.pyplot as plt

#     # 데이터 로드 
#     input_file_path = r"/home/jhpark/HY/data/imputed_data_with_knn.csv"
#     try:
#         df_raw = pd.read_csv(input_file_path, dtype={'hhidpn': str})
#     except FileNotFoundError:
#         print(f"Error: Input file not found at '{input_file_path}'")
#         exit()

#     print("Step 1: Reshaping data from wide to long format...")
#     df_long = reshape_in_memory(df_raw)

#     # 분석할 Threshold 범위 설정
#     thresholds_to_test = np.arange(0.5, 1.01, 0.1)
#     density_results = []

#     print("\nStep 2: Starting analysis for each threshold...")
#     # 메인 루프에 tqdm 적용하여 전체 진행률 표시
#     for th in tqdm(thresholds_to_test, desc="Overall Progress"):
#         current_threshold = round(th, 1)
#         print(f"\nProcessing for threshold = {current_threshold}...")

#         # 그래프 생성
#         graph = build_intra_layer_graph_gower_no_topk(
#             df_long,
#             numeric_vars_example,
#             binary_vars_example,
#             categorical_vars_example,
#             threshold=current_threshold
#         )

#         print(f" Graph threshold {current_threshold}, {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
#         print(f" 그래프 저장 시작")

#         # CSV 엣지 리스트 파일로 저장 (훨씬 빠름)
#         output_filename_csv = f"similar_person_th_{current_threshold}.csv"
#         save_graph_as_edgelist_csv(graph, output_filename_csv)

#         print(f" 그래프 파일 saved.")

#         # 밀도 계산 및 결과 저장
#         density = nx.density(graph)
#         density_results.append({
#             'threshold': current_threshold,
#             'nodes': graph.number_of_nodes(),
#             'edges': graph.number_of_edges(),
#             'density': density
#         })
        
#         # 메모리 관리를 위해 그래프 객체 삭제
#         del graph
#         gc.collect()

#     # --- 3. 최종 결과 요약 및 시각화 ---
#     results_df = pd.DataFrame(density_results)
#     print("\n\n[Final Analysis Summary]")
#     print(results_df.to_string(index=False))

#     # 결과 시각화
#     plt.style.use('seaborn-v0_8-whitegrid')
#     fig, ax = plt.subplots(figsize=(12, 7))
#     sns.lineplot(data=results_df, x='threshold', y='density', marker='o', ax=ax, markersize=8)
#     ax.set_title('Graph Density vs. Similarity Threshold (Top-K Removed)', fontsize=16, pad=20)
#     ax.set_xlabel('Similarity Threshold', fontsize=12)
#     ax.set_ylabel('Graph Density', fontsize=12)
#     ax.grid(True, which='both', linestyle='--', linewidth=0.5)

#     # 각 포인트에 밀도 값 텍스트 추가
#     for i, row in results_df.iterrows():
#         ax.text(row['threshold'], row['density'], f" {row['density']:.4f}",
#                 ha='left', va='center', fontsize=9, color='darkblue', weight='bold')

#     plt.tight_layout()
#     plot_filename = 'density_vs_threshold_no_topk.png'
#     plt.savefig(plot_filename)

#     print(f"\nDensity analysis plot saved to '{plot_filename}'")
#     print("\nAll tasks completed successfully!")

import pandas as pd
import numpy as np
import networkx as nx
import re
import gc
from tqdm import tqdm
from collections import defaultdict
import json
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# 1. Wide → Long 변환 함수 (수정된 버전)
# ==============================================================================
def reshape_in_memory(df_raw):
    """
    새로운 데이터는 이미 Long 형식이므로 복잡한 변환이 필요 없습니다.
    입력된 데이터프레임을 그대로 반환합니다.
    추후 필요한 전처리가 있다면 이 함수에 추가할 수 있습니다.
    """
    print("데이터가 이미 Long 형식이므로 변환 과정은 생략합니다.")
    
    # 원본 코드와의 일관성을 위해 hhidpn 타입을 문자열로 유지합니다.
    if 'hhidpn' in df_raw.columns:
        df_raw['hhidpn'] = df_raw['hhidpn'].astype(str)
        
    return df_raw

# ==============================================================================
# 2. Vectorized Gower 유사도 계산 함수
# ==============================================================================
def compute_gower_similarity_matrix_vectorized(df_wave, numeric_vars, binary_vars, categorical_vars):
    n = df_wave.shape[0]
    total_diff = np.zeros((n, n))
    var_count = len(numeric_vars) + len(binary_vars) + len(categorical_vars)

    if var_count == 0:
        return np.zeros((n, n))

    # tqdm을 제거하여 루프 내 출력 간소화
    for col in numeric_vars:
        arr = df_wave[col].to_numpy(dtype=float)
        rng = np.nanmax(arr) - np.nanmin(arr)
        if rng == 0: rng = 1.0
        diff = np.abs(arr[:, None] - arr[None, :]) / rng
        total_diff += np.nan_to_num(diff)

    for col in binary_vars:
        arr = df_wave[col].to_numpy(dtype=float)
        diff = (arr[:, None] != arr[None, :]).astype(float)
        total_diff += np.nan_to_num(diff)

    for col in categorical_vars:
        arr = df_wave[col].to_numpy()
        diff = (arr[:, None] != arr[None, :]).astype(float)
        total_diff += np.nan_to_num(diff)

    avg_diff = total_diff / var_count
    sim_matrix = 1 - avg_diff
    return sim_matrix

# ==============================================================================
# 3. 그래프 생성 함수
# ==============================================================================
def build_intra_layer_graph_gower_no_topk(df, numeric_vars, binary_vars, categorical_vars, threshold=0.8):
    G_intra = nx.Graph()
    waves = sorted(df['wave'].unique())

    # 바깥쪽 tqdm: Wave별 진행률
    for w in tqdm(waves, desc="[build_graph] Processing Waves", leave=False):
        sub = df[df['wave'] == w].copy().reset_index(drop=True)
        person_ids = sub['hhidpn'].tolist()

        for pid in person_ids:
            node = ("person", pid, w)
            G_intra.add_node(node, node_type="person", wave=w)

        if len(sub) < 2:
            continue

        filtered_numeric_vars = [var for var in numeric_vars if var in sub.columns and sub[var].notna().any()]
        filtered_binary_vars = [var for var in binary_vars if var in sub.columns and sub[var].notna().any()]
        filtered_categorical_vars = [var for var in categorical_vars if var in sub.columns and sub[var].notna().any()]

        if not (filtered_numeric_vars or filtered_binary_vars or filtered_categorical_vars):
            continue

        sim_matrix = compute_gower_similarity_matrix_vectorized(
            sub, filtered_numeric_vars, filtered_binary_vars, filtered_categorical_vars
        )
        n = len(sub)

        for i in tqdm(range(n), desc=f"  - Comparing Pairs (Wave {w})", leave=False):
            for j in range(i + 1, n):
                sim = sim_matrix[i, j]
                if sim >= threshold:
                    node_i = ("person", person_ids[i], w)
                    node_j = ("person", person_ids[j], w)
                    G_intra.add_edge(node_i, node_j, edge_type="intra-layer", weight=sim)

        del sim_matrix
        gc.collect()

    return G_intra

# ==============================================================================
# 4. 그래프를 저장하는 함수
# ==============================================================================
# def save_graph_as_adjacency_list_json(G, output_file):
#     adjacency_list = defaultdict(list)
#     for u, v, data in G.edges(data=True):
#         u_key = f"{u[1]}_{u[2]}"
#         v_key = f"{v[1]}_{v[2]}"
#         weight = data.get("weight", 0.0)
#         adjacency_list[u_key].append([v_key, weight])
#         adjacency_list[v_key].append([u_key, weight])

#     for key in adjacency_list:
#         adjacency_list[key] = sorted(adjacency_list[key], key=lambda x: x[1], reverse=True)

#     try:
#         with open(output_file, 'w', encoding='utf-8') as f:
#             json.dump(adjacency_list, f, indent=4, ensure_ascii=False)
#         print(f"Graph saved to '{output_file}'")
#     except Exception as e:
#         print(f"Error saving JSON file: {e}")

def save_graph_as_edgelist_csv(G, output_file):
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            # CSV 헤더(첫 줄) 작성
            f.write("source,target,weight\n")
            # 모든 엣지를 순회하며 한 줄씩 파일에 쓰기
            for u, v, data in G.edges(data=True):
                u_key = f"{u[1]}_{u[2]}"
                v_key = f"{v[1]}_{v[2]}"
                weight = data.get("weight", 0.0)
                f.write(f"{u_key},{v_key},{weight}\n")
        print(f"Graph saved to '{output_file}' (CSV format)")
    except Exception as e:
        print(f"Error saving CSV file: {e}")

# ==============================================================================
# 5. 변수 리스트 설정 
# ==============================================================================
numeric_vars_example = [
    "doctim", "oopmd", "cesd", "imrc", "dlrc", "ser7",
    "bmi", "height", "weight", "bpsys", "bpdia", "bppuls", "puff", "shltc"
]
original_binary_vars = [
    "hosp", "nrshom", "doctor", "homcar", "drugs", "outpt", "dentst",
    "spcfac", "depres", "effort", "sleepr", "whappy", "flone", "fsad",
    "going", "hibp", "diab", "cancr", "lung", "heart", "strok",
    "psych", "arthr", "drink", "smokev", "smoken", "memory"
]
disease_vars = ["hibp", "diab", "cancr", "lung", "heart", "strok", "psych", "arthr", "memory"]
binary_vars_example = [var for var in original_binary_vars if var not in disease_vars]
categorical_vars_example = ["shlt", "hltc3", "slfmem", "bwc20"]


# ==============================================================================
# 6. 메인 실행 블록 (CSV 저장으로 수정됨)
# ==============================================================================
if __name__ == "__main__":
    # matplotlib 백엔드 설정 (서버 환경용)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # 1.파일 경로를 지정
    input_file_path = r"./imputed_data_with_knn_person_wave_v3.csv"
    try:
        # person_wave_id는 ID와 wave의 조합이므로 특별히 타입을 지정할 필요는 없습니다.
        # hhidpn은 원본 코드와 같이 문자열로 읽어옵니다.
        df_raw = pd.read_csv(input_file_path, dtype={'hhidpn': str})
    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_file_path}'")
        exit()

    print("Step 1: Reshaping data from wide to long format...")
    # 2. 수정된 함수를 호출합니다. (실제 변환은 일어나지 않음)
    df_long = reshape_in_memory(df_raw)

    # 분석할 Threshold 범위 설정
    thresholds_to_test = np.arange(0.5, 0.6, 0.1)
    density_results = []

    print("\nStep 2: Starting analysis for each threshold...")
    # 메인 루프에 tqdm 적용하여 전체 진행률 표시
    for th in tqdm(thresholds_to_test, desc="Overall Progress"):
        current_threshold = round(th, 1)
        print(f"\nProcessing for threshold = {current_threshold}...")

        # 그래프 생성
        graph = build_intra_layer_graph_gower_no_topk(
            df_long,
            numeric_vars_example,
            binary_vars_example,
            categorical_vars_example,
            threshold=current_threshold
        )

        print(f" Graph threshold {current_threshold}, {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
        print(f" 그래프 저장 시작")

        # CSV 엣지 리스트 파일로 저장 (훨씬 빠름)
        output_filename_csv = f"similar_person_th_{current_threshold}.csv"
        save_graph_as_edgelist_csv(graph, output_filename_csv)

        print(f" 그래프 파일 saved.")

        # 밀도 계산 및 결과 저장
        density = nx.density(graph)
        density_results.append({
            'threshold': current_threshold,
            'nodes': graph.number_of_nodes(),
            'edges': graph.number_of_edges(),
            'density': density
        })
        
        # 메모리 관리를 위해 그래프 객체 삭제
        del graph
        gc.collect()

    # --- 3. 최종 결과 요약 및 시각화 ---
    results_df = pd.DataFrame(density_results)
    print("\n\n[Final Analysis Summary]")
    print(results_df.to_string(index=False))

    # 결과 시각화
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.lineplot(data=results_df, x='threshold', y='density', marker='o', ax=ax, markersize=8)
    ax.set_title('Graph Density vs. Similarity Threshold (Top-K Removed)', fontsize=16, pad=20)
    ax.set_xlabel('Similarity Threshold', fontsize=12)
    ax.set_ylabel('Graph Density', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # 각 포인트에 밀도 값 텍스트 추가
    for i, row in results_df.iterrows():
        ax.text(row['threshold'], row['density'], f" {row['density']:.4f}",
                ha='left', va='center', fontsize=9, color='darkblue', weight='bold')

    plt.tight_layout()
    plot_filename = 'density_vs_threshold_no_topk.png'
    plt.savefig(plot_filename)

    print(f"\nDensity analysis plot saved to '{plot_filename}'")
    print("\nAll tasks completed successfully!")