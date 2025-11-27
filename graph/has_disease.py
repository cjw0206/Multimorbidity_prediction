# # # modified_has_disease.py

# # import pandas as pd
# # import re
# # from collections import defaultdict
# # import json

# # # 1. CSV 파일 불러오기
# # file_path = r"C:\Users\ADS_Lab\Desktop\ADS_JHY\my_model\comparison_model\imputed_data_with_knn.csv"
# # try:
# #     df = pd.read_csv(file_path, dtype={'hhidpn': str})
# #     print(f"'{file_path}' 파일에서 데이터를 성공적으로 불러왔습니다.")
# # except FileNotFoundError:
# #     print(f"오류: 입력 파일 '{file_path}'을(를) 찾을 수 없습니다.")
# #     exit()

# # # CSV 파일에 hhidpn 컬럼이 없다면 인덱스를 식별자로 사용
# # if "hhidpn" not in df.columns:
# #     df.reset_index(inplace=True)
# #     df.rename(columns={"index": "hhidpn"}, inplace=True)

# # # 2. 질병 변수 리스트
# # disease_list = ["hibp", "diab", "cancr", "lung",
# #                 "heart", "strok", "psych", "arthr", "memory"]

# # # 3. Adjacency List로 저장할 딕셔너리 초기화
# # # defaultdict를 사용하면 키가 없을 때 자동으로 빈 리스트를 생성해줍니다.
# # adjacency_list = defaultdict(list)

# # # 4. CSV 순회: 각 행(환자)에 대해 보유한 질병을 Adjacency List에 추가
# # print("데이터를 순회하며 Adjacency List를 생성합니다...")
# # for _, row in df.iterrows():
# #     patient_id = row["hhidpn"]
    
# #     for col in df.columns:
# #         m = re.match(r"r(\d+)(\w+)", col)
# #         if m:
# #             wave_num = m.group(1)
# #             var_name = m.group(2)
            
# #             if var_name in disease_list:
# #                 try:
# #                     value = float(row[col])
# #                 except (ValueError, TypeError):
# #                     value = None
                
# #                 # 값이 1이면 해당 질병을 보유한 것으로 간주
# #                 if value == 1:
# #                     # Key: "개인ID_wave", Value: 질병 리스트
# #                     node_key = f"{patient_id}_{wave_num}"
# #                     if var_name not in adjacency_list[node_key]:
# #                         adjacency_list[node_key].append(var_name)

# # # 5. Adjacency List를 JSON 파일로 저장
# # output_file = "has_disease_adjacency_list.json"
# # #print(f"Adjacency List를 '{output_file}' 파일로 저장합니다...")
# # try:
# #     with open(output_file, 'w', encoding='utf-8') as f:
# #         # 생성된 딕셔너리를 가독성 좋게 JSON 파일로 저장
# #         json.dump(adjacency_list, f, indent=4, ensure_ascii=False)
# #     print(f"Bipartite graph Adjacency List가 '{output_file}'로 성공적으로 저장되었습니다.")
# # except Exception as e:
# #     print(f"JSON 파일 저장 중 오류 발생: {e}")

# import pandas as pd
# import re
# from collections import defaultdict
# import csv # csv 모듈을 임포트합니다.

# # 1. CSV 파일 불러오기
# file_path = r"C:\Users\ADS_Lab\Desktop\ADS_JHY\my_model\comparison_model\imputed_data_with_knn_v2.csv"
# try:
#     df = pd.read_csv(file_path, dtype={'hhidpn': str})
#     print(f"'{file_path}' 파일에서 데이터를 성공적으로 불러왔습니다.")
# except FileNotFoundError:
#     print(f"오류: 입력 파일 '{file_path}'을(를) 찾을 수 없습니다.")
#     exit()

# if "hhidpn" not in df.columns:
#     df.reset_index(inplace=True)
#     df.rename(columns={"index": "hhidpn"}, inplace=True)

# # 2. 질병 변수 리스트
# disease_list = ["hibp", "diab", "cancr", "lung",
#                 "heart", "strok", "psych", "arthr", "memory"]

# # 3. Adjacency List로 사용될 딕셔너리 초기화
# adjacency_list = defaultdict(list)

# # 4. CSV 순회: 각 행(환자)에 대해 보유한 질병을 리스트에 추가
# print("데이터를 순회하며 환자-질병 관계를 생성합니다...")
# for _, row in df.iterrows():
#     patient_id = row["hhidpn"]
    
#     for col in df.columns:
#         m = re.match(r"r(\d+)(\w+)", col)
#         if m:
#             wave_num = m.group(1)
#             var_name = m.group(2)
            
#             if var_name in disease_list:
#                 try:
#                     value = float(row[col])
#                 except (ValueError, TypeError):
#                     value = None
                
#                 # 값이 1이면 해당 질병을 보유한 것으로 간주
#                 if value == 1:
#                     node_key = f"{patient_id}_{wave_num}"
#                     if var_name not in adjacency_list[node_key]:
#                         adjacency_list[node_key].append(var_name)

# # 5. Adjacency List를 CSV 엣지 리스트 파일로 저장 (수정된 부분)
# output_file_csv = "has_disease_edgelist_v2.csv"
# print(f"Bipartite graph 엣지 리스트를 '{output_file_csv}' 파일로 저장합니다...")
# try:
#     with open(output_file_csv, 'w', encoding='utf-8', newline='') as f:
#         writer = csv.writer(f)
#         # CSV 헤더 작성
#         writer.writerow(["person_wave_id", "disease"])
        
#         # 딕셔너리를 순회하며 데이터 작성
#         for person_wave_id, diseases in adjacency_list.items():
#             for disease in diseases:
#                 writer.writerow([person_wave_id, disease])
                
#     print(f"Bipartite graph 엣지 리스트가 '{output_file_csv}'로 성공적으로 저장되었습니다.")
# except Exception as e:
#     print(f"CSV 파일 저장 중 오류 발생: {e}")

import pandas as pd
import csv

# 1. CSV 파일 불러오기
# 긴 포맷(long format) 데이터 파일을 지정합니다.
file_path = r"C:\Users\ADS_Lab\Desktop\ADS_JHY\HRS_datas\preprocessed_data\imputed_data_with_knn_person_wave_v3.csv"
try:
    # person_wave_id를 문자열 타입으로 읽어옵니다.
    df = pd.read_csv(file_path, dtype={'person_wave_id': str})
    print(f"'{file_path}' 파일에서 데이터를 성공적으로 불러왔습니다.")
except FileNotFoundError:
    print(f"오류: 입력 파일 '{file_path}'을(를) 찾을 수 없습니다.")
    exit()

# 2. 질병 변수 리스트
# 데이터프레임의 컬럼 이름과 일치하는 질병 목록입니다.
disease_list = ["hibp", "diab", "cancr", "lung",
                "heart", "strok", "psych", "arthr", "memory"]

# 3. 엣지 리스트로 사용할 리스트 초기화
edge_list = []

# 4. 데이터프레임 순회
# 이전 코드와 달리, 각 행을 순회하며 질병 컬럼의 값을 바로 확인합니다.
print("데이터를 순회하며 환자-질병 관계를 생성합니다...")
for _, row in df.iterrows():
    person_wave_id = row["person_wave_id"]
    
    # 각 질병에 대해 컬럼 값을 확인
    for disease in disease_list:
        # 데이터프레임에 해당 질병 컬럼이 있고, 그 값이 1인지 확인합니다.
        # 값이 NaN일 수 있으므로 pd.notna()로 먼저 확인하는 것이 안전합니다.
        if disease in row and pd.notna(row[disease]) and row[disease] == 1:
            # 엣지 리스트에 [소스 노드, 타겟 노드] 형태로 추가
            edge_list.append([person_wave_id, disease])

# 5. 엣지 리스트를 CSV 파일로 저장
# 입력 파일에 맞춰 출력 파일 이름을 지정합니다.
output_file_csv = "has_disease_edgelist_long_v3.csv"
print(f"Bipartite graph 엣지 리스트를 '{output_file_csv}' 파일로 저장합니다...")
try:
    with open(output_file_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        # CSV 헤더 작성
        writer.writerow(["person_wave_id", "disease"])
        # 리스트에 저장된 모든 엣지를 한 번에 파일에 씁니다.
        writer.writerows(edge_list)
                
    print(f"✅ Bipartite graph 엣지 리스트가 '{output_file_csv}'로 성공적으로 저장되었습니다.")
except Exception as e:
    print(f"❌ CSV 파일 저장 중 오류 발생: {e}")