import pandas as pd
from collections import defaultdict
import csv

def create_adjacency_list(input_path, target_max=0.9):
    """
    질병 유사도 파일을 읽어 정규화를 수행하고,
    결과를 Adjacency List(인접 리스트) 형태로 변환합니다.

    :param input_path: 'Disease_final_similarity_score.txt' 파일 경로
    :param target_max: 정규화 시 목표로 할 최대값
    :return: Adjacency List (딕셔너리)
    """
    # 1. 데이터 불러오기
    try:
        df = pd.read_csv(input_path, sep='\t')
        print(f"'{input_path}' 파일에서 데이터를 성공적으로 불러왔습니다.")
    except FileNotFoundError:
        print(f"오류: 입력 파일 '{input_path}'을(를) 찾을 수 없습니다.")
        return None

    # 2. Min-Max Scaling으로 데이터 정규화
    print(f"유사도 점수를 0과 {target_max} 사이로 정규화합니다...")
    non_self_scores = df[df['SimilarityScore'] < 1.0]['SimilarityScore']
    
    max_score = non_self_scores.max() if not non_self_scores.empty else 0
    
    if max_score > 0:
        df['weight'] = (df['SimilarityScore'] / max_score) * target_max
    else:
        df['weight'] = df['SimilarityScore']
    
    # 3. Adjacency List 생성
    adjacency_list = defaultdict(list)
    for index, row in df.iterrows():
        disease1 = row['Disease1']
        disease2 = row['Disease2']
        weight = row['weight']

        if disease1 == disease2 or weight <= 0:
            continue
            
        # 양방향 관계를 추가합니다.
        adjacency_list[disease1].append((disease2, weight))
        adjacency_list[disease2].append((disease1, weight))
    
    print("Adjacency List 생성이 완료되었습니다.")
    return adjacency_list

def save_as_edgelist_csv(adjacency_list, output_path):
    """
    Adjacency List를 CSV 엣지 리스트 파일로 저장합니다.
    중복된 엣지(예: A-B, B-A)는 한 번만 저장합니다.

    :param adjacency_list: Adjacency List (딕셔너리)
    :param output_path: 저장할 CSV 파일 경로
    """
    processed_edges = set()
    try:
        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["source", "target", "weight"])  # CSV 헤더 작성

            for source, neighbors in adjacency_list.items():
                for target, weight in neighbors:
                    # 엣지를 정렬된 튜플로 만들어 중복 저장을 방지합니다.
                    edge = tuple(sorted((source, target)))
                    if edge not in processed_edges:
                        writer.writerow([source, target, weight])
                        processed_edges.add(edge)
        print(f"엣지 리스트가 '{output_path}' 파일로 성공적으로 저장되었습니다.")
    except Exception as e:
        print(f"CSV 파일 저장 중 오류 발생: {e}")

if __name__ == '__main__':
    # 입력 파일 경로는 사용자의 환경에 맞게 확인해주세요.
    input_file = r'C:\Users\ADS_Lab\Desktop\ADS_JHY\my_model\disease_network_ontology\disease_similarity\Disease_final_similarity_score.txt'
    # 결과를 저장할 파일 이름을 .csv로 변경합니다.
    output_file_csv = 'disease_edgelist.csv' 

    # 함수를 실행하여 Adjacency List를 생성합니다.
    adj_list = create_adjacency_list(input_file, target_max=0.9)

    # 생성된 리스트가 있다면 CSV 파일로 저장합니다.
    if adj_list:
        save_as_edgelist_csv(adj_list, output_file_csv)