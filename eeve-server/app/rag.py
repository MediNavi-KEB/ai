import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import pandas as pd
import re
import torch
from transformers import DistilBertTokenizer, DistilBertModel
import faiss

file_path = '../../data/total_data_240723.csv'
df = pd.read_csv(file_path)

# 증상 문장 전처리 함수 (특수 문자 제거만 수행): , .
def preprocess_text(texts):
    processed_texts = []
    for text in texts:
        # 특수 문자 제거
        text = re.sub(r"[^ㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z0-9\s]", "", text)
        processed_texts.append(text)
    return processed_texts

# DistilBERT 모델 및 토크나이저 초기화
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-multilingual-cased")
model = DistilBertModel.from_pretrained("distilbert-base-multilingual-cased")

# 배치 크기 설정
batch_size = 16

# 배치 처리 함수
def embed_passages_in_batches(passages, batch_size):
    all_embeddings = []
    for i in range(0, len(passages), batch_size):
        batch = passages[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state.mean(dim=1)  # CLS 토큰 대신 평균 풀링 사용
        all_embeddings.append(batch_embeddings)
    return torch.cat(all_embeddings, dim=0)

faiss_index_path = '../../data/faiss_index1.bin'
index = faiss.read_index(faiss_index_path)

# 유사 증상 검색 함수
def search(query, top_k=5):
    preprocessed_query = preprocess_text([query])[0]
    query_embedding = embed_passages_in_batches([preprocessed_query], batch_size).numpy()

    # FAISS를 사용하여 유사한 벡터 검색
    similarities, indices = index.search(query_embedding, top_k)
    indices = indices[0]
    similarities = similarities[0]

    # 검색된 결과를 데이터프레임으로 정리
    results = df.iloc[indices][['병명']]
    results['거리'] = similarities

    return results.reset_index(drop=True)

def calculate_weighted_scores(data):
    """
    주어진 데이터에 대해 순위와 가중치를 계산하고 각 질병에 대해 가중치를 고려한 합을 반환하는 함수.

    Parameters:
    data (dict): 병명, 증상, 거리 데이터를 포함하는 딕셔너리.

    Returns:
    pd.DataFrame: 병명별 가중 점수의 합계가 포함된 데이터프레임.
    """
    # 데이터프레임 생성
    df = data.copy()

    # 거리 기준으로 순위 계산 (거리가 낮을수록 높은 순위)
    df['순위'] = df.index + 1

    # 순위에 따라 가중치 계산 (순위가 낮을수록 높은 가중치)
    df['가중치'] = 1 / df['순위']

    # 가중 점수 계산 (거리 * 가중치)
    df['가중 점수'] = df['거리'] * df['가중치']

    # 각 병명별 가중 점수 합계 계산
    disease_weighted_sum = df.groupby('병명')['가중 점수'].sum().reset_index()

    # 가중 점수 합계 기준으로 정렬
    disease_weighted_sum = disease_weighted_sum.sort_values(by='가중 점수', ascending=False).reset_index(drop=True)

    return disease_weighted_sum


