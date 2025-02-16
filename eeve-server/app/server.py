from fastapi import APIRouter, HTTPException, Depends
from translate import chain
from message_dto import ChatUserInput
from rag import search, calculate_weighted_scores
from fastapi.responses import StreamingResponse
import asyncio
import logging
import re

ai_router = APIRouter(tags=["AI"])


@ai_router.post("/disease_recommendation")
async def disease_recommendation(request: ChatUserInput):
    try:
        # 디버깅을 위해 입력 내용 로그에 남김
        input_text = request.input
        print(f"Input to model: {input_text}")

        # faiss 검색
        search_results = search(input_text, top_k=100)
        # 거리가 20 이내인 결과 필터링
        filtered_results = search_results[search_results['거리'] < 20]
        print(filtered_results)

        # 필터링된 결과가 50개 이상인지 확인
        if len(filtered_results) >= 50:
            # 가중치 계산
            weighted_scores = calculate_weighted_scores(filtered_results)
            print("Weighted Scores:", weighted_scores)

            # 상위 3개 항목 추출
            top_diseases = weighted_scores.head(3)
            disease_names = top_diseases['질병'].tolist()

            # 모델의 응답 로그에 남김
            print(f"disease_names: {disease_names}")

            output = "아래와 같은 질병이 의심됩니다. 궁금하신 질병을 선택해주세요."
            return {"output": output, "disease": disease_names}

        else:
            print("증상을 이해할 수 없습니다. 다시 입력해주세요.")
            return {"output": "증상을 이해할 수 없습니다. 다시 입력해주세요."}
    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail=str(e))


# 실시간으로 챗봇화면에 챗봇 응답이 보여지도록 token 형태로 데이터 전달
async def generate_response(data: str):
    try:
        for token in chain.stream({"input": data}):
            print(token)
            yield token
    except Exception as e:
        yield f"Error: {str(e)}"


# 질병 관련 조언 API
@ai_router.post("/disease-advice")
async def disease_advice(request: ChatUserInput):
    data = request.input
    return StreamingResponse(generate_response(data), media_type="text/plain")
