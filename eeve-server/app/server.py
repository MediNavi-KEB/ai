from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from translate import chain, chain2
from message_dto import ChatUserInput
from rag import search, calculate_weighted_scores

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.post("/api/disease_recommendation")
async def disease_recommendation(request: ChatUserInput):
    try:
        # 디버깅을 위해 입력 내용 로그에 남김
        input_text = request.input
        print(f"Input to model: {input_text}")

        # faiss 검색
        search_results = search(input_text, top_k=100)
        # 거리가 30 이내인 결과 필터링
        filtered_results = search_results[search_results['거리'] < 30]

        # 필터링된 결과가 30개 이상인지 확인
        if len(filtered_results) >= 30:
            # 가중치 계산
            weighted_scores = calculate_weighted_scores(filtered_results)
            print("Weighted Scores:", weighted_scores)

            # 상위 3개 항목 추출
            top_diseases = weighted_scores.head(3)
            disease_names = top_diseases['병명'].tolist()

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


@app.post("/api/hospital_recommendation")
async def hospital_recommendation(request: ChatUserInput):
    print(request)
    disease = request.input
    result = chain.invoke({"input": disease})
    print(result)
    return {"description": result}


@app.post("/api/disease-advice")
async def deisease_advice(request: ChatUserInput):
    print(request)
    data = request.input
    result = chain2.invoke({"input": data})
    print(result)
    return {"description": result}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
