from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from translate import chain
from message_dto import Input_ms
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

@app.post("/api/translate")
async def invoke_chain(data: Input_ms):
    try:
        print("Received request:", data)
        if not data.messages or not isinstance(data.messages, list):
            raise ValueError("Invalid input: 'messages' must be a non-empty list")

        # 디버깅을 위해 입력 내용 로그에 남김
        input_text = data.messages[0].content
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

            return {"output": disease_names}
        else:
            print("증상을 이해할 수 없습니다. 다시 입력해주세요.")
            return {"output": "증상을 이해할 수 없습니다. 다시 입력해주세요."}
    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
