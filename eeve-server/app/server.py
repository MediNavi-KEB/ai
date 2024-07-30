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

        search_results = search(input_text, top_k=100)
        print(search_results)
        weighted_scores = calculate_weighted_scores(search_results)
        print(weighted_scores)
        additional_info = weighted_scores.iloc[0]

        result = chain.invoke({"input": input_text, "additional_info": additional_info})

        # 모델의 응답 로그에 남김
        print(f"Raw result from model: {result}")

        result = result.split("참고:")[0].strip()

        # 결과 데이터 길이 제한
        max_length = 500  # 원하는 길이로 설정
        if len(result) > max_length:
            result = result[:max_length] + "..."

        print("Result:", result)
        return {"output": result}
    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
