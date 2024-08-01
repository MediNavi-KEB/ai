from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from translate import chain  # AI 모델의 chain 모듈 import
from message_dto import Input_ms  # AI 모델의 DTO 모듈 import
from rag import search  # AI 모델의 rag 모듈 import
from db.session import get_db
from db.model.user import User  # User 모델을 import

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
async def invoke_chain(data: Input_ms, db: Session = Depends(get_db)):
    try:
        print("Received request:", data)
        if not data.messages or not isinstance(data.messages, list):
            raise ValueError("Invalid input: 'messages' must be a non-empty list")

        # 디버깅을 위해 입력 내용 로그에 남김
        input_text = data.messages[0].content
        print(f"Input to model: {input_text}")

        # faiss index 사용
        additional_info = search(input_text, 10)
        print(additional_info)

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
