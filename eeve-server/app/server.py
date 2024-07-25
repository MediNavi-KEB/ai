from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Union
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from translate import chain

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


class MessageBase(BaseModel):
    content: str
    additional_kwargs: dict = {}
    response_metadata: dict = {}
    name: str
    id: str
    example: bool = False


class HumanMessage(MessageBase):
    type: str = "human"


class AIMessage(MessageBase):
    type: str = "ai"
    tool_calls: List = []
    invalid_tool_calls: List = []
    usage_metadata: dict = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0
    }


class SystemMessage(MessageBase):
    type: str = "system"


class Input_ms(BaseModel):
    messages: List[Union[HumanMessage, AIMessage, SystemMessage]]


@app.post("/api/translate")
async def invoke_chain(data: Input_ms):
    try:
        print("Received request:", data)
        if not data.messages or not isinstance(data.messages, list):
            raise ValueError("Invalid input: 'messages' must be a non-empty list")

        # 디버깅을 위해 입력 내용 로그에 남김
        input_text = data.messages[0].content
        print(f"Input to model: {input_text}")

        result = chain.invoke({"input": input_text})

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
