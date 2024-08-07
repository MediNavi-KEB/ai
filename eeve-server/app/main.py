from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from server import ai_router

# 출처 등록 (CORS)
origins = ["*"]

app = FastAPI()

app.include_router(ai_router, prefix="/ai")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
