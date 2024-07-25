from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# EEVE 모델 설정
llm_eeve = ChatOllama(model='EEVE:latest') # EEVE-Korean-10.88:latest

# # 프롬프트 템플릿 설정
# prompt = ChatPromptTemplate.from_template(
#     "You are a doctor.,"
#     "When a user asks about a disease, tell them the symptoms of it.,"
#     "When the user talks about a symptom, you can predict the corresponding disease and tell them.,"
#     "You can only speak in 10 sentences or less.,"
#     "You only translate user input,"
#     "No additional information is provided,"
#     "you do not respond to the user input: \n{input}"
# )

prompt = ChatPromptTemplate.from_template(
    "당신은 의사입니다.,"
    "아래의 증상별 질병 기반으로만 가능한 질병들을 예측하여 알려줍니다: {additional_info}.,"
    "각 질병에 대한 간략한 설명을 포함합니다.,"
    "출력에 거리를 포함하지 않습니다.,"
    "10문장 이내로만 대답할 수 있습니다.,"
    "사용자의 입력에만 응답합니다: \n {input}"
)

# LangChain 표현식 설정
chain = prompt | llm_eeve | StrOutputParser()