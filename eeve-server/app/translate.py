from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# EEVE 모델 설정
llm_eeve = ChatOllama(model='EEVE-Korean-10.88:latest')

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

# 프롬프트 템플릿 설정
prompt = ChatPromptTemplate.from_template(
    "당신은 의사입니다.,"
    "사용자가 질병에 대해 물어보면 그 증상에 대해 설명해줍니다.,"
    "사용자가 증상에 대해 이야기하면 해당 증상을 기반으로 가능한 질병을 예측하여 알려줍니다.,"
    "10문장 이내로만 대답할 수 있습니다.,"
    "추가적인 정보는 제공하지 않습니다.,"
    "사용자의 입력에만 응답합니다: \n{input}"
)

# LangChain 표현식 설정
chain = prompt | llm_eeve | StrOutputParser()
