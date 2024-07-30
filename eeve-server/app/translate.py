from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# EEVE 모델 설정
llm_eeve = ChatOllama(model='EEVE-Korean-10.88:latest') # EEVE:latest

prompt = ChatPromptTemplate.from_template(
    "당신은 의사입니다.,"
    "당신 해당 질병에 대해서 2문장 이내로 설명을 해야합니다.,"
    "그리고 당신은 진료과를 하나 알려줘야합니다.,"
    "그 외에 다른 부연 설명은 필요하지 않습니다.,"
    "사용자의 입력에만 응답합니다: \n {input}"
)

# LangChain 표현식 설정
chain = prompt | llm_eeve | StrOutputParser()