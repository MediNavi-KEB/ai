from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# EEVE 모델 설정
llm_eeve = ChatOllama(model='EEVE:latest') # EEVE-Korean-10.88:latest

prompt = ChatPromptTemplate.from_template(
    "{additional_info}에 대한 정보를 3줄로 요약해줘. 다만 앞에 숫자를 써주지 말고 그냥 줄글로 요약해줘"
)

# LangChain 표현식 설정
chain = prompt | llm_eeve | StrOutputParser()