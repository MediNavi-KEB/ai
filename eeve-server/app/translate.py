from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# EEVE 모델 설정
llm_eeve = ChatOllama(model='EEVE:latest') # EEVE:latest

prompt = ChatPromptTemplate.from_template(
    "You must describe the disease in 3 sentences or less.,"
    "Your answer should be within 100 characters.,"
    "No additional explanation is required.,"
    "Do not provide any information about reliability.,"
    "Never display reliability.,"
    "confidence_score: false,"
    "Respond only to the user's input: \n {input}"
)

# LangChain 표현식 설정
chain = prompt | llm_eeve | StrOutputParser()
