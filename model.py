import make_vectordb
from langchain_core.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
import joblib

retriever = make_vectordb.vectordb.as_retriever(search_kwargs={'k':2})

template = """
당신은 토스뱅크 비상금 대출에 대해 설명하는 챗봇입니다. 주어진 검색 결과를 바탕으로 답변하세요.
검색 결과에 없는 내용이라면 답변할 수 없다고 하세요. 존댓말로 정중하게 답변하세요.

{context}

Question : {question}
Answer :
"""
input_variables = ["context", "question"]

prompt = PromptTemplate(template=template, input_variables=input_variables)

llm = ChatOpenAI(
    model_name = 'gpt-4o',
    streaming=True,
    # temparature=0,
    callbacks=[StreamingStdOutCallbackHandler()]
)
query="토스뱅크 비상금 대출에 대해 알려줘"
def qa_chain(query):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type_kwargs={"prompt":prompt},
        retriever=retriever, # 검색기의 조회 결과를 {context}에 넣어준다.
        return_source_documents=True)
    return qa_chain(query)

def chatbot_message(query):
    chatbot_response=qa_chain(query)
    return chatbot_response['result']


# def respond():
#     query=request.json_
#     chatbot_response = qa_chain(query)
#     return chatbot_response
    


