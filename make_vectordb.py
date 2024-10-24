from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

import os

OPENAI_API_KEY = "OPENAI_API_KEY"
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

# 데이터 불러오기
loader = PyPDFLoader('/home/ubuntu/working/project_file/data/토스뱅크 _ 토스뱅크 비상금 대출-병합됨.pdf')
texts = loader.load_and_split()

# Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=30
)
texts = text_splitter.split_documents(texts)

# Embedding
embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

# VectorDB 구성
vectordb = Chroma.from_documents(
    documents=texts,
    embedding=embedding
)
