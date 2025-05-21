from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
import os

# [디렉토리 내 문서들을 순회하며 전부 임베딩 후 벡터 DB 에 저장]
# 디렉토리 내 모든 .txt 파일을 로드
docs = []
for filename in os.listdir('./my_docs'):
    if filename.endswith('.txt'):
        file_path = os.path.join('./my_docs', filename)
        loader = TextLoader(file_path, encoding="utf-8")
        docs.extend(loader.load())  # list of Document

# 텍스트 분할
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = splitter.split_documents(docs)

# 임베딩 및 VectorDB 저장
embedding = HuggingFaceEmbeddings(model_name="intfloat/e5-base-v2")
vectorstore = FAISS.from_documents(split_docs, embedding)
vectorstore.save_local("my_vectorstore")
