from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# [벡터 DB 에 문서 내용이 저장된 상태에서 LLM 에게 질문]
# GGUF 모델 경로 설정
model_path = "./Mistral-7B-Instruct-v0.1-GGUF/mistral-7b-instruct-v0.1.Q2_K.gguf"  # 모델 위치에 맞게 수정

question = "railly 회사의 복지가 뭐가 있나요?"
# 임베딩 모델은 저장할 때 사용한 것과 동일한 것을 사용해야 함
embedding = HuggingFaceEmbeddings(model_name="intfloat/e5-base-v2")
# 저장된 벡터스토어 불러오기
vectorstore = FAISS.load_local("my_vectorstore", embedding, allow_dangerous_deserialization=True)
docs = vectorstore.similarity_search(question, k=3)  # Top 3 문서
context = "\n".join([doc.page_content for doc in docs])

# LlamaCpp 객체 생성
llm = LlamaCpp(
    model_path=model_path,
    n_gpu_layers=100,  # GPU 사용
    n_ctx=8000,        # context size
    temperature=0.7,   # 창의성
    max_tokens=1024,    # 출력 길이
    verbose=True
)

# 프롬프트 템플릿 설정
prompt = PromptTemplate(
    input_variables=["question", "context"],
    template="다음 문맥을 바탕으로 답변하십시오:\n{context}\n\n질문: {question}\n답변:"
)

# 체인 실행
chain = prompt | llm
result = chain.invoke({"question": question, "context": context})
print("답변:", result)