from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from search_documents import search_documents  # 위에서 만든 검색 함수 임포트

# GGUF 모델 경로 설정
model_path = "./Mistral-7B-Instruct-v0.1-GGUF/mistral-7b-instruct-v0.1.Q2_K.gguf"  # 모델 위치에 맞게 수정

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

# LangChain 체인 구성
result = prompt | llm
question = "railly 회사의 복지가 뭐가 있나요?"

# 검색된 문서 가져오기
search_results = search_documents(doc_folder='./my_docs')
context = "\n".join(search_results)  # 검색된 문서들을 하나의 문맥으로 결합

print("답변:", result.invoke({"question": question, "context": context}))