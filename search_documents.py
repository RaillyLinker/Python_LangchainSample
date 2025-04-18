import os

def search_documents(doc_folder: str):
    # doc_folder 내의 모든 .txt 파일을 읽어서 내용 합치기
    documents = []

    # my_docs 폴더 내의 모든 파일을 확인
    for filename in os.listdir(doc_folder):
        file_path = os.path.join(doc_folder, filename)

        # .txt 파일만 처리
        if filename.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as file:
                documents.append(file.read())  # 파일 내용 읽기

    # 모든 문서를 하나의 문자열로 결합
    return documents

# 테스트
search_results = search_documents(doc_folder='./my_docs')
print(search_results)
