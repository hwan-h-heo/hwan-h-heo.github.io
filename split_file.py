import os
import copy
from bs4 import BeautifulSoup

def split_language_content(file_path):
    """
    Markdown 파일 내의 HTML 콘텐츠를 언어에 따라 분리합니다.
    - 'lang kor' 태그는 content-kor.md로 이동합니다.
    - 'lang eng' 태그는 원본 파일에 남습니다.
    - lang 클래스가 없는 태그는 양쪽 파일에 모두 유지됩니다.
    - 모든 'style' 속성은 제거됩니다.

    :param file_path: 처리할 원본 .md 파일의 경로
    """
    if not os.path.exists(file_path):
        print(f"오류: 파일 '{file_path}'를 찾을 수 없습니다.")
        return

    # 출력 파일 경로 설정
    directory = os.path.dirname(file_path)
    kor_output_path = os.path.join(directory, 'content-kor.md')

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"파일을 읽는 중 오류가 발생했습니다: {e}")
        return

    # 원본 콘텐츠를 BeautifulSoup 객체로 파싱
    soup = BeautifulSoup(content, 'html.parser')

    # --- 1. content-kor.md 파일 생성 로직 ---
    # 원본 soup를 깊은 복사하여 한국어 콘텐츠용 soup 생성
    kor_soup = copy.deepcopy(soup)
    
    # 한국어 soup에서 'lang eng' 클래스를 가진 모든 태그를 제거
    for tag in kor_soup.find_all(class_='lang eng'):
        tag.decompose()

    # 한국어 soup에서 모든 style 속성을 제거
    for tag in kor_soup.find_all(style=True):
        del tag['style']


    # --- 2. 원본 파일 수정 로직 ---
    # 원본 soup에서 'lang kor' 클래스를 가진 모든 태그를 제거
    for tag in soup.find_all(class_='lang kor'):
        tag.decompose()

    # 원본 soup에서 모든 style 속성을 제거
    for tag in soup.find_all(style=True):
        del tag['style']


    # --- 3. 파일 저장 ---
    try:
        # content-kor.md 파일 저장 (HTML 태그만 있는 경우 불필요한 <html><body> 태그 제외)
        with open(kor_output_path, 'w', encoding='utf-8') as f:
            f.write(kor_soup.prettify())

        # 원본 파일 덮어쓰기
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(soup.prettify())
            
        print(f"성공: '{kor_output_path}' 파일이 생성/업데이트되었습니다.")
        print(f"성공: 원본 파일 '{file_path}'이 업데이트되었습니다.")

    except Exception as e:
        print(f"파일을 쓰는 도중 오류가 발생했습니다: {e}")


if __name__ == '__main__':
    # 이 스크립트를 실행하려면 beautifulsoup4 라이브러리가 필요합니다.
    # 터미널에서 'pip install beautifulsoup4'를 실행하세요.
    
    input_file_path = input("처리할 마크다운(.md) 파일의 전체 경로를 입력하세요: ")
    split_language_content(input_file_path)