import os
import re
import requests
from urllib.parse import urlparse

def process_blog_posts(root_dir='posts'):
    """
    지정된 루트 디렉토리의 모든 블로그 게시물을 처리합니다.
    """
    if not os.path.isdir(root_dir):
        print(f"오류: '{root_dir}' 디렉토리를 찾을 수 없습니다. 스크립트가 'posts' 폴더와 동일한 위치에 있는지 확인하세요.")
        return

    print(f"'{root_dir}' 디렉토리에서 게시글 스캔을 시작합니다...")

    # posts 폴더 내의 모든 항목을 순회
    for post_folder_name in os.listdir(root_dir):
        post_path = os.path.join(root_dir, post_folder_name)

        # 디렉토리인 경우에만 처리
        if os.path.isdir(post_path):
            md_file_path = os.path.join(post_path, 'content.md')

            if os.path.isfile(md_file_path):
                print(f"\n--- '{post_folder_name}' 게시글 처리 중 ---")
                process_markdown_file(post_path, post_folder_name, md_file_path)
            else:
                print(f"경고: '{post_path}'에서 content.md 파일을 찾을 수 없습니다.")

def process_markdown_file(post_path, post_folder_name, md_file_path):
    """
    단일 마크다운 파일을 읽고, 외부 이미지를 다운로드한 후, 경로를 업데이트합니다.
    """
    try:
        with open(md_file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 'http' 또는 'www'로 시작하는 외부 이미지 링크를 찾는 정규식
        # 지원하는 확장자: png, jpg, jpeg, webp, gif
        img_pattern = re.compile(r'<img[^>]+src="((?:https?:\/\/|www\.)[^"]+\.(?:png|jpg|jpeg|webp|gif))"', re.IGNORECASE)
        external_images = img_pattern.findall(content)

        if not external_images:
            print("외부 이미지를 찾지 못했습니다. 다음 게시글로 넘어갑니다.")
            return

        print(f"총 {len(external_images)}개의 외부 이미지를 찾았습니다.")

        # 'assets' 폴더 생성 (없을 경우)
        assets_dir = os.path.join(post_path, 'assets')
        os.makedirs(assets_dir, exist_ok=True)

        has_changed = False
        # 중복된 URL은 한 번만 처리하기 위해 set() 사용
        for img_url in set(external_images):
            try:
                # URL에서 파일 이름 추출
                parsed_url = urlparse(img_url)
                original_image_name = os.path.basename(parsed_url.path)

                if not original_image_name:
                    print(f"경고: URL에서 유효한 파일 이름을 추출할 수 없습니다: {img_url}")
                    continue

                image_name = original_image_name
                local_image_path = os.path.join(assets_dir, image_name)
                counter = 1

                # ★★★ 파일명 중복 처리 로직 ★★★
                # 파일명이 겹치면 뒤에 숫자를 붙여 새로운 파일명 생성 (예: image-1.png)
                while os.path.exists(local_image_path):
                    name, ext = os.path.splitext(original_image_name)
                    image_name = f"{name}-{counter}{ext}"
                    local_image_path = os.path.join(assets_dir, image_name)
                    counter += 1
                
                if original_image_name != image_name:
                    print(f"알림: 파일명이 중복되어 '{image_name}'으로 변경합니다.")

                # 이미지 다운로드
                print(f"다운로드 중: {img_url}")
                response = requests.get(img_url, stream=True, headers={'User-Agent': 'Mozilla/5.0'})
                response.raise_for_status()  # HTTP 오류 발생 시 예외 발생

                with open(local_image_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"저장 완료: {local_image_path}")

                # 마크다운에서 사용할 새로운 로컬 경로 생성 (수정된 image_name 사용)
                new_local_path = f'./{post_folder_name}/assets/{image_name}'

                # 원본 content의 이미지 URL을 새로운 로컬 경로로 교체
                # 원본 URL(img_url)을 찾아 고유하게 생성된 경로(new_local_path)로 변경
                content = content.replace(img_url, new_local_path)
                print(f"경로 업데이트: '{img_url}' -> '{new_local_path}'")
                has_changed = True

            except requests.exceptions.RequestException as e:
                print(f"이미지 다운로드 오류: {img_url} - {e}")
            except Exception as e:
                print(f"이미지 처리 중 예상치 못한 오류 발생: {img_url} - {e}")

        # 변경 사항이 있을 경우에만 파일에 다시 쓰기
        if has_changed:
            with open(md_file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"성공: '{md_file_path}' 파일이 업데이트되었습니다.")
        else:
            print("업데이트할 내용이 없습니다.")

    except IOError as e:
        print(f"파일 읽기/쓰기 오류: {md_file_path} - {e}")
    except Exception as e:
        print(f"'{md_file_path}' 처리 중 오류 발생: {e}")


if __name__ == "__main__":
    # 스크립트 실행
    process_blog_posts()
    print("\n--- 모든 작업이 완료되었습니다. ---")