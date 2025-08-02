

import os
import re
import json
import shutil
from datetime import datetime
import markdown
from jinja2 import Environment, FileSystemLoader

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
POSTS_DIR = os.path.join(BASE_DIR, 'posts')
LAYOUTS_DIR = os.path.join(BASE_DIR, 'layouts')
JS_OUTPUT_PATH = os.path.join(BASE_DIR, 'js', 'posts-data.js')
SITE_OUTPUT_DIR = os.path.join(BASE_DIR, '_site')

# --- 1. DATA GATHERING & JS FILE GENERATION ---

def parse_metadata(file_path):
    metadata = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    meta_match = re.search(r'^---\n(.*?)\n---\n', content, re.DOTALL)
    if meta_match:
        meta_text = meta_match.group(1)
        for line in meta_text.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                metadata[key.strip()] = value.strip()
        content = content[meta_match.end():]
    else:
        lines = content.split('\n')
        meta_lines, content_lines, in_meta = [], [], True
        for line in lines:
            if ':' in line and in_meta:
                meta_lines.append(line)
            else:
                in_meta = False
                content_lines.append(line)
        for line in meta_lines:
            key, value = line.split(':', 1)
            metadata[key.strip()] = value.strip()
        content = '\n'.join(content_lines)

    return metadata, content

def get_all_posts_data():
    all_posts = []
    for post_id in os.listdir(POSTS_DIR):
        post_path = os.path.join(POSTS_DIR, post_id)
        if not (os.path.isdir(post_path) and (re.match(r'\d{6}_', post_id) or 'editor' in post_id)):
            continue

        post_info = {'id': post_id, 'languages': []}
        for lang_code in ['eng', 'kor']:
            md_path = os.path.join(post_path, f'content-{lang_code}.md')
            if os.path.exists(md_path):
                metadata, content = parse_metadata(md_path)
                post_info['languages'].append(lang_code)
                post_info[f'content_{lang_code}'] = content
                for key, value in metadata.items():
                    post_info[f'{key}_{lang_code}'] = value

        if not post_info['languages']:
            continue

        post_info['date'] = post_info.get('date_eng', post_info.get('date_kor', ''))
        post_info['series'] = post_info.get('series_eng', post_info.get('series_kor', None))
        post_info['category'] = post_info.get('category_eng', post_info.get('category_kor', 'post'))
        post_info['url'] = f'/posts/{post_id}/'

        if not post_info['date']:
            try:
                date_str = post_id.split('_')[0]
                post_info['date'] = datetime.strptime(date_str, '%y%m%d').strftime('%Y-%m-%d')
            except (ValueError, IndexError):
                post_info['date'] = '1970-01-01'

        all_posts.append(post_info)

    all_posts.sort(key=lambda x: x['date'], reverse=True)
    return all_posts

def generate_posts_data_js(posts):
    posts_for_js = []
    for post in posts:
        js_post = {
            'id': post['id'], 'date': post['date'], 'series': post['series'],
            'category': post['category'], 'languages': post['languages'],
            'title_eng': post.get('title_eng', ''), 'subtitle_eng': post.get('subtitle_eng', ''),
            'title_kor': post.get('title_kor', ''), 'subtitle_kor': post.get('subtitle_kor', ''),
        }
        posts_for_js.append(js_post)

    series_info = {
        'nerf-and-gs': {'eng': 'Radiance Fields & Gaussian Splatting', 'kor': 'Radiance Fields & Gaussian Splatting'},
        '3d-generation': {'eng': '3D Generative AI', 'kor': '3D 생성 AI'},
        'web-3d': {'eng': '3D in Web', 'kor': '웹에서 3D 구현하기'},
        'linear-algebra': {'eng': 'Linear Algrebra for Deeplearning', 'kor': '딥러닝을 위한 선형대수'},
        'computer-vision': {'eng': 'Classical Computer Vision', 'kor': '고전 Computer Vision'},
    }

    js_content = f"const postsData = {json.dumps(posts_for_js, indent=4)};\n\nconst seriesInfo = {json.dumps(series_info, indent=4)};"
    with open(JS_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        f.write(js_content)
    print(f"Successfully generated {JS_OUTPUT_PATH}")

# --- 2. STATIC SITE GENERATION ---

def generate_static_site(posts):
    env = Environment(loader=FileSystemLoader(LAYOUTS_DIR))
    if os.path.exists(SITE_OUTPUT_DIR):
        shutil.rmtree(SITE_OUTPUT_DIR)
    os.makedirs(SITE_OUTPUT_DIR)

    post_template = env.get_template('post.html')
    for post in posts:
        for lang in post['languages']:
            post_output_dir = os.path.join(SITE_OUTPUT_DIR, 'posts', post['id'], lang)
            os.makedirs(post_output_dir, exist_ok=True)

            content_html = markdown.markdown(post[f'content_{lang}'])
            
            available_languages = []
            for other_lang in post['languages']:
                if other_lang != lang:
                    available_languages.append({
                        'name': 'English' if other_lang == 'eng' else '한국어',
                        'url': f'../{other_lang}/',
                    })

            render_context = {
                'title': post.get(f'title_{lang}', ''),
                'author': post.get(f'author_{lang}', 'Hwan Heo'),
                'date': post['date'],
                'content': content_html,
                'available_languages': available_languages
            }
            
            with open(os.path.join(post_output_dir, 'index.html'), 'w', encoding='utf-8') as f:
                f.write(post_template.render(render_context))

    print(f"Generated {sum(len(p['languages']) for p in posts)} post pages in {len(posts)} posts.")

    home_template = env.get_template('home.html')
    home_posts_context = []
    for post in posts:
        default_lang = 'eng' if 'eng' in post['languages'] else 'kor'
        home_posts_context.append({
            'title': post.get(f'title_{default_lang}', 'No Title'),
            'subtitle': post.get(f'subtitle_{default_lang}', ''),
            'date': post['date'],
            'url': f'/posts/{post["id"]}/{default_lang}/'
        })

    with open(os.path.join(SITE_OUTPUT_DIR, 'index.html'), 'w', encoding='utf-8') as f:
        f.write(home_template.render(posts=home_posts_context))
    
    print("Generated home page.")
    print(f"\nBuild finished. Static site is in: {SITE_OUTPUT_DIR}")

def main():
    all_posts_data = get_all_posts_data()
    generate_posts_data_js(all_posts_data)
    generate_static_site(all_posts_data)

if __name__ == "__main__":
    main()
