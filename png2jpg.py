from glob import glob
import os
from PIL import Image

images = sorted(glob('blogs/posts/230202_ngp/assets/*'))
for img in images:
    if 'png' in img:
        pil = Image.open(img)

        # print(pil.mode)
        if pil.mode == 'P':  # P 모드 처리
            pil = pil.convert('RGBA')
        if pil.mode == 'RGBA':
            # 흰색 배경으로 새 이미지 생성
            background = Image.new('RGB', pil.size, (255, 255, 255))
            background.paste(pil, mask=pil.split()[3])  # 알파 채널을 마스크로 사용
            jpg = background
        else:
            jpg = pil.convert('RGB')
        jpg.save(img.replace('png', 'jpg'))