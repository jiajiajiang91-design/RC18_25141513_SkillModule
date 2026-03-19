from icrawler.builtin import BingImageCrawler
import os

base_dir = r'D:\Claude_jiajia\comfyui_reference'

keywords = [
    'weed plant closeup nature',
    'wildflower closeup isolated',
    'moss closeup macro',
    'ivy plant closeup',
]

for kw in keywords:
    folder = os.path.join(base_dir, kw.replace(' ', '_'))
    os.makedirs(folder, exist_ok=True)
    crawler = BingImageCrawler(storage={'root_dir': folder})
    crawler.crawl(keyword=kw, max_num=50)

print("全部下载完成！")