from PIL import Image
import os

input_dir = r'D:\Claude_jiajia\comfyui_reference'
output_dir = r'D:\Claude_jiajia\comfyui_reference_128'
os.makedirs(output_dir, exist_ok=True)

def crop_and_resize(img_path, save_path, size=128, bottom_crop_ratio=0.10):
    try:
        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        
        # 裁掉底部水印
        h_crop = int(h * (1 - bottom_crop_ratio))
        img = img.crop((0, 0, w, h_crop))
        
        # 从中心取正方形
        w, h = img.size
        min_side = min(w, h)
        left = (w - min_side) // 2
        top = (h - min_side) // 2
        img = img.crop((left, top, left + min_side, top + min_side))
        
        # 缩放到128x128
        img = img.resize((size, size), Image.LANCZOS)
        img.save(save_path)
        return True
    except Exception as e:
        print(f"跳过 {img_path}: {e}")
        return False

# 处理所有图片（包括子文件夹）
count = 0
for root, dirs, files in os.walk(input_dir):
    for fname in files:
        if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            src = os.path.join(root, fname)
            dst = os.path.join(output_dir, f'{count:06d}.jpg')
            if crop_and_resize(src, dst):
                count += 1

print(f"完成！共处理 {count} 张图片")
print(f"保存到: {output_dir}")