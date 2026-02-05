import cv2
import numpy as np
import math
import os

def calculate_psnr(img1, img2):
    # img1 和 img2 是 BGR 图像，范围 [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

# 设置路径
current_dir = os.path.dirname(os.path.abspath(__file__))
blurry_path = os.path.join(current_dir, 'test_img', 'target.png')
torch_result_path = os.path.join(current_dir,'test_img', 'torch_result.jpg')
jittor_result_path = os.path.join(current_dir, 'test_img', 'jt_result.jpg')
blurry = cv2.imread(blurry_path) 
img_torch = cv2.imread(torch_result_path)
img_jt = cv2.imread(jittor_result_path)

if blurry is None or img_torch is None or img_jt is None:
    print("错误：请确保所有图片路径正确且尺寸一致。")
else:
    # 确保尺寸一致
    if blurry.shape != img_jt.shape:
        img_jt = cv2.resize(img_jt, (blurry.shape[1], blurry.shape[0]))
        img_torch = cv2.resize(img_torch, (blurry.shape[1], blurry.shape[0]))

    psnr_torch = calculate_psnr(blurry, img_torch)
    psnr_jt = calculate_psnr(blurry, img_jt)
    
    # 对比两个框架的一致性
    diff_psnr = calculate_psnr(img_torch, img_jt)

    print(f"--- 性能对比 ---")
    print(f"PyTorch vs GT PSNR: {psnr_torch:.4f} dB")
    print(f"Jittor  vs GT PSNR: {psnr_jt:.4f} dB")
    print(f"\n--- 迁移一致性验证 ---")
    print(f"PyTorch 与 Jittor 结果图的 PSNR: {diff_psnr:.4f} dB")
    print("(说明：如果该数值 > 40-50dB，说明两个框架的输出在像素级几乎完全相同)")