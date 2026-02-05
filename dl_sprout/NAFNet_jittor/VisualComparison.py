import torch
import jittor as jt
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

from torch_model import NAFNet as TorchNAFNet
from model import NAFNet as JtNAFNet

# 设置硬件
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
jt.flags.use_cuda = 1

def run_comparison():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # --- 1. 路径准备 ---
    img_path = os.path.join(current_dir, 'test_img', 'test.png')
    
    torch_weights = os.path.join(current_dir, 'saved_models', 'NAFNet-GoPro-width32.pth')
    jittor_weights = os.path.join(current_dir, 'saved_models', 'jt_NAFNet.pkl')

    # --- 2. 初始化并加载 PyTorch 模型 ---
    print("正在加载 PyTorch 模型...")
    t_model = TorchNAFNet(img_channel=3, width=32, middle_blk_num=1,
                          enc_blk_nums=[1, 1, 1, 28], dec_blk_nums=[1, 1, 1, 1])
    t_ckpt = torch.load(torch_weights, map_location='cpu')
    t_state = t_ckpt['params'] if 'params' in t_ckpt else t_ckpt
    t_model.load_state_dict(t_state)
    t_model.to(device).eval()

    # --- 3. 初始化并加载 Jittor 模型 ---
    print("正在加载 Jittor 模型...")
    j_model = JtNAFNet(img_channel=3, width=32, middle_blk_num=1,
                       enc_blk_nums=[1, 1, 1, 28], dec_blk_nums=[1, 1, 1, 1])
    j_model.load(jittor_weights)
    j_model.eval()

    # --- 4. 图像预处理 ---
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"无法读取测试图片: {img_path}")
        return
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # 归一化并增加 Batch 维度 (N, C, H, W) 
    inp_np = img_rgb.astype(np.float32) / 255.0
    inp_np = np.transpose(inp_np, (2, 0, 1))[None, ...]

    # --- 5. 执行推理 ---
    print("执行推理中...")
    # PyTorch 推理
    with torch.no_grad():
        t_inp = torch.from_numpy(inp_np).to(device)
        t_out = t_model(t_inp)
        t_res = t_out.squeeze().permute(1, 2, 0).cpu().numpy()

    # Jittor 推理
    with jt.no_grad():
        j_inp = jt.array(inp_np)
        j_out = j_model(j_inp)
        j_res = j_out.squeeze().transpose(1, 2, 0).numpy()

    # --- 6. 结果后处理 ---
    t_res = (np.clip(t_res, 0, 1) * 255.0).astype(np.uint8)
    j_res = (np.clip(j_res, 0, 1) * 255.0).astype(np.uint8)

    # --- 7. 可视化对比 ---
    titles = ['Input (Blurry)', 'PyTorch Output', 'Jittor Output']
    images = [img_rgb, t_res, j_res]

    plt.figure(figsize=(20, 7))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.imshow(images[i])
        plt.title(titles[i], fontsize=15)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

    # 计算两张图片的相似度（psdb）
    mse = np.mean((t_res - j_res) ** 2)
    psdb = 10 * np.log10((255.0 ** 2) / mse) if mse != 0 else float('inf')
    print(f'PyTorch 与 Jittor 输出的 PSDB 相似度: {psdb:.2f} dB')

if __name__ == '__main__':
    run_comparison()