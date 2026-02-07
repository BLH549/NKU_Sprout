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
    
    # --- 1. 路径与配置准备 ---
    # 定义要读取的图片列表
    img_names = ['test1.png', 'test2.png']
    torch_weights = os.path.join(current_dir, 'saved_models', 'NAFNet-GoPro-width32.pth')
    jittor_weights = os.path.join(current_dir, 'training_results', 'NAFNet_recovered_best.pkl')

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

    # 准备绘图画布 (2行3列)
    plt.figure(figsize=(18, 10))
    titles = ['Input (Blurry)', 'PyTorch Output', 'Jittor Output']

    # --- 4. 循环处理每一张图片 ---
    for idx, img_name in enumerate(img_names):
        img_path = os.path.join(current_dir, 'test_img', img_name)
        print(f"\n处理图片: {img_name}")

        # 读取并处理图像
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"无法读取测试图片: {img_path}，跳过该图。")
            continue
            
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # 归一化并增加 Batch 维度 (N, C, H, W) 
        inp_np = img_rgb.astype(np.float32) / 255.0
        inp_np = np.transpose(inp_np, (2, 0, 1))[None, ...]

        # --- 执行推理 ---
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

        # --- 结果后处理 ---
        t_res = (np.clip(t_res, 0, 1) * 255.0).astype(np.uint8)
        j_res = (np.clip(j_res, 0, 1) * 255.0).astype(np.uint8)

        # 计算 PSNR (相似度)
        mse = np.mean((t_res.astype(np.float32) - j_res.astype(np.float32)) ** 2)
        psnr = 10 * np.log10((255.0 ** 2) / mse) if mse != 0 else float('inf')
        print(f'{img_name} - PyTorch 与 Jittor 输出的 PSNR 相似度: {psnr:.2f} dB')

        # --- 5. 绘制到子图中 ---
        # images 列表包含当前图片的三种状态
        row_images = [img_rgb, t_res, j_res]
        for i in range(3):
            # idx * 3 + i + 1 计算子图位置：
            # test1: 1, 2, 3
            # test2: 4, 5, 6
            plt.subplot(2, 3, idx * 3 + i + 1)
            plt.imshow(row_images[i])
            if idx == 0:
                plt.title(titles[i], fontsize=14)
            if i == 0:
                plt.ylabel(img_name, fontsize=14, fontweight='bold')
            plt.xticks([]) # 隐藏坐标刻度
            plt.yticks([])

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    run_comparison()