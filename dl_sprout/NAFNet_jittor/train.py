import jittor as jt
from jittor import nn, optim
import os, math, numpy as np
from model import NAFNet
from dataset import PairedImageDataset

jt.flags.use_cuda = 1

# 参数 
BATCH_SIZE = 4      
LR = 5e-5          
EPOCHS = 10         
PATCH_SIZE = 256    
NOISE_STD = 0.03    

current_dir = os.path.dirname(os.path.abspath(__file__))
#DATA_ROOT = os.path.join(current_dir, '../NAFNet/datasets/GoPro/train')
DATA_ROOT = "/home/liu/projects/dl-sprout/NAFNet/datasets/GoPro/train"
PRETRAINED_PATH = os.path.join(current_dir, 'saved_models/jt_NAFNet.pkl')
SAVE_DIR = os.path.join(current_dir, 'training_results')
os.makedirs(SAVE_DIR, exist_ok=True)

def add_gaussian_noise(model, std=0.01):
    print(f">>> 正在为模型参数注入标准差为 {std} 的高斯噪声...")
    params = model.state_dict()
    for name, param in params.items():
        # 仅针对卷积层的权重注入噪声
        if 'weight' in name and ('conv' in name or 'intro' in name or 'ending' in name):
            noise = jt.randn(param.shape) * std
            # 使用 assign 更新参数值，这样能保持 Jittor 变量的引用关系
            param.assign(param + noise)
    print(">>> 噪声注入完成。")


def calculate_psnr_numpy(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    return 20 * math.log10(255.0 / math.sqrt(mse)) if mse > 0 else 100

def validate(model, val_loader):
    model.eval()
    total_psnr = 0.0
    count = 0
    with jt.no_grad():
        for input, target in val_loader:
            output = model(input)
            out_np = np.clip(output.numpy(), 0, 1) * 255.0
            tar_np = target.numpy() * 255.0
            for i in range(out_np.shape[0]):
                total_psnr += calculate_psnr_numpy(np.transpose(out_np[i], (1,2,0)), 
                                                   np.transpose(tar_np[i], (1,2,0)))
                count += 1
    avg_psnr = total_psnr / count
    return avg_psnr

def train():
    model = NAFNet(img_channel=3, width=32, middle_blk_num=1,
                   enc_blk_nums=[1, 1, 1, 28], dec_blk_nums=[1, 1, 1, 1])
    
    # A. 加载预训练模型
    if os.path.exists(PRETRAINED_PATH):
        model.load(PRETRAINED_PATH)
        print("成功加载预训练模型。")
    
    # B. 注入噪声 (在加载之后，训练之前)
    add_gaussian_noise(model, std=NOISE_STD)

    # C. 准备数据
    train_dataset = PairedImageDataset(DATA_ROOT, mode='train', 
                                       patch_size=PATCH_SIZE, batch_size=BATCH_SIZE, split_ratio=0.9)
    val_dataset = PairedImageDataset(DATA_ROOT, mode='val', 
                                     patch_size=PATCH_SIZE*2, batch_size=1, split_ratio=0.9)


    # D. 训练前的“初始评估” (Persuasiveness 核心)
    print("\n--- 训练前初始评估 ---")
    init_psnr = validate(model, val_dataset)
    print(f"注入噪声后的初始 PSNR: {init_psnr:.4f} dB")

    optimizer = optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.9))
    criterion = nn.MSELoss()
    
    best_psnr = init_psnr

    for epoch in range(EPOCHS):

        model.train()
        for i, (input, target) in enumerate(train_dataset):
            output = model(input)
            loss = criterion(output, target)
            optimizer.step(loss)
            if i % 100 == 0:
                print(f"Epoch [{epoch+1}] Loss: {loss.item():.6f}")

        # 每个 Epoch 评估一次，观察恢复情况
        current_psnr = validate(model, val_dataset)
        print(f"Epoch [{epoch+1}] 验证集 PSNR: {current_psnr:.4f} dB (提升: {current_psnr - init_psnr:.4f})")
        
        if current_psnr > best_psnr:
            best_psnr = current_psnr
            model.save(os.path.join(SAVE_DIR, "nafnet_recovered_best.pkl"))

if __name__ == '__main__':
    train()