import jittor as jt
from jittor.dataset import Dataset
import os
import cv2
import numpy as np
import random

class PairedImageDataset(Dataset):
    def __init__(self, root_dir, mode='train', patch_size=256, batch_size=4, split_ratio=0.9):
        super().__init__()
        self.mode = mode
        self.patch_size = patch_size
        self.batch_size = batch_size
        
        self.target_dir = os.path.join(root_dir, 'target')
        self.input_dir = os.path.join(root_dir, 'input')
        
        # 获取所有图片文件名
        all_files = sorted([x for x in os.listdir(self.input_dir) 
                            if x.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        split_idx = int(len(all_files) * split_ratio)
        if mode == 'train':
            self.image_filenames = all_files[:split_idx] 
            self.shuffle = True 
        else:
            self.image_filenames = all_files[split_idx:] 
            self.shuffle = False 
            
        self.total_len = len(self.image_filenames)
        print(f"[{mode}] 加载数据集: {len(self.image_filenames)} 张图片")
        
        self.set_attrs(batch_size=self.batch_size, total_len=self.total_len, shuffle=self.shuffle)

    def __getitem__(self, index):
        filename = self.image_filenames[index]
        input_path = os.path.join(self.input_dir, filename)
        target_path = os.path.join(self.target_dir, filename)
        
        img_input = cv2.imread(input_path)
        img_target = cv2.imread(target_path)
        
        # 训练模式：随机裁剪
        if self.mode == 'train':
            H, W, _ = img_input.shape
            if H >= self.patch_size and W >= self.patch_size:
                rnd_h = random.randint(0, H - self.patch_size)
                rnd_w = random.randint(0, W - self.patch_size)
                img_input = img_input[rnd_h:rnd_h+self.patch_size, rnd_w:rnd_w+self.patch_size, :]
                img_target = img_target[rnd_h:rnd_h+self.patch_size, rnd_w:rnd_w+self.patch_size, :]
            
            # 翻转增强
            if random.random() < 0.5:
                img_input = np.fliplr(img_input)
                img_target = np.fliplr(img_target)
        
        # 验证模式：为了加快和不超出显存，做中心裁剪
        elif self.mode == 'val':
            H, W, _ = img_input.shape
            # 裁剪中心区域
            start_h = (H - self.patch_size) // 2
            start_w = (W - self.patch_size) // 2
            if start_h >= 0 and start_w >= 0:
                img_input = img_input[start_h:start_h+self.patch_size, start_w:start_w+self.patch_size, :]
                img_target = img_target[start_h:start_h+self.patch_size, start_w:start_w+self.patch_size, :]

        # BGR -> RGB -> Norm -> CHW
        img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_target = cv2.cvtColor(img_target, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        img_input = np.transpose(img_input, (2, 0, 1))
        img_target = np.transpose(img_target, (2, 0, 1))
        
        return img_input.copy(), img_target.copy()
    

# --- 测试模块 ---
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    data_root = "/home/liu/projects/dl-sprout/NAFNet/datasets/GoPro/train"
    print(f"正在测试数据路径: {data_root}")
    
    try:
        ds = PairedImageDataset(data_root, mode='train', patch_size=256, batch_size=4)
        
        for inputs, targets in ds:
            print(f"\n成功读取一个 Batch!")
            print(f"Input Shape: {inputs.shape} (N, C, H, W)")
            print(f"Target Shape: {targets.shape} (N, C, H, W)")
            print(f"Input Range: [{inputs.min():.2f}, {inputs.max():.2f}]")
            
            # 可视化第一张图
            img_show = inputs[0].transpose(1, 2, 0) # CHW -> HWC
            plt.imshow(img_show)
            plt.title("Cropped Input Sample")
            plt.show()
            
            # 只测一次
            break
            
    except Exception as e:
        print(f"\n❌ 测试失败，错误信息: {e}")
        print("请检查你的文件夹结构是否包含 'input' 和 'target' 子文件夹")