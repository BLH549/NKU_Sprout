import jittor as jt
from jittor.dataset import Dataset
from jittor.dataset import DataLoader
import os
import random
import numpy as np
from PIL import Image

jt.flags.use_cuda = 1

def populate_train_list(orig_images_path, hazy_images_path, split_ratio=0.9):

    print("Populating train/val lists...")
    tmp_dict = {}
    
    # 遍历所有有雾图像
    for hazy_filename in os.listdir(hazy_images_path):
        
        # 确保是jpg文件
        if not hazy_filename.endswith('.jpg'):
            continue
            
        parts = hazy_filename.split('_')

        # 组合成原图的key，例如 "NYU2_1.jpg"
        key = parts[0] + '_' + parts[1] + '.jpg'
        
        if key not in tmp_dict:
            tmp_dict[key] = []
        tmp_dict[key].append(hazy_filename)
    
    all_keys = list(tmp_dict.keys())
    if not all_keys:
        raise FileNotFoundError(f"No valid images found or keys matched in {hazy_images_path}")
        
    random.shuffle(all_keys)
    
    # 分割key列表
    split_idx = int(len(all_keys) * split_ratio)
    train_keys = all_keys[:split_idx]
    val_keys = all_keys[split_idx:]

    train_list = []
    val_list = []

    # 5. 根据分割好的keys生成最终的文件路径列表
    for key in train_keys:
        orig_path = os.path.join(orig_images_path, key)
        for hazy_filename in tmp_dict[key]:
            hazy_path = os.path.join(hazy_images_path, hazy_filename)
            train_list.append((orig_path, hazy_path))
            
    for key in val_keys:
        orig_path = os.path.join(orig_images_path, key)
        for hazy_filename in tmp_dict[key]:
            hazy_path = os.path.join(hazy_images_path, hazy_filename)
            val_list.append((orig_path, hazy_path))
    
    random.shuffle(train_list)
    random.shuffle(val_list)

    print(f"Total unique keys: {len(all_keys)}")
    print(f"Train keys: {len(train_keys)}, Val keys: {len(val_keys)}")
    print(f"Total training images: {len(train_list)}")
    print(f"Total validation images: {len(val_list)}")

    return train_list, val_list

class MyDataset(Dataset):

    def __init__(self, data_list,batch_size=1,shuffle=True,mode='train',num_workers=1):
        super().__init__()
        self.data_list = data_list
        self.total_len = len(self.data_list)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.mode = mode
        self.num_workers=num_workers

    def __getitem__(self, index):
        # 从列表中获取文件路径
        (data_orig_path, data_hazy_path) = self.data_list[index]
        
        with Image.open(data_orig_path) as data_orig_img:
            data_orig = data_orig_img.resize((480, 640), Image.Resampling.LANCZOS)
            # 归一化并转为 numpy
            data_orig = (np.array(data_orig) / 255.0)

        with Image.open(data_hazy_path) as data_hazy_img:
            data_hazy = data_hazy_img.resize((480, 640), Image.Resampling.LANCZOS)
            data_hazy = (np.array(data_hazy) / 255.0)
        
        # 转换为 Jittor Var
        data_orig = jt.array(data_orig).float32()
        data_hazy = jt.array(data_hazy).float32()
        
        # 转换: (H, W, C) -> (C, H, W)
        return data_orig.permute((2, 0, 1)), data_hazy.permute((2, 0, 1))

    def __len__(self):
        return self.total_len
    
if (__name__ == '__main__'):
    
    current_dir = os.path.dirname(os.path.abspath(__file__))

    orig_images_path = os.path.join(current_dir, 'data', 'original_images')  
    hazy_images_path = os.path.join(current_dir, 'data', 'hazy_images')  

    (train_list, val_list) = populate_train_list(orig_images_path, hazy_images_path, split_ratio=1.0)
    
    # 创建 Dataset 实例
    train_dataset = MyDataset(data_list=train_list)
    val_dataset = MyDataset(data_list=val_list) 
    
    # 测试 __getitem__
    (x, y) = train_dataset[0] 
    
    print("\nDataset test data:\n")
    print(f"Orig shape: {x.shape}, Hazy shape: {y.shape}") # (C, H, W)
    print(f"Data type: {x.dtype}")

    print("\n===============================\n")

    # 创建 DataLoader
    BATCH_SIZE = 32 

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE, 
        shuffle=True  
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE, 
        shuffle=False 
    )

    print(f"Train loader created with batch size {BATCH_SIZE}")
    print(f"Val loader created with batch size {BATCH_SIZE}")

    for i, (batch_orig, batch_hazy) in enumerate(train_loader):
        print(f"Batch {i}:")
        print(f"  Orig batch shape: {batch_orig.shape}") # (B, C, H, W)
        print(f"  Hazy batch shape: {batch_hazy.shape}")
        if i == 0: 
            break