import jittor as jt
from jittor import nn, init
from jittor.dataset import DataLoader
from jittor.lr_scheduler import ReduceLROnPlateau  # 注意：必须从 lr_scheduler 导入
import os
import math
from jt_model import jt_AODNet
from jt_data import MyDataset, populate_train_list

jt.flags.use_cuda = 1
BATCH_INTERVAL = 100 

def weights_init(m):
    """
    自定义权重初始化
    """
    classname = m.__class__.__name__
    if (classname.find('Conv') != -1):
        init.gauss_(m.weight, mean=0.0, std=0.02)
    elif (classname.find('BatchNorm') != -1):
        init.gauss_(m.weight, mean=1.0, std=0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)

def compute_psnr(mse):
    """根据 MSE 计算 PSNR"""
    if mse == 0: return 100
    return 10 * math.log10(1.0 / mse)

def train(orig_images_path, hazy_images_path, batch_size, epochs):
    (train_list, val_list) = populate_train_list(orig_images_path, hazy_images_path, split_ratio=0.9)
    train_loader = DataLoader(MyDataset(train_list), batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(MyDataset(val_list), batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True)

    total_train_images = len(train_loader)
    total_train_batches = math.ceil(total_train_images / batch_size)
    
    total_val_images = len(val_loader)
    total_val_batches = total_val_images // batch_size

    print(f"Batches per Epoch: {total_train_batches} | Batch Size: {batch_size}")
    print(f"Validation batches per Epoch: {total_val_batches} | Batch Size: {batch_size}")

    dehaze_net = jt_AODNet() 
    dehaze_net.apply(weights_init)
    
    criterion = nn.MSELoss()
    optimizer = jt.optim.Adam(dehaze_net.parameters(), lr=0.001, weight_decay=0.0001)
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(current_dir, 'saved_models')
    os.makedirs(save_dir, exist_ok=True) 
    
    best_psnr = 0.0
    for epoch in range(epochs):
        dehaze_net.train()
        epoch_loss = 0
        for iteration, (img_orig, img_haze) in enumerate(train_loader):
            clean_image = dehaze_net(img_haze)
            loss = criterion(clean_image, img_orig)
            
            optimizer.step(loss) 
            epoch_loss += loss.item() 

            if (iteration + 1) % BATCH_INTERVAL == 0:
                print(f'EPOCH {epoch:04d} | Batch {iteration+1}/{total_train_batches} | Loss: {loss.item():.6f}')

        avg_train_loss = epoch_loss / total_train_batches
        
        dehaze_net.eval()
        val_mse_sum = 0
        with jt.no_grad():
            for img_orig, img_haze in val_loader:
                output = dehaze_net(img_haze)
                val_mse_sum += criterion(output, img_orig).item()
        
        avg_val_mse = val_mse_sum / total_val_batches
        current_psnr = compute_psnr(avg_val_mse)
        
        scheduler.step(avg_val_mse)
        
        current_lr = optimizer.lr
        print(f'===> EPOCH: {epoch:04d} | Train Loss: {avg_train_loss:.4f} | Val PSNR: {current_psnr:.2f} | LR: {current_lr:.6f}')
        
        jt.save(dehaze_net.state_dict(), os.path.join(save_dir, 'AODNet_jt_latest.pth'))
        if current_psnr > best_psnr:
            best_psnr = current_psnr
            best_path = os.path.join(save_dir, 'AODNet_jt_best.pth')
            jt.save(dehaze_net.state_dict(), best_path)
            print(f"[*] Best Model Updated! PSNR: {best_psnr:.2f}")

if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
    orig_path = "/home/liu/projects/dl-sprout/AODNet-jittor/data/original_images"
    hazy_path = "/home/liu/projects/dl-sprout/AODNet-jittor/data/hazy_images"
    
    if os.path.exists(hazy_path):
        train(orig_path, hazy_path, batch_size=32, epochs=20)
    else:
        print("Data path not found.")