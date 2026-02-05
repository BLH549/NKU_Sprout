import jittor as jt
from jittor import nn
from jittor import init
from jittor.dataset import DataLoader
import os
from jt_model import jt_AODNet
from jt_data import MyDataset, populate_train_list

jt.flags.use_cuda = 1
BATCH_INTERVAL = 100 

def weights_init(m):
    """
    自定义权重初始化
    """
    classname = m.__class__.__name__
    if (classname.find('Conv') != (- 1)):
        init.gauss_(m.weight, mean=0.0, std=0.02)
    elif (classname.find('BatchNorm') != (- 1)):
        init.gauss_(m.weight, mean=1.0, std=0.02)
        m.bias.data.fill_(0)


def train(orig_images_path, hazy_images_path, batch_size, epochs):

    (train_list, val_list) = populate_train_list(orig_images_path, hazy_images_path, split_ratio=0.9)
    train_dataset = MyDataset(data_list=train_list)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4) # 可选: 增加 num_workers 加速
    total_batches = len(train_loader)//batch_size

    print(f"Total training images: {len(train_list)}")
    print(f"Total validation images: {len(val_list)}")
    print(f"Training loader batches: {len(train_loader)}")
    
    dehaze_net = jt_AODNet() 
    dehaze_net.apply(weights_init)
    
    criterion = nn.MSELoss()
    optimizer = jt.optim.Adam(dehaze_net.parameters(), lr=0.001, weight_decay=0.0001)
    dehaze_net.train()
    
    # 定义模型保存目录
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        current_dir = os.getcwd() 

    save_dir = os.path.join(current_dir,'saved_models')


    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True) 
    
    for epoch in range(epochs):
        epoch_loss = 0
        
        for (iteration, (img_orig, img_haze)) in enumerate(train_loader):
            
            clean_image = dehaze_net(img_haze)
            
            loss = criterion(clean_image, img_orig)

            optimizer.zero_grad()
            optimizer.backward(loss)
            optimizer.step()
            
            epoch_loss += loss.item() 

            if (iteration + 1) % BATCH_INTERVAL == 0:
                progress = (iteration + 1) / total_batches * 100
                print(f'EPOCH {epoch:04d} | Process {progress:.1f}% | Current Loss: {loss.item():.6f}')


        avg_loss = epoch_loss / total_batches
        
        print(f'EPOCH: {epoch:04d}  LOSS: {avg_loss:.4f}')
        
        # 保存模型 
        save_path = os.path.join(save_dir, f'MyDehazeNet_jt_epoch_{epoch}.pth')
        jt.save(dehaze_net.state_dict(), save_path)
        print(f"Model saved to {save_path}")

if (__name__ == '__main__'):
    
    current_dir = os.path.dirname(os.path.abspath(__file__))


    orig_images_path = os.path.join(current_dir, 'data', 'original_images')
    hazy_images_path = os.path.join(current_dir, 'data', 'hazy_images')
    
    # 检查路径是否存在 
    if not os.path.exists(hazy_images_path):
        print(f"Error: Hazy images path not found at {hazy_images_path}")
        exit() 
        
    batch_size = 32
    epochs = 10
    
    print("Starting training...")
    train(orig_images_path, hazy_images_path, batch_size, epochs)
    print("Training finished.")