import jittor as jt
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os  
from jt_model import jt_AODNet

def dehaze_image(image_name, model_path):
    data_hazy = Image.open(image_name)
    data_hazy = (np.array(data_hazy) / 255.0)
    
    # 转换为 Jittor 变量
    data_hazy = jt.array(data_hazy).float()
    data_hazy = data_hazy.permute((2, 0, 1))
    data_hazy = data_hazy.unsqueeze(0)

    state_dict = jt.load(model_path)
    dehaze_net = jt_AODNet()
    dehaze_net.load_state_dict(state_dict)
    
    dehaze_net.eval()
    
    # 推理与可视化
    clean_image = dehaze_net(data_hazy).detach().numpy().squeeze()
    clean_image = np.swapaxes(clean_image, 0, 1)
    clean_image = np.swapaxes(clean_image, 1, 2)
    
    return clean_image

def show_images(images_data):
    """
    显示n行3列的图片
    images_data: 列表，每个元素为 (original_img, torch_img, jt_img)
    """
    # 设置字体大小
    plt.rcParams.update({'font.size': 16})
    
    n = len(images_data)
    fig, axes = plt.subplots(n, 3, figsize=(15, 5*n))
    
    if n == 1:
        axes = axes.reshape(1, -1)
    
    titles = ['Original Image', 'Dehazed Image By PyTorch', 'Dehazed Image By Jittor']
    
    for i, (original_img, torch_img, jt_img) in enumerate(images_data):
        # 第一列：原图
        axes[i, 0].imshow(original_img)
        axes[i, 0].axis('off')
        if i == 0:  
            axes[i, 0].set_title(titles[0])
        
        # 第二列
        axes[i, 1].imshow(torch_img)
        axes[i, 1].axis('off')
        if i == 0:  
            axes[i, 1].set_title(titles[1])
        
        # 第三列
        axes[i, 2].imshow(jt_img)
        axes[i, 2].axis('off')
        if i == 0: 
            axes[i, 2].set_title(titles[2])
    
    plt.tight_layout()
    plt.show()

def get_image_paths(n, test_images_dir):
    """获取前n张测试图片的路径"""
    image_paths = []
    for i in range(n):
        img_path = os.path.join(test_images_dir, f'test{i}.png')
        if os.path.exists(img_path):
            image_paths.append(img_path)
        else:
            print(f"警告：图片不存在：{img_path}")
    return image_paths

if __name__ == '__main__':
    current_dir = os.path.dirname(__file__)
    
    test_images_dir = os.path.join(current_dir, 'test_images')
    torch_model_path = os.path.join(current_dir, 'saved_models', 'dehaze_net_epoch_17_jt.pth')
    jt_model_path = os.path.join(current_dir, 'saved_models', 'AODNet_jt_best.pth')   
    
    # 设置要显示的图片数量n
    n = 2  
    
    if not os.path.exists(torch_model_path):
        print(f"模型不存在：{torch_model_path}")
        exit()
    if not os.path.exists(jt_model_path):
        print(f"模型不存在：{jt_model_path}")
        exit()

    image_paths = get_image_paths(n, test_images_dir)
    if not image_paths:
        print("没有找到可用的测试图片")
        exit()
    
    # 处理所有图片
    images_data = []
    for img_path in image_paths:
        print(f"处理图片：{img_path}")
        torch_img = dehaze_image(img_path, torch_model_path)
        jt_img = dehaze_image(img_path, jt_model_path)
        original_img = np.array(Image.open(img_path))
        
        images_data.append((original_img, torch_img, jt_img))
    
    show_images(images_data)