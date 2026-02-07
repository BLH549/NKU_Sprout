import jittor as jt
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os  
from jt_model import jt_AODNet

# 环境配置
jt.flags.use_cuda = 1

def dehaze_image(image_path, model_path):
    """单张图片去雾推理"""
    # 1. 图像预处理
    original_img = Image.open(image_path).convert('RGB')
    data_hazy = np.array(original_img).astype('float32') / 255.0
    
    # 转换为 Jittor 变量并调整维度 (H,W,C) -> (1,C,H,W)
    data_hazy = jt.array(data_hazy).permute(2, 0, 1).unsqueeze(0)

    # 2. 加载模型
    dehaze_net = jt_AODNet()
    state_dict = jt.load(model_path)
    dehaze_net.load_state_dict(state_dict)
    dehaze_net.eval()
    
    # 3. 执行推理
    with jt.no_grad():
        clean_image = dehaze_net(data_hazy)
    
    clean_image = clean_image.squeeze().clamp(0.0, 1.0).numpy()
    clean_image = clean_image.transpose(1, 2, 0)
    
    # 转换为 0-255 uint8 格式便于保存
    clean_image_uint8 = (clean_image * 255).astype(np.uint8)
    
    return np.array(original_img), clean_image_uint8

def run_test(test_dir, model_path, output_dir, num_images=2):
    """批量运行测试并保存结果"""
    os.makedirs(output_dir, exist_ok=True)
    results = []

    # 获取测试图片
    image_files = [f for f in os.listdir(test_dir) if f.endswith(('.png', '.jpg', '.jpeg'))][:num_images]
    
    for filename in image_files:
        img_path = os.path.join(test_dir, filename)
        print(f"正在去雾: {filename}...")
        
        orig, dehazed = dehaze_image(img_path, model_path)
        
        # 保存去雾后的单张图
        save_path = os.path.join(output_dir, f"dehazed_{filename}")
        Image.fromarray(dehazed).save(save_path)
        
        results.append((orig, dehazed, filename))

    # 可视化展示
    show_results(results)

def show_results(results):
    n = len(results)
    if n == 0: return
    
    fig, axes = plt.subplots(n, 2, figsize=(10, 5 * n))
    if n == 1: axes = np.expand_dims(axes, axis=0)
    
    for i, (orig, dehazed, name) in enumerate(results):
        axes[i, 0].imshow(orig)
        axes[i, 0].set_title(f"Original: {name}")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(dehazed)
        axes[i, 1].set_title(f"Jittor Dehazed")
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # 获取当前目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 路径配置
    test_images_dir = os.path.join(current_dir, 'test_images')
    model_path = os.path.join(current_dir, 'saved_models', 'AODNet_jt_best.pth')
    output_dir = os.path.join(current_dir, 'output_results')

    if not os.path.exists(model_path):
        print(f"错误：找不到权重文件 {model_path}")
    elif not os.path.exists(test_images_dir):
        print(f"错误：找不到测试图片目录 {test_images_dir}")
    else:
        run_test(test_images_dir, model_path, output_dir, num_images=5)
        print(f"\n处理完成！去雾后的图片已保存至: {output_dir}")