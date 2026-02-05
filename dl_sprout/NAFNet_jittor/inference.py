import jittor as jt
import cv2
import numpy as np
import os
from model import NAFNet 

jt.flags.use_cuda = 1 

def inference():
    print("正在初始化 NAFNet 模型...")
    net = NAFNet(img_channel=3, width=32, middle_blk_num=1,
                 enc_blk_nums=[1, 1, 1, 28], dec_blk_nums=[1, 1, 1, 1])
    
    # 加载转换好的权重
    Current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(Current_dir, 'saved_models', 'jt_NAFNet.pkl')
    net.load(model_path)
    net.eval()
    print("模型加载完成。")

    # 读取图片
    img_path = os.path.join(Current_dir, 'test_img', 'test.png') 
    if not os.path.exists(img_path):
        print(f"错误：找不到图片 {img_path}")
        return

    img_bgr = cv2.imread(img_path)
    # 转换为 RGB 并归一化
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    
    # 转换为 Jittor 张量 
    input_tensor = jt.array(img_rgb).transpose(2, 0, 1).unsqueeze(0)

    # 推理
    print("正在进行去模糊处理...")
    with jt.no_grad():
        output = net(input_tensor)
        output.sync() 

    # 保存
    output_img = output.squeeze(0).transpose(1, 2, 0).numpy()
    output_img = np.clip(output_img, 0, 1) * 255.0
    output_img = output_img.astype(np.uint8)
    
    save_path = os.path.join(Current_dir, 'test_img', 'jt_result.png')
    cv2.imwrite(save_path, cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))
    print(f"处理成功！结果已保存至: {save_path}")

if __name__ == "__main__":
    inference()