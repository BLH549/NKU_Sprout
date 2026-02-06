import jittor as jt
import torch
import numpy as np
from model import NAFNet 
def convert_nafnet_weights(torch_pth_path, jittor_save_path):
    print("正在读取 PyTorch 权重文件...")
    torch_dict = torch.load(torch_pth_path, map_location='cpu')
    
    # 2. 处理 BasicSR 格式：官方权重通常在 'params' 或 'params_ema' 键下
    if 'params' in torch_dict:
        state_dict = torch_dict['params']
    elif 'state_dict' in torch_dict:
        state_dict = torch_dict['state_dict']
    else:
        state_dict = torch_dict

    # 3. 初始化你的 Jittor 模型 (严格按照你的 yml 配置)
    # width=32, enc_blk_nums=[1, 1, 1, 28], middle_blk_num=1, dec_blk_nums=[1, 1, 1, 1]
    jt_model = NAFNet(img_channel=3, width=32, middle_blk_num=1,
                      enc_blk_nums=[1, 1, 1, 28], dec_blk_nums=[1, 1, 1, 1])
    
    new_jt_dict = {}
    
    print("开始逐层转换权重...")
    for k, v in state_dict.items():
        # 处理可能存在的 module. 前缀
        name = k.replace('module.', '')
        
        # 核心转换逻辑：断开计算图 -> CPU -> NumPy -> Jittor Array
        if isinstance(v, torch.Tensor):
            v_numpy = v.detach().cpu().numpy()
            new_jt_dict[name] = jt.array(v_numpy)
        else:
            new_jt_dict[name] = v
            
    # 4. 加载到 Jittor 模型
    try:
        jt_model.load_state_dict(new_jt_dict)
        # 5. 保存为 Jittor 格式 (.pkl 或 .pth 均可，Jittor 习惯用 .pkl)
        jt.save(jt_model.state_dict(), jittor_save_path)
        print(f"恭喜！权重已成功转换并保存至: {jittor_save_path}")
    except Exception as e:
        print(f"加载失败，请检查模型结构是否匹配。错误信息: {e}")

if __name__ == "__main__":
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__)) 
    TORCH_MODEL_PATH = os.path.join(current_dir, 'saved_models', 'NAFNet-GoPro-width32.pth') 
    JITTOR_MODEL_PATH = os.path.join(current_dir, 'saved_models', 'jt.pth')
    # 使用你 yml 中指定的路径
    convert_nafnet_weights(TORCH_MODEL_PATH,JITTOR_MODEL_PATH)