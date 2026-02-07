import jittor as jt
import torch
import numpy as np
from model import NAFNet 

def convert_nafnet_weights(torch_pth_path, jittor_save_path):
    print("正在读取 PyTorch 权重文件...")
    torch_dict = torch.load(torch_pth_path, map_location='cpu')
    if 'params' in torch_dict:
        state_dict = torch_dict['params']
    elif 'state_dict' in torch_dict:
        state_dict = torch_dict['state_dict']
    else:
        state_dict = torch_dict
    jt_model = NAFNet(img_channel=3, width=32, middle_blk_num=1,
                      enc_blk_nums=[1, 1, 1, 28], dec_blk_nums=[1, 1, 1, 1])
    
    new_jt_dict = {}
    
    print("开始逐层转换权重...")
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        if isinstance(v, torch.Tensor):
            v_numpy = v.detach().cpu().numpy()
            new_jt_dict[name] = jt.array(v_numpy)
        else:
            new_jt_dict[name] = v
    try:
        jt_model.load_state_dict(new_jt_dict)
        jt.save(jt_model.state_dict(), jittor_save_path)
        print(f"恭喜！权重已成功转换并保存至: {jittor_save_path}")
    except Exception as e:
        print(f"加载失败，请检查模型结构是否匹配。错误信息: {e}")

if __name__ == "__main__":
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__)) 
    TORCH_MODEL_PATH = os.path.join(current_dir, 'saved_models', 'NAFNet-GoPro-width32.pth') 
    JITTOR_MODEL_PATH = os.path.join(current_dir, 'saved_models', 'jt_NAFNet.pkl')
    convert_nafnet_weights(TORCH_MODEL_PATH,JITTOR_MODEL_PATH)