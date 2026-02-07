import os
import re
import matplotlib.pyplot as plt

def parse_log(filename):
    """解析日志文件，提取 Batch Loss 和 Epoch PSNR"""
    batch_losses = []
    val_psnrs = []
    
    # 匹配 Batch Loss: | Loss: 0.164232
    loss_re = re.compile(r"Loss:\s+([\d.]+)")
    # 匹配 Val PSNR: | Val PSNR: 18.37
    psnr_re = re.compile(r"Val PSNR:\s+([\d.]+)")
    
    if not os.path.exists(filename):
        print(f"警告: 未找到文件 {filename}")
        return [], []

    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            loss_match = loss_re.search(line)
            if loss_match:
                batch_losses.append(float(loss_match.group(1)))
            
            psnr_match = psnr_re.search(line)
            if psnr_match:
                val_psnrs.append(float(psnr_match.group(1)))
                
    return batch_losses, val_psnrs

def plot_comparison(jt_file, torch_file):
    # 解析数据
    jt_loss, jt_psnr = parse_log(jt_file)
    torch_loss, torch_psnr = parse_log(torch_file)

    # 创建画布 (1行2列)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # --- 左图：Loss 对比 ---
    ax1.plot(jt_loss, label='Jittor', color='#1f77b4', alpha=0.8)
    ax1.plot(torch_loss, label='PyTorch', color='#ff7f0e', alpha=0.8)
    ax1.set_title('Training Batch Loss Comparison', fontsize=12)
    ax1.set_xlabel('Steps (per 100 batches)', fontsize=10)
    ax1.set_ylabel('Loss', fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend()

    # --- 右图：PSNR 对比 ---
    epochs_jt = range(len(jt_psnr))
    epochs_torch = range(len(torch_psnr))
    ax2.plot(epochs_jt, jt_psnr, label='Jittor ', color='#1f77b4', marker='o', linewidth=2)
    ax2.plot(epochs_torch, torch_psnr, label='PyTorch', color='#ff7f0e', marker='s', linewidth=2)
    ax2.set_title('Validation PSNR Comparison', fontsize=12)
    ax2.set_xlabel('Epochs', fontsize=10)
    ax2.set_ylabel('PSNR (dB)', fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    
    # 保存结果
    output_name = 'framework_comparison.png'
    plt.savefig(output_name, dpi=300)
    print(f"对比图生成成功：{os.path.abspath(output_name)}")
    plt.show()

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))

    jt_log = os.path.join(current_dir, 'jt2.log')
    torch_log =os.path.join(current_dir, 'torch.log')
    plot_comparison(jt_log, torch_log)