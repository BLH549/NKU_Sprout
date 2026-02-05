# NKU Sprout 

## 项目概述

本项目是南开大学新芽计划的大作业，实现了两个经典图像复原模型在 Jittor 深度学习框架上的完整复现：

| 模型 | 任务 | 论文 | 
|-----|------|------|
| **AOD-Net** | 单图像去雾 | [ICCV 2017](https://openaccess.thecvf.com/content_ICCV_2017/papers/Li_AOD-Net_All-In-One_Dehazing_ICCV_2017_paper.pdf) | 
| **NAFNet** | 通用图像复原 | [ECCV 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136670017.pdf)|


## 快速开始

### 环境配置

```bash
pip install jittor pytorch opencv-python pillow numpy scipy scikit-image tqdm
```


### 克隆项目

```bash
git clone https://github.com/BLH549/NKU_Sprout.git
cd NKU_Sprout
```

## 项目结构

```
NKU_Sprout/
├── README.md                          # 本文件
├── requirements.txt                   # 依赖列表
└── dl_sprout/
    ├── AODNet_jittor/
    │   ├── jt_model.py               # AOD-Net Jittor 模型定义
    │   ├── jt_data.py                # 数据加载模块
    │   ├── jt_train.py               # 训练脚本
    │   ├── jt_inference1.py          # 推理脚本
    │   ├── model.py                  # PyTorch 原始模型 
    │   ├── saved_models/             # 预训练模型目录
    │   │   ├── MyDehazeNet_jt_epoch_9.pth
    │   │   └── dehaze_net_epoch_17_jt.pth
    │   ├── data/
    │   │   ├── original_images/      # 原始清晰图像
    │   │   └── hazy_images/          # 有雾图像
    │   └── test_images/              # 测试图像
    │
    └── NAFNet_jittor/
        ├── model.py                  # NAFNet Jittor 模型定义
        ├── dataset.py                # 数据加载模块
        ├── train.py                  # 训练脚本
        ├── inference.py              # 推理脚本
        ├── calc_psnr.py              # PSNR 计算工具
        ├── VisualComparison.py       # 可视化对比工具
        ├── torch_model.py            # PyTorch 原始模型 
        ├── saved_models/             # 预训练模型目录
        │   └── NAFNet-GoPro-width32.pth
        │   └── jt_NAFNet.pkl
        └── test_img/                 # 测试图像
```

##  模型推理

### AOD-Net 推理(包含可视化对比)

```bash
cd dl_sprout/AODNet_jittor
python jt_inference1.py 

```


### NAFNet 推理

```bash
cd dl_sprout/NAFNet_jittor
python inference.py
```

### NAFNet可视化对比
```bash
cd dl_sprout/NAFNet_jittor
python VisualComparison.py 
```



---

## 训练数据准备

### AOD-Net 数据集

链接：[作者项目主页](https://sites.google.com/site/boyilics/website-builder/project-page)

#### 数据集目录结构

```
data/
├── original_images/
│   ├── NYU2_1.jpg
│   ├── NYU2_2.jpg
│   └── ...
└── hazy_images/
    ├── NYU2_1_1_1.jpg
    ├── NYU2_1_1_2.jpg
    └── ...
```


### NAFNet 数据集

使用的是原论文中的GoPro数据集。(可以按照原论文中[GoPro.md](./GoPro.md) 的说明准备)

训练集 ：
[Google Drive](https://drive.google.com/file/d/1zgALzrLCC_tcXKu_iHQTHukKUVT1aodI/) |
[百度网盘](https://pan.baidu.com/s/1fdsn-M5JhxCL7oThEgt1Sw?pwd=9d26)

测试集:：
[Google Drive](https://drive.google.com/file/d/1abXSfeRGrzj2mQ2n2vIBHtObU6vXvr7C/) |
[百度网盘](https://pan.baidu.com/s/1oZtEtYB7-2p3fCIspky_mw?pwd=rmv9)

#### 数据集目录结构

```
datasets/GoPro/
├── train/
│   ├── input/        # 模糊图像
│   │   ├── GOPR0372_07_00-000047.png
│   │   └── ...
│   └── target/       # 清晰图像
│       ├── GOPR0372_07_00-000047.png
│       └── ...
└── test/
    ├── input.lmdb    # 测试输入 (LMDB 格式)
    └── target.lmdb   # 测试目标 (LMDB 格式)
```

---

##  模型训练

### AOD-Net 训练

```bash
cd dl_sprout/AODNet_jittor

python jt_train.py 
```


### NAFNet 训练

```bash
cd dl_sprout/NAFNet_jittor

python train.py 
```




