# M²-MOEP: 一种基于Mamba与度量学习的混合专家预测框架

本项目实现了`M²-MOEP`框架，这是一个基于 "M²-MOEP: 一种基于Mamba与度量学习的混合专家模型用于端到端时变工作负载预测" 方案的新型时间序列预测架构。

该框架的核心创新点包括：
- 一个由FFT增强的、用于鲁棒特征提取的多尺度Mamba (`ms-Mamba`) 专家网络。
- 一个完全端到端可训练的混合专家 (MoE) 结构。
- 一个通过深度度量学习（Triplet Loss）训练的、能够学习任务特定相似性度量的门控机制。
- 基于专家性能的动态伪标签生成机制，用于指导门控网络的学习。
- 使用同方差不确定性加权方法，实现损失的自动平衡。

## 项目结构

```
M2-MOEP/
├── configs/              # 实验配置文件 (.yaml)
├── data/                 # 数据加载与处理模块
├── models/               # 模型定义 (M2-MOEP, Expert, Gating, Flow)
├── utils/                # 工具模块 (损失函数, 评估指标)
├── main.py               # 启动训练的主入口文件
├── pretrain_flow.py      # 用于预训练Normalizing Flow模型的脚本
└── README.md             # 本说明文档
```

## 安装与设置

1.  **克隆仓库** (如果适用)。

2.  **创建并激活Python环境** (例如, 使用conda):
    ```bash
    conda create -n m2-moep python=3.9
    conda activate m2-moep
    ```

3.  **安装所需依赖**:
    本项目需要PyTorch及其他几个库。`mamba-ssm`是核心依赖之一。

    **第一步：安装PyTorch**
    根据您的环境 (CUDA 12.4)，推荐使用以下命令安装与CUDA 12.1兼容的PyTorch版本（通常向前兼容）。
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```

    **第二步：安装其他核心依赖**
    ```bash
    pip install pyyaml scikit-learn pandas tqdm "mamba-ssm>=1.2.0" causal-conv1d>=1.1.0
    ```

## 如何运行实验

运行一个完整的实验需要两步。所有命令都应在项目的根目录 (`new/`) 下执行。

### 步骤一: 预训练Normalizing Flow模型

门控机制依赖于由预训练的Normalizing Flow模型生成的潜在空间表征。因此，您必须首先执行此预训练步骤。

```bash
python pretrain_flow.py
```

此命令将:
- 加载默认配置文件 (`configs/base_experiment.yaml`) 中指定的训练数据。
- 训练`NormalizingFlow`模型。
- 将训练好的模型权重保存到根目录下的 `flow_model.pth` 文件中。

### 步骤二: 运行M²-MOEP主模型训练

当 `flow_model.pth` 文件生成后，您就可以开始`M²-MOEP`框架的端到端主训练流程。

```bash
python main.py
```

此命令将:
- 加载默认配置。
- 初始化`TimeSeriesDataModule`，它会自动加载预训练好的`flow_model.pth`来处理数据。
- 使用完整的`M²-MOEP`模型初始化`Trainer`。
- 开始训练循环，包括前向/反向传播、损失计算以及周期性地更新用于Triplet Loss的伪标签。

## 配置与消融实验

所有的实验参数都通过 `configs/` 目录下的 `.yaml` 文件进行控制。您可以通过传递 `--config` 参数来运行特定的实验。

例如，要进行一项消融研究，您可以创建一个新的配置文件 `configs/ablation_study.yaml`，在其中修改参数，然后使用以下命令运行：

```bash
python main.py --config configs/ablation_study.yaml
```
这种模块化的方法是本项目设计的核心，它为进行严谨的科学研究提供了便利。 