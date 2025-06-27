import torch
import torch.optim as optim
import yaml
from tqdm import tqdm
import numpy as np
import os

from data.dataset import TimeSeriesDataModule
from models.flow import NormalizingFlow

def pretrain_flow(config):
    """
    独立训练Normalizing Flow模型并保存。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 加载数据
    print("Initializing data module for pre-training...")
    data_module = TimeSeriesDataModule(config)
    train_loader = data_module.get_train_loader()

    # 2. 准备用于Flow模型训练的数据
    # Flow模型处理向量，因此我们将每个时间序列窗口展平
    all_segments = []
    for x, _ in train_loader:
        # x shape: [Batch, seq_len, input_dim]
        all_segments.append(x.flatten(start_dim=1))
    
    if not all_segments:
        print("No data found for pre-training. Aborting.")
        return

    x_train_tensor = torch.cat(all_segments, dim=0).to(device)
    print(f"Prepared {x_train_tensor.shape[0]} samples for Flow pre-training.")

    # 3. 初始化Flow模型和优化器
    seq_len = config['data']['seq_len']
    input_dim = 3 # cpu, memory, disk
    flow_input_size = seq_len * input_dim
    
    flow_model = NormalizingFlow(input_size=flow_input_size, flow_layers=4).to(device)
    optimizer = optim.Adam(flow_model.parameters(), lr=1e-3)

    # 4. 训练循环
    print("Starting Flow model pre-training...")
    num_epochs = config.get('pretraining', {}).get('flow_epochs', 500) # 在config中可配置
    
    for epoch in range(num_epochs):
        flow_model.train()
        optimizer.zero_grad()
        
        # 损失是负对数似然
        loss = -torch.mean(flow_model.log_prob(x_train_tensor))
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            print(f"[Epoch {epoch+1}/{num_epochs}] Flow Pre-training Loss: {loss.item():.4f}")

    # 5. 保存模型
    save_path = "flow_model.pth"
    torch.save(flow_model.state_dict(), save_path)
    print(f"Flow model pre-trained and saved to {save_path}")

if __name__ == '__main__':
    config_path = 'configs/base_experiment.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    pretrain_flow(config) 