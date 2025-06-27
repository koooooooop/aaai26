import torch.nn as nn

class GatingEncoder(nn.Module):
    """
    孪生编码器，用于将潜在序列z映射到嵌入空间。
    """
    def __init__(self, config):
        super().__init__()
        # 从config中读取输入维度和输出维度
        latent_seq_dim = config['model']['gating_params']['latent_seq_dim']
        embedding_dim = config['model']['embedding_dim']
        
        # 定义一个简单的多层感知机(MLP)作为编码器
        # Flatten()将输入的序列展平为一个长向量
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(latent_seq_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3), # Dropout用于防止过拟合
            nn.Linear(1024, embedding_dim)
        )
    
    def forward(self, z):
        # z的输入维度: [Batch, Sequence, Features]
        # 经过net后输出: [Batch, embedding_dim]
        return self.net(z) 