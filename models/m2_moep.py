import torch
import torch.nn as nn
import torch.nn.functional as F

from .expert import FFTmsMambaExpert
from .gating import GatingEncoder
# from .flow import NormalizingFlow # Flow model is used externally for data pre-processing

class M2_MOEP(nn.Module):
    """
    M²-MOEP 框架的主模型。
    集成了度量学习门控和多个专家网络。
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        num_experts = config['model']['num_experts']
        embedding_dim = config['model']['embedding_dim']
        
        # 注意: Flow模型被假设在数据预处理阶段使用，这里不再作为模型的一部分
        # 以简化端到端训练的复杂度。输入z是预处理好的。
        
        # 1. 门控网络 (孪生编码器)
        self.gating_encoder = GatingEncoder(config)
        
        # 2. 专家网络列表 (FFT+ms-Mamba)
        self.experts = nn.ModuleList([FFTmsMambaExpert(config) for _ in range(num_experts)])
        
        # 3. 可学习的专家原型 (K个)
        self.expert_prototypes = nn.Parameter(torch.randn(num_experts, embedding_dim))
        
        # 4. Softmax温度系数
        self.temperature = config['model']['temperature']

    def forward(self, x_expert, z_flow):
        """
        前向传播
        :param x_expert: 用于专家网络的原始序列输入, 维度 [Batch, seq_len, input_dim]
        :param z_flow: 用于门控的潜在序列输入, 维度 [Batch, latent_seq_dim]
        :return: final_prediction, routing_weights, embedding
        """
        
        # === 步骤1: 度量学习门控 ===
        embedding = self.gating_encoder(z_flow)

        # === 步骤2: 专家路由 ===
        # 计算嵌入与专家原型的距离 (欧氏距离的平方)
        dists = torch.cdist(embedding, self.expert_prototypes, p=2).pow(2)
        
        # 将距离转换为路由权重
        routing_weights = F.softmax(-dists / self.temperature, dim=1)

        # === 步骤3: 专家预测与结果聚合 ===
        expert_outputs = [expert(x_expert) for expert in self.experts]
        expert_outputs_stacked = torch.stack(expert_outputs, dim=2)
        
        # 使用 einsum 实现高效的加权求和, b: batch, d: out_dim, e: num_experts
        final_prediction = torch.einsum('be,bde->bd', routing_weights, expert_outputs_stacked)
        
        return final_prediction, routing_weights, embedding 