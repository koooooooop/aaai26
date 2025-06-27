import torch
import torch.nn as nn
import torch.nn.functional as F

def calculate_triplet_loss(embeddings, labels, margin=0.5):
    """
    使用Batch Hard策略计算Triplet Loss。
    'Batch Hard'策略寻找每个锚点最难的正样本和最难的负样本。
    :param embeddings: 批次中所有样本的嵌入向量, 维度 [B, embedding_dim]
    :param labels: 批次中所有样本的"模式"伪标签, 维度 [B]
    :param margin: 三元组损失的边界 alpha
    :return: 该批次的平均三元组损失
    """
    # 1. 计算批次内所有嵌入向量对的平方欧氏距离
    pairwise_dist = torch.cdist(embeddings, embeddings, p=2).pow(2)

    # 2. 挖掘最难的正样本和负样本
    batch_size = embeddings.size(0)
    # mask_positive[i, j] = 1 if labels[i] == labels[j]
    mask_positive = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
    mask_negative = 1.0 - mask_positive

    # Hardest Positive: 对于每个锚点, 找到距离最远的正样本
    hardest_positive_dist = (pairwise_dist * mask_positive).max(dim=1)[0]

    # Hardest Negative: 对于每个锚点, 找到距离最近的负样本
    # 通过给正样本对的位置加上一个极大值来屏蔽它们
    max_dist = pairwise_dist.max().item() + 1.0
    negative_dists = pairwise_dist + max_dist * mask_positive
    hardest_negative_dist = negative_dists.min(dim=1)[0]
    
    # 3. 计算三元组损失
    loss = F.relu(hardest_positive_dist - hardest_negative_dist + margin)
    
    # 如果一个样本没有正样本或负样本，其损失为0，不应计入平均值
    num_non_zero_triplets = torch.count_nonzero(loss)
    if num_non_zero_triplets == 0:
        return torch.tensor(0.0, device=embeddings.device)
        
    return loss.sum() / num_non_zero_triplets


class CompositeLoss(nn.Module):
    """
    复合损失函数，使用不确定性加权自动平衡预测损失和路由损失。
    """
    def __init__(self, config):
        super().__init__()
        self.triplet_margin = config['training']['triplet_margin']
        # 将log(sigma^2)作为可学习参数, 0: 路由损失(cl), 1: 预测损失(pr)
        self.log_vars = nn.Parameter(torch.zeros(2))

    def forward(self, predictions, embeddings, ground_truth, pseudo_labels):
        """
        :param predictions: 模型的最终预测
        :param embeddings: 门控网络输出的嵌入向量
        :param ground_truth: 真实的未来序列Y
        :param pseudo_labels: 用于三元组挖掘的伪标签
        """
        
        # 1. 计算各个损失
        loss_pr = F.mse_loss(predictions, ground_truth.squeeze(1))
        loss_cl = calculate_triplet_loss(embeddings, pseudo_labels, self.triplet_margin)

        # 2. 应用不确定性加权
        # L_total = (1/(2*σ_cl^2)) * L_cl + 0.5*log(σ_cl^2) + (1/(2*σ_pr^2)) * L_pr + 0.5*log(σ_pr^2)
        precision_cl = torch.exp(-self.log_vars[0])
        precision_pr = torch.exp(-self.log_vars[1])
        
        total_loss = (0.5 * precision_cl * loss_cl + 0.5 * self.log_vars[0]) + \
                     (0.5 * precision_pr * loss_pr + 0.5 * self.log_vars[1])
        
        # 返回总损失和包含各个子损失的字典，用于日志记录
        loss_dict = {
            "total_loss": total_loss.item(),
            "loss_pr": loss_pr.item(),
            "loss_cl": loss_cl.item(),
            "sigma_cl": self.log_vars[0].item(),
            "sigma_pr": self.log_vars[1].item()
        }
        
        return total_loss, loss_dict 