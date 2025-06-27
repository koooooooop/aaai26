import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None
    print("Warning: mamba-ssm is not installed. Falling back to LSTM placeholder.")

class FFTmsMambaExpert(nn.Module):
    """
    专家网络，实现多尺度Mamba (ms-Mamba) 架构。
    通过并行的、具有不同卷积核大小的Mamba模块来捕捉多尺度特征。
    """
    def __init__(self, config):
        super().__init__()
        if Mamba is None:
            raise ImportError("mamba-ssm is not installed. Please install it with 'pip install mamba-ssm'.")
            
        self.config = config
        self.seq_len = config['data']['seq_len']
        self.input_dim = 3  # cpu, memory, disk
        self.output_dim = self.input_dim * config['data']['pred_len']
        
        # --- Multi-Scale Mamba Implementation ---
        mamba_scales = config['model']['expert_params']['mamba_scales']
        self.mamba_d_model = config['model']['expert_params']['mamba_d_model']
        
        fft_feature_len = (self.seq_len // 2 + 1) * 2 * self.input_dim
        fused_input_dim = self.input_dim + fft_feature_len

        # 创建多个并行的Mamba骨干网络
        self.mamba_backbones = nn.ModuleList([
            Mamba(
                d_model=fused_input_dim,
                d_state=16,
                d_conv=d_conv_size, # <-- 使用不同的卷积核尺寸实现多尺度
                expand=2,
            ) for d_conv_size in mamba_scales
        ])

        # 输出投影层的输入维度是所有Mamba模块输出维度的总和
        concatenated_output_dim = fused_input_dim * len(mamba_scales)
        self.output_proj = nn.Linear(concatenated_output_dim, self.output_dim)

    def forward(self, x):
        """
        x: Original time series input, shape [Batch, seq_len, input_dim]
        """
        batch_size = x.shape[0]
        
        # 步骤 1: FFT 特征提取
        fft_result = torch.fft.rfft(x, dim=1, norm='ortho')
        fft_amp = fft_result.abs()
        fft_phase = fft_result.angle()
        
        fft_features = torch.cat([fft_amp, fft_phase], dim=2)
        fft_features_flat = fft_features.flatten(start_dim=1)

        # 步骤 2: 特征融合
        fft_features_expanded = fft_features_flat.unsqueeze(1).expand(-1, self.seq_len, -1)
        fused_input = torch.cat([x, fft_features_expanded], dim=2)

        # 步骤 3: 并行地通过多个Mamba骨干网络
        mamba_outputs = []
        for backbone in self.mamba_backbones:
            mamba_outputs.append(backbone(fused_input))
        
        # 步骤 4: 结果聚合
        # 沿特征维度拼接所有Mamba模块的输出
        concatenated_output = torch.cat(mamba_outputs, dim=2)
        
        # 步骤 5: 输出投影 (使用最后一个时间步的聚合输出来做预测)
        prediction = self.output_proj(concatenated_output[:, -1, :])
        
        # Reshape to [Batch, pred_len, D_out]
        prediction = prediction.view(batch_size, self.config['data']['pred_len'], self.input_dim)
        
        return prediction.squeeze(1) 