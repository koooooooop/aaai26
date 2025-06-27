import numpy as np

def MSE(true, pre):
    """
    计算均方误差 (Mean Squared Error)
    """
    true = true.detach().cpu().numpy()
    pre = pre.detach().cpu().numpy()
    return np.mean((true - pre) ** 2)

def MAE(true, pre):
    """
    计算平均绝对误差 (Mean Absolute Error)
    """
    true = true.detach().cpu().numpy()
    pre = pre.detach().cpu().numpy()
    return np.mean(np.abs(true - pre))

def MAPE(true, pre):
    """
    计算平均绝对百分比误差 (Mean Absolute Percentage Error)
    """
    true = true.detach().cpu().numpy()
    pre = pre.detach().cpu().numpy()
    # 仅在 true != 0 的地方计算 MAPE，避免除以0
    mask = true != 0
    if np.any(mask):
        return np.mean(np.abs((true[mask] - pre[mask]) / true[mask])) * 100
    else:
        # 如果所有真实值都为0，则无法计算MAPE
        return float('nan')

class MetricsEvaluator:
    """
    一个简单的评估器类，用于在评估阶段累积和计算指标。
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.mses = []
        self.maes = []
        self.mapes = []

    def update(self, true, pre):
        self.mses.append(MSE(true, pre))
        self.maes.append(MAE(true, pre))
        self.mapes.append(MAPE(true, pre))

    def get_results(self):
        return {
            "MSE": np.mean(self.mses),
            "MAE": np.mean(self.maes),
            "MAPE": np.nanmean(self.mapes) # nanmean忽略NaN值
        } 