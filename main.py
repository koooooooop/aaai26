import yaml
import argparse

from data.dataset import TimeSeriesDataModule
from train import Trainer

def main():
    # 1. 解析命令行参数，获取配置文件路径
    parser = argparse.ArgumentParser(description="M²-MOEP Framework Training")
    parser.add_argument('--config', type=str, default='configs/base_experiment.yaml',
                        help='Path to the experiment configuration file.')
    args = parser.parse_args()
    
    # 2. 加载YAML配置
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        print("Configuration loaded successfully:")
        print(config)
    except FileNotFoundError:
        print(f"Error: Config file not found at {args.config}")
        return
    except Exception as e:
        print(f"Error loading config file: {e}")
        return

    # TODO: 设置随机种子，保证实验可复现
    # import torch
    # import numpy as np
    # torch.manual_seed(config.get('seed', 42))
    # np.random.seed(config.get('seed', 42))
            
    # 3. 初始化数据模块
    print("\nInitializing data module...")
    data_module = TimeSeriesDataModule(config)

    # 4. 初始化训练器
    print("\nInitializing trainer...")
    trainer = Trainer(config, data_module)
    
    # 5. 开始训练
    print("\nStarting training...")
    trainer.run()
    print("\nTraining finished.")

if __name__ == "__main__":
    main() 