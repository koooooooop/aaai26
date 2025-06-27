import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.preprocessing import MinMaxScaler
import warnings

from models.flow import NormalizingFlow # 导入Flow模型

class FullDataset(Dataset):
    """
    一个完整的数据集类，存储 X, Y, Z 和伪标签。
    """
    def __init__(self, x_data, y_data, z_data, labels):
        self.x_data = torch.tensor(x_data, dtype=torch.float32)
        self.y_data = torch.tensor(y_data, dtype=torch.float32)
        self.z_data = torch.tensor(z_data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx], self.z_data[idx], self.labels[idx]

    def update_labels(self, new_labels):
        """用于更新伪标签的接口"""
        self.labels = torch.tensor(new_labels, dtype=torch.long)


class TimeSeriesDataModule:
    """
    数据模块，封装了所有数据加载和预处理的逻辑，包括Flow模型集成和伪标签管理。
    """
    def __init__(self, config):
        self.config = config
        self.data_dir = config['data']['data_dir']
        self.seq_len = config['data']['seq_len']
        self.pred_len = config['data']['pred_len']
        self.batch_size = config['training']['batch_size']
        self.num_experts = config['model']['num_experts']
        
        self.train_dir = os.path.join(self.data_dir, 'train')
        self.test_dir = os.path.join(self.data_dir, 'test')
        self.val_dir = os.path.join(self.data_dir, 'test') 
        
        self.scaler = None
        self.flow_model = None
        self.train_dataset = None
        self.val_dataset = None
        
        self.setup()

    def setup(self):
        """执行数据加载和预处理的核心步骤。"""
        self.scaler = self._fit_scaler()
        self.flow_model = self._load_flow_model()
        
        print("Loading and processing training data...")
        self.train_dataset = self._load_split('train')
        print("Loading and processing validation data...")
        self.val_dataset = self._load_split('val')

    def _load_flow_model(self):
        print("Loading pre-trained Flow model...")
        flow_model_path = "flow_model.pth"
        if not os.path.exists(flow_model_path):
            raise FileNotFoundError(f"Flow model not found at {flow_model_path}. Please run pretrain_flow.py first.")
        
        input_dim = 3
        flow_input_size = self.seq_len * input_dim
        model = NormalizingFlow(input_size=flow_input_size, flow_layers=4)
        model.load_state_dict(torch.load(flow_model_path))
        model.eval() # 设置为评估模式
        return model

    def _fit_scaler(self):
        print("Fitting scaler on training data...")
        all_series = []
        if not os.path.exists(self.train_dir):
            warnings.warn(f"Training directory not found: {self.train_dir}. Scaler will not be fitted.")
            return MinMaxScaler()

        for filename in os.listdir(self.train_dir):
            if filename.endswith('.csv'):
                df = pd.read_csv(os.path.join(self.train_dir, filename))[['cpu', 'memory', 'disk']]
                all_series.append(df)
        
        if not all_series:
            warnings.warn(f"No CSV files found in {self.train_dir}. Scaler will not be fitted.")
            return MinMaxScaler()
            
        scaler = MinMaxScaler()
        all_concat = pd.concat(all_series)
        scaler.fit(all_concat)
        print("Scaler fitted.")
        return scaler


    def _load_split(self, split='train'):
        target_dir = {'train': self.train_dir, 'val': self.val_dir, 'test': self.test_dir}[split]
        if not os.path.exists(target_dir):
            warnings.warn(f"Directory not found for split '{split}': {target_dir}. Returning empty dataset.")
            return Dataset()

        all_x, all_y = [], []
        for filename in os.listdir(target_dir):
            if filename.endswith('.csv'):
                df = pd.read_csv(os.path.join(target_dir, filename))[['cpu', 'memory', 'disk']]
                scaled_data = self.scaler.transform(df)
                
                for i in range(len(scaled_data) - self.seq_len - self.pred_len + 1):
                    all_x.append(scaled_data[i:i+self.seq_len])
                    all_y.append(scaled_data[i+self.seq_len:i+self.seq_len+self.pred_len])
        
        if not all_x:
            warnings.warn(f"No data generated for split '{split}' in {target_dir}.")
            return Dataset()
            
        all_x_np = np.array(all_x)
        all_y_np = np.array(all_y)

        # 使用Flow模型生成z
        with torch.no_grad():
            x_flat = torch.tensor(all_x_np.reshape(len(all_x_np), -1), dtype=torch.float32)
            z, _ = self.flow_model(x_flat)
            all_z_np = z.cpu().numpy()
            
        # 为训练集生成初始随机伪标签
        if split == 'train':
            initial_labels = np.random.randint(0, self.num_experts, size=len(all_x_np))
        else: # 验证集不需要伪标签
            initial_labels = np.zeros(len(all_x_np))

        return FullDataset(all_x_np, all_y_np, all_z_np, initial_labels)

    def get_train_loader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def get_val_loader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        
    def get_train_eval_loader(self):
        # 用于更新伪标签的非打乱数据加载器
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4) 