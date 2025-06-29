import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import os

from models.m2_moep import M2_MOEP
from utils.losses import CompositeLoss
from utils.metrics import MetricsEvaluator

class Trainer:
    def __init__(self, config, data_module):
        self.config = config
        self.data_module = data_module
        self.train_loader = data_module.get_train_loader()
        self.val_loader = data_module.get_val_loader()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model = M2_MOEP(config).to(self.device)
        self.loss_fn = CompositeLoss(config).to(self.device)
        self.optimizer = self._create_optimizer()
        
        self.evaluator = MetricsEvaluator()
        
        # --- 用于模型保存的变量 ---
        self.best_val_metric = float('inf')
        self.run_dir = config.get('run_dir', 'runs')
        self.exp_dir = os.path.join(self.run_dir, config.get('run_name', 'default_exp'))
        os.makedirs(self.exp_dir, exist_ok=True)
        print(f"Models and logs will be saved to: {self.exp_dir}")

    def _create_optimizer(self):
        params = list(self.model.parameters()) + list(self.loss_fn.parameters())
        if self.config['training']['optimizer'] == 'Adam':
            return optim.Adam(params, lr=self.config['training']['learning_rate'])
        else:
            raise NotImplementedError

    def train_epoch(self, epoch):
        self.model.train()
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        
        for i, (x, y, z, pseudo_labels) in progress_bar:
            x, y, z, pseudo_labels = x.to(self.device), y.to(self.device), z.to(self.device), pseudo_labels.to(self.device)

            self.optimizer.zero_grad()
            
            predictions, _, embeddings = self.model(x, z)
            
            total_loss, loss_dict = self.loss_fn(predictions, embeddings, y, pseudo_labels)
            
            total_loss.backward()
            self.optimizer.step()
            
            progress_bar.set_postfix({k: f"{v:.4f}" for k, v in loss_dict.items()})

    def evaluate(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        
        scaler = self.data_module.scaler
        # 获取特征数量，用于反归一化
        num_features = scaler.n_features_in_
        
        with torch.no_grad():
            for x, y, z, _ in self.val_loader:
                x, y, z = x.to(self.device), y.to(self.device), z.to(self.device)
                
                predictions, _, _ = self.model(x, z)
                
                # --- 反归一化以获得真实尺度的评估指标 ---
                # 调整形状为 [n_samples, n_features] 以适应scaler
                y_reshaped = y.squeeze(1).cpu().numpy().reshape(-1, num_features)
                pred_reshaped = predictions.cpu().numpy().reshape(-1, num_features)
                
                y_inv = scaler.inverse_transform(y_reshaped)
                pred_inv = scaler.inverse_transform(pred_reshaped)
                
                self.evaluator.update(torch.from_numpy(y_inv), torch.from_numpy(pred_inv))

        metrics = self.evaluator.get_results()
        # 使用f-string格式化输出
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        print(f"Epoch {epoch+1} Validation Metrics: {metrics_str}")
        return metrics

    def update_triplet_labels(self):
        """
        Dynamically updates pseudo-labels based on which expert performs best for each sample.
        """
        print("\nUpdating pseudo-labels for triplet mining...")
        self.model.eval()
        train_eval_loader = self.data_module.get_train_eval_loader()
        
        all_new_labels = []
        with torch.no_grad():
            for x, y, _, _ in tqdm(train_eval_loader, desc="Updating Labels"):
                x, y = x.to(self.device), y.to(self.device)
                
                # Get predictions from each expert individually
                expert_losses = []
                for expert in self.model.experts:
                    expert_pred = expert(x)
                    # Calculate per-sample MSE loss
                    loss = F.mse_loss(expert_pred, y.squeeze(1), reduction='none').mean(dim=1)
                    expert_losses.append(loss)
                
                # Stack losses and find the best expert (minimum loss) for each sample
                # Shape: [num_experts, batch_size] -> [batch_size, num_experts]
                expert_losses_stacked = torch.stack(expert_losses, dim=1)
                best_expert_indices = torch.argmin(expert_losses_stacked, dim=1)
                all_new_labels.append(best_expert_indices.cpu().numpy())
        
        # Concatenate all labels and update the dataset
        new_labels_full = np.concatenate(all_new_labels)
        self.data_module.train_dataset.update_labels(new_labels_full)
        print("Pseudo-labels updated successfully.")

    def save_checkpoint(self, epoch, val_metric):
        """保存模型检查点"""
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_fn_state_dict': self.loss_fn.state_dict(),
            'best_val_metric': self.best_val_metric
        }
        # 保存最新模型
        torch.save(state, os.path.join(self.exp_dir, 'latest_model.pth'))
        print(f"Saved latest model checkpoint at epoch {epoch+1}.")

        # 如果是最佳模型，则额外保存
        if val_metric < self.best_val_metric:
            self.best_val_metric = val_metric
            torch.save(state, os.path.join(self.exp_dir, 'best_model.pth'))
            print(f"Saved best model checkpoint with validation MAE: {val_metric:.4f}")

    def run(self):
        for epoch in range(self.config['training']['epochs']):
            # 每5个epoch更新一次伪标签 (除第一次外)
            if epoch > 0 and (epoch + 1) % 5 == 0:
                self.update_triplet_labels()

            self.train_epoch(epoch)
            val_metrics = self.evaluate(epoch)
            
            # 使用MAE作为评估指标来保存最佳模型
            self.save_checkpoint(epoch, val_metrics['MAE']) 