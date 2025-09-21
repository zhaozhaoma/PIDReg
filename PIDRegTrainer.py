import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy import stats
from sklearn.decomposition import PCA
import logging
import os
from tqdm import tqdm
import json
from torch.optim.lr_scheduler import ReduceLROnPlateau

class PIDRegTrainer:
    def __init__(self, config, pid_model):
        """
        PIDReg Trainer
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.pid_model = pid_model.to(self.device)
        
        self.modal1_name = config.get('modal1_name', 'Modal1')
        self.modal2_name = config.get('modal2_name', 'Modal2')
        
        model_params = list(self.pid_model.predictor.parameters()) + \
                      list(self.pid_model.modal1_projector.parameters()) + \
                      list(self.pid_model.modal2_projector.parameters())
            
        self.pred_optimizer = Adam(
            model_params,
            lr=config['learning_rate']
        )
        
        lambda_lr = config.get('lambda_learning_rate', 0.005)
        self.lambda_optimizer = Adam(
            [self.pid_model.fmri_lambda_param, self.pid_model.smri_lambda_param],
            lr=lambda_lr
        )
        print(f"Lambda parameters will use separate optimizer with learning rate: {lambda_lr}")
        
        # Add learning rate scheduler for predictor
        self.pred_scheduler = ReduceLROnPlateau(
            self.pred_optimizer, 
            mode='min', 
            factor=0.5,
            patience=10,
            verbose=True, 
            min_lr=1e-6
        )
        
        self.result_dir = config['result_dir']
        os.makedirs(self.result_dir, exist_ok=True)
        
        # Fixed loss weights
        self.loss_weights = {
            'pred': 1,       # Prediction loss weight
            'lcs': 0.1,      # LCS loss weight
            'cmi': 0.1,      # CMI loss weight
            'normality': 0.1  # Gauss loss weight
        }
        
        self.train_losses = {
            'total_loss': [], 'pred_loss': [], 
            'modal1_lcs_loss': [], 'modal2_lcs_loss': [],
            'modal1_cmi_loss': [], 'modal2_cmi_loss': [],
            'normality_loss': []
        }
        self.val_losses = {k: [] for k in self.train_losses.keys()}
        
        self.pid_params = {
            'unique_x': [], 
            'unique_y': [], 
            'redundancy': [], 
            'synergy': [],
            'epoch': []
        }
        
        self.lambda_history = {
            'modal1_lambda': [],
            'modal2_lambda': [],
            'epoch': [],
            'batch_modal1_lambda': [],
            'batch_modal2_lambda': [],
            'batch_index': []
        }
        
        self.epoch_pid_params = {
            'unique_x': [], 
            'unique_y': [], 
            'redundancy': [], 
            'synergy': []
        }
        
        self.window_size = 5
        self.current_fusion_weights = None
        self.pid_fixed = False
        self.pid_stability_counter = 0
        self.pid_stability_threshold = 5
        self.pid_change_threshold = 0.01
        self.prev_pid_values = None
        
        self.best_metrics = {
            'val_loss': float('inf'),
            'r2': -float('inf'),
            'rmse': float('inf'),
            'mae': float('inf')
        }

    def train_epoch(self, train_loader, epoch_idx):
        """
        Train one epoch
        """
        self.pid_model.train()
        
        current_epoch_pid = {
            'unique_x': [], 
            'unique_y': [], 
            'redundancy': [], 
            'synergy': []
        }
            
        epoch_losses = {k: 0.0 for k in self.train_losses.keys()}
        num_batches = len(train_loader)
        
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch_idx}", leave=True)
        
        current_modal1_lambda = self.pid_model.fmri_lambda.item()
        current_modal2_lambda = self.pid_model.smri_lambda.item()
        
        self.lambda_history['modal1_lambda'].append(current_modal1_lambda)
        self.lambda_history['modal2_lambda'].append(current_modal2_lambda)
        self.lambda_history['epoch'].append(epoch_idx)
        
        for batch_data in progress_bar:
            inputs_dict, labels = batch_data
            inputs_dict = {k: v.to(self.device) for k, v in inputs_dict.items()}
            labels = labels.to(self.device)
            self.pred_optimizer.zero_grad()
            self.lambda_optimizer.zero_grad()
            
            if self.pid_fixed:
                pred, losses, batch_pid_params = self.pid_model(inputs_dict, None)
            else:
                pred, losses, batch_pid_params = self.pid_model(inputs_dict, labels)
            
            if not self.pid_fixed and batch_pid_params is not None:
                current_epoch_pid['unique_x'].append(batch_pid_params[0])
                current_epoch_pid['unique_y'].append(batch_pid_params[1])
                current_epoch_pid['redundancy'].append(batch_pid_params[2])
                current_epoch_pid['synergy'].append(batch_pid_params[3])
            
            pred_loss = F.mse_loss(pred.view(-1), labels.view(-1))
            weighted_pred_loss = self.loss_weights['pred'] * pred_loss
            
            total_loss = (weighted_pred_loss + 
                        self.loss_weights['lcs'] * (losses['modal1_lcs'] + losses['modal2_lcs']) +
                        self.loss_weights['cmi'] * (losses['modal1_cmi'] + losses['modal2_cmi']) +
                        self.loss_weights['normality'] * losses['normality'])
            
            # Backward propagation
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.pid_model.predictor.parameters(), max_norm=1.0)
            
            # Update parameters
            self.pred_optimizer.step()
            self.lambda_optimizer.step()
            
            batch_idx = len(self.lambda_history['batch_index'])
            if batch_idx % 10 == 0:
                self.lambda_history['batch_modal1_lambda'].append(self.pid_model.fmri_lambda.item())
                self.lambda_history['batch_modal2_lambda'].append(self.pid_model.smri_lambda.item())
                self.lambda_history['batch_index'].append(batch_idx)
            
            epoch_losses['total_loss'] += total_loss.item()
            epoch_losses['pred_loss'] += pred_loss.item()
            epoch_losses['modal1_lcs_loss'] += losses['modal1_lcs'].item()
            epoch_losses['modal2_lcs_loss'] += losses['modal2_lcs'].item()
            epoch_losses['modal1_cmi_loss'] += losses['modal1_cmi'].item()
            epoch_losses['modal2_cmi_loss'] += losses['modal2_cmi'].item()
            epoch_losses['normality_loss'] += losses['normality'].item()
                
            progress_bar.set_postfix({
                'pred_loss': f"{pred_loss.item():.4f}",
                'total_loss': f"{total_loss.item():.4f}"
            })
        
        for k in epoch_losses:
            epoch_losses[k] /= num_batches
            self.train_losses[k].append(epoch_losses[k])
        
        if not self.pid_fixed and current_epoch_pid['unique_x']:
            avg_ux = np.mean(current_epoch_pid['unique_x'])
            avg_uy = np.mean(current_epoch_pid['unique_y'])
            avg_r = np.mean(current_epoch_pid['redundancy'])
            avg_s = np.mean(current_epoch_pid['synergy'])
            
            self.epoch_pid_params['unique_x'].append(avg_ux)
            self.epoch_pid_params['unique_y'].append(avg_uy)
            self.epoch_pid_params['redundancy'].append(avg_r)
            self.epoch_pid_params['synergy'].append(avg_s)
            
            self.pid_params['unique_x'].append(avg_ux)
            self.pid_params['unique_y'].append(avg_uy)
            self.pid_params['redundancy'].append(avg_r)
            self.pid_params['synergy'].append(avg_s)
            self.pid_params['epoch'].append(epoch_idx)
            
            self.update_fusion_weights()

        self.save_fusion_weights()
        
        print("\nEpoch {} Summary:".format(epoch_idx))
        for k, v in epoch_losses.items():
            print(f"{k}: {v:.6f}")
        
        print(f"Current Lambda Values: {self.modal1_name} λ={current_modal1_lambda:.4f}, {self.modal2_name} λ={current_modal2_lambda:.4f}")
        
        if self.current_fusion_weights is not None:
            print(f"Current Fusion Weights: w1={self.current_fusion_weights[0]:.4f}, "
                  f"w2={self.current_fusion_weights[1]:.4f}, "
                  f"w3={self.current_fusion_weights[2]:.4f}")
            
            if self.pid_fixed:
                print(f"PID parameters have been fixed, no longer computing new PID values")
        
        return epoch_losses
        
    def update_fusion_weights(self):
        """Update fusion weights using sliding window to compute average PID parameters"""
        # If PID is fixed, print information but don't update weights
        if self.pid_fixed:
            print(f"PID parameters have been fixed, using fixed fusion weights: w1={self.current_fusion_weights[0]:.4f}, "
                  f"w2={self.current_fusion_weights[1]:.4f}, "
                  f"w3={self.current_fusion_weights[2]:.4f}")
            return
        
        if len(self.epoch_pid_params['unique_x']) <= 1:
            start_idx = 0
        elif len(self.epoch_pid_params['unique_x']) <= self.window_size:
            start_idx = 0
        else:
            start_idx = len(self.epoch_pid_params['unique_x']) - self.window_size
        
        avg_ux = np.mean(self.epoch_pid_params['unique_x'][start_idx:])
        avg_uy = np.mean(self.epoch_pid_params['unique_y'][start_idx:])
        avg_r = np.mean(self.epoch_pid_params['redundancy'][start_idx:])
        avg_s = np.mean(self.epoch_pid_params['synergy'][start_idx:])
        
        current_pid_values = (avg_ux, avg_uy, avg_r, avg_s)
        if self.prev_pid_values is not None:
            changes = [abs(curr - prev) for curr, prev in zip(current_pid_values, self.prev_pid_values)]
            max_change = max(changes)
            
            if max_change < self.pid_change_threshold:
                self.pid_stability_counter += 1
                print(f"PID parameters stabilizing: {self.pid_stability_counter}/{self.pid_stability_threshold}, "
                      f"max change: {max_change:.6f}")
                
                if self.pid_stability_counter >= self.pid_stability_threshold:
                    print(f"PID parameters have reached stability criteria, fixing fusion weights")
                    self.pid_fixed = True
            else:
                self.pid_stability_counter = 0
                print(f"PID parameters changed significantly: {max_change:.6f} > {self.pid_change_threshold}, resetting stability counter")
        
        # Update previous PID values
        self.prev_pid_values = current_pid_values
        
        total = avg_ux + avg_uy + avg_r + avg_s
        eps = 1e-10
        if total < eps:
            w1 = w2 = w3 = 1/3
        else:
            # Bernoulli distribution
            bernoulli_sample = np.random.binomial(1, 0.5)
            if bernoulli_sample == 1:
                # ri assigned to w1
                w1 = (avg_ux + avg_r) / total
                w2 = avg_uy / total
            else:
                # ri assigned to w2
                w1 = avg_ux / total
                w2 = (avg_uy + avg_r) / total
            w3 = avg_s / total
        
        self.current_fusion_weights = (w1, w2, w3)
        self.pid_model.set_fusion_weights(w1, w2, w3)

    def validate(self, val_loader):
        """
        Evaluate
        """
        self.pid_model.eval()
        
        val_losses = {k: 0.0 for k in self.val_losses.keys()}
        num_batches = len(val_loader)
        
        progress_bar = tqdm(val_loader, desc="Validating", leave=True)
        
        with torch.no_grad():
            for batch_data in progress_bar:
                inputs_dict, labels = batch_data
                inputs_dict = {k: v.to(self.device) for k, v in inputs_dict.items()}
                labels = labels.to(self.device)
                
                pred, losses, _ = self.pid_model(inputs_dict, None)
                pred_loss = F.mse_loss(pred.view(-1), labels.view(-1))
                
                total_loss = (self.loss_weights['pred'] * pred_loss + 
                            self.loss_weights['lcs'] * (losses['modal1_lcs'] + losses['modal2_lcs']) +
                            self.loss_weights['cmi'] * (losses['modal1_cmi'] + losses['modal2_cmi']) +
                            self.loss_weights['normality'] * losses['normality'])

                val_losses['total_loss'] += total_loss.item()
                val_losses['pred_loss'] += pred_loss.item()
                val_losses['modal1_lcs_loss'] += losses['modal1_lcs'].item()
                val_losses['modal2_lcs_loss'] += losses['modal2_lcs'].item()
                val_losses['modal1_cmi_loss'] += losses['modal1_cmi'].item()
                val_losses['modal2_cmi_loss'] += losses['modal2_cmi'].item()
                val_losses['normality_loss'] += losses['normality'].item()
                
                progress_bar.set_postfix({
                    'val_pred_loss': f"{pred_loss.item():.4f}",
                    'val_total_loss': f"{total_loss.item():.4f}"
                })
        
        print("\nValidation Results:")
        for k in val_losses:
            val_losses[k] /= num_batches
            self.val_losses[k].append(val_losses[k])
            print(f"{k}: {val_losses[k]:.6f}")
        
        print(f"Current Lambda Values: {self.modal1_name} λ={self.pid_model.fmri_lambda.item():.4f}, "
              f"{self.modal2_name} λ={self.pid_model.smri_lambda.item():.4f}")
        
        return val_losses

    def evaluate(self, test_loader, scalers=None):
        """
        Test
        """
        self.pid_model.eval()
            
        predictions = []
        true_labels = []
        fusion_features = []
        
        with torch.no_grad():
            for batch_data in tqdm(test_loader, desc="Evaluating"):
                inputs_dict, labels = batch_data
                
                inputs_dict = {k: v.to(self.device) for k, v in inputs_dict.items()}
                labels = labels.to(self.device)
                
                pred = self.pid_model.predict(inputs_dict)
                features = self.pid_model.get_fusion_features(inputs_dict)
                
                predictions.append(pred.cpu())
                fusion_features.append(features.cpu())
                true_labels.append(labels.cpu())
        
        predictions = torch.cat(predictions).numpy()
        fusion_features = torch.cat(fusion_features).numpy()
        true_labels = torch.cat(true_labels).numpy()
        
        if scalers is not None and 'labels' in scalers:
            predictions = scalers['labels'].inverse_transform(predictions.reshape(-1, 1)).ravel()
            true_labels = scalers['labels'].inverse_transform(true_labels.reshape(-1, 1)).ravel()
        
        metrics = {
            'MSE': mean_squared_error(true_labels, predictions),
            'RMSE': np.sqrt(mean_squared_error(true_labels, predictions)),
            'MAE': mean_absolute_error(true_labels, predictions),
            'R2': r2_score(true_labels, predictions)
        }
        
        print("\nTest Results:")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.6f}")
        
        print(f"Final Lambda Values: {self.modal1_name} λ={self.pid_model.fmri_lambda.item():.4f}, "
              f"{self.modal2_name} λ={self.pid_model.smri_lambda.item():.6f}")
        
        return {
            'predictions': predictions,
            'fusion_features': fusion_features,
            'true_labels': true_labels,
            'metrics': metrics
        }



    def save_fusion_weights(self):
        if self.current_fusion_weights is not None:
            weights_file = os.path.join(self.result_dir, 'fusion_weights.json')
            with open(weights_file, 'w') as f:
                weights = {
                    'w1': float(self.current_fusion_weights[0]),
                    'w2': float(self.current_fusion_weights[1]),
                    'w3': float(self.current_fusion_weights[2]),
                    'epochs': len(self.epoch_pid_params['unique_x']),
                    'pid_fixed': self.pid_fixed,
                    f'{self.modal1_name}_lambda': float(self.pid_model.fmri_lambda.item()),
                    f'{self.modal2_name}_lambda': float(self.pid_model.smri_lambda.item())
                }
                json.dump(weights, f, indent=4)



    def save_model(self, save_path, epoch, metrics=None):
        """
        Save model
        """
        checkpoint = {
            'model_state_dict': self.pid_model.state_dict(),
            'config': self.config,
            'epoch': epoch,
            'fusion_weights': self.current_fusion_weights if self.current_fusion_weights else (1/3, 1/3, 1/3),
            'pid_fixed': self.pid_fixed,
            'pid_values': self.prev_pid_values,
            f'{self.modal1_name}_lambda': self.pid_model.fmri_lambda.item(),
            f'{self.modal2_name}_lambda': self.pid_model.smri_lambda.item()
        }
        
        if metrics is not None:
            checkpoint['metrics'] = metrics
        
        torch.save(checkpoint, save_path)
        print(f"Model saved to {save_path}")

    def load_model(self, load_path):
        """
        Load model
        """
        checkpoint = torch.load(load_path)
        self.pid_model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'pid_fixed' in checkpoint:
            self.pid_fixed = checkpoint['pid_fixed']
            self.prev_pid_values = checkpoint['pid_values']
        
        if 'fusion_weights' in checkpoint:
            w1, w2, w3 = checkpoint['fusion_weights']
            self.pid_model.set_fusion_weights(w1, w2, w3)
            self.current_fusion_weights = (w1, w2, w3)
        
        print(f"Model loaded from {load_path}")
        
        return checkpoint.get('metrics', None)