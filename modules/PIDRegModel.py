import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import sys
from src.pid import exact_gauss_tilde_pid
from CMICalculator import ConditionalMICalculatorTorch

class PIDRegModel(nn.Module):
    """
    PIDReg Version that processes static features
    """
    def __init__(self, hidden_dim=256, latent_dim=64, modal1_dim=None, modal2_dim=None, 
                 fmri_lambda=0.7, smri_lambda=0.7):
        super().__init__()
        
        self.modal1_dim = modal1_dim
        self.modal2_dim = modal2_dim
        
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # modal1 feature projection
        self.modal1_projector = nn.Sequential(
            nn.Linear(modal1_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, latent_dim),
            nn.BatchNorm1d(latent_dim)
        )

        # modal2 feature projection
        self.modal2_projector = nn.Sequential(
            nn.Linear(modal2_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, latent_dim),
            nn.BatchNorm1d(latent_dim)
        )

        # Information bottleneck parameters
        self.fmri_lambda_param = nn.Parameter(torch.tensor(float(fmri_lambda)))
        self.smri_lambda_param = nn.Parameter(torch.tensor(float(smri_lambda)))
        
        # Initialize CMI calculator
        self.cmi_calculator = ConditionalMICalculatorTorch(eta=0.05)
        
        # Predictor network
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1) 
        )
        
        self._initialize_weights()
        self.fusion_weights = (1/3, 1/3, 1/3)  # Default to equal weights
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def set_fusion_weights(self, w1, w2, w3):
        self.fusion_weights = (w1, w2, w3)
    
    @property
    def fmri_lambda(self):
        return torch.sigmoid(self.fmri_lambda_param).clamp(0.001, 0.999)
    
    @property
    def smri_lambda(self):
        return torch.sigmoid(self.smri_lambda_param).clamp(0.001, 0.999)

    def apply_info_bottleneck(self, features, projector, lambda_val):
        """
        Apply information bottleneck
        """
        projected = projector(features)
        
        proj_mean = projected.mean(dim=0, keepdim=True)
        proj_var = projected.var(dim=0, keepdim=True) + 1e-8
        
        epsilon = torch.randn_like(projected)
        scaled_noise = epsilon * torch.sqrt(proj_var) + proj_mean
        
        z = lambda_val * projected + (1 - lambda_val) * scaled_noise
        
        return projected, z

    def extract_modal1_features(self, modal1_data):
        """
        Process first modality features
        """
        # Ensure data is on correct device
        device = next(self.parameters()).device
        features = modal1_data.to(device)
        # Apply information bottleneck using current computed lambda value
        mean, z = self.apply_info_bottleneck(features, self.modal1_projector, self.fmri_lambda)
        
        return features, mean, z

    def extract_modal2_features(self, modal2_data):
        """
        Process second modality features
        """
        # Ensure data is on correct device
        device = next(self.parameters()).device
        features = modal2_data.to(device)
        # Apply information bottleneck using current computed lambda value
        mean, z = self.apply_info_bottleneck(features, self.modal2_projector, self.smri_lambda)
        
        return features, mean, z

    def compute_normality_loss(self, features):
        """
        Compute normality loss
        """
        try:
            device = features.device
            B, d = features.shape
            
            X = features - features.mean(dim=0, keepdim=True)
            
            try:
                U, s, Vt = torch.linalg.svd(X, full_matrices=False)
                s = torch.clamp(s, min=1e-6)
                Z = U * torch.sqrt(torch.tensor(B-1., device=device))
            except RuntimeError:
                try:
                    U, s, Vt = torch.linalg.svd(X.cpu(), full_matrices=False)
                    U, s, Vt = U.to(device), s.to(device), Vt.to(device)
                    s = torch.clamp(s, min=1e-6)
                    Z = U * torch.sqrt(torch.tensor(B-1., device=device))
                except RuntimeError:
                    return torch.tensor(1.0, device=device)
            
            z_vec = Z.reshape(-1)
            z_sorted, _ = torch.sort(z_vec)

            n_total = len(z_vec)
            p = (torch.arange(1, n_total+1, device=device) - 3/8) / (n_total + 1/4)
            m = torch.distributions.Normal(0, 1).icdf(p)
            
            if n_total <= 5000:
                c = 1 / torch.sqrt(torch.tensor(n_total, device=device))
                a = m / torch.norm(m) + c * (m**3 - 3*m) / 6
            else:
                a = m / torch.norm(m)
            
            numerator = torch.sum(a * z_sorted) ** 2
            denominator = torch.sum((z_vec - torch.mean(z_vec)) ** 2)
            W = numerator / (denominator + 1e-10)
            
            normality_loss = -torch.log(W + 1e-10)
            
            if not torch.isfinite(normality_loss):
                return torch.tensor(1.0, device=device)
                
            return normality_loss
            
        except Exception as e:
            logging.error(f"Normality loss computation error: {str(e)}")
            return torch.tensor(1.0, device=device)

    def robust_gaussianize(self, x, strength=0.5):
        """
        Precprocess target
        """
        device = x.device
        x_np = x.detach().cpu().numpy()
        gaussianized_data = np.zeros_like(x_np)
        
        for i in range(x_np.shape[1]):
            column = x_np[:, i]
            n = len(column)
            
            mean_val = np.mean(column)
            std_val = np.std(column)
            
            q1, q3 = np.percentile(column, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 2 * iqr
            upper_bound = q3 + 2 * iqr
            valid_mask = (column >= lower_bound) & (column <= upper_bound)
            valid_data = column[valid_mask]
            
            if len(valid_data) < 2:
                gaussianized_data[:, i] = column
                continue
                
            ranks = np.zeros(n)
            valid_ranks = np.argsort(np.argsort(valid_data))
            ranks[valid_mask] = valid_ranks
            
            n_valid = len(valid_data)
            quantiles = (ranks[valid_mask] + 0.5) / (n_valid + 1)
            
            from scipy import special as sp
            gaussian_values = np.zeros(n)
            
            gaussian_values[valid_mask] = np.sqrt(2) * sp.erfinv(2 * quantiles - 1)
            
            if not np.all(valid_mask):
                lower_mask = column < lower_bound
                upper_mask = column > upper_bound
                
                if np.any(lower_mask):
                    relative_dev = (column[lower_mask] - lower_bound) / (std_val + 1e-10)
                    max_dev = np.clip(relative_dev, -3, 0)
                    gaussian_values[lower_mask] = -2.5 + 0.5 * max_dev
                
                if np.any(upper_mask):
                    relative_dev = (column[upper_mask] - upper_bound) / (std_val + 1e-10)
                    max_dev = np.clip(relative_dev, 0, 3)
                    gaussian_values[upper_mask] = 2.5 + 0.5 * max_dev
            
            standardized_orig = (column - mean_val) / (std_val + 1e-10)
            standardized_orig = np.clip(standardized_orig, -3, 3)
            
            mixed_values = (1 - strength) * standardized_orig + strength * gaussian_values
            
            gaussianized_data[:, i] = mixed_values
        
        return torch.tensor(gaussianized_data, device=device, dtype=x.dtype)

    def compute_pid_weights(self, a1, a2, target):
        """
        Compute fusion weights
        """
        try:
            eps = 1e-6
            a1_diff = (a1 - a1.mean(dim=0, keepdim=True)) / (a1.std(dim=0, unbiased=False, keepdim=True) + eps)
            a2_diff = (a2 - a2.mean(dim=0, keepdim=True)) / (a2.std(dim=0, unbiased=False, keepdim=True) + eps)
            
            target_np = target.detach().cpu().numpy()
            if len(target_np.shape) == 1:
                target_np = target_np.reshape(-1, 1)
            
            target_gauss = self.robust_gaussianize(torch.tensor(target_np, device=a1.device), strength=0.6)
            features_tensor_diff = torch.cat([target_gauss, a1_diff, a2_diff], dim=1)
            
            normality_loss = self.compute_normality_loss(features_tensor_diff)
            a1_np = a1.detach().cpu().numpy()
            a2_np = a2.detach().cpu().numpy()
            
            a1_mean = np.mean(a1_np, axis=0)
            a2_mean = np.mean(a2_np, axis=0)
            
            a1_std = np.std(a1_np, axis=0, ddof=0)
            a2_std = np.std(a2_np, axis=0, ddof=0)
            
            a1_std[a1_std < eps] = eps
            a2_std[a2_std < eps] = eps
            
            a1_std = (a1_np - a1_mean) / a1_std
            a2_std = (a2_np - a2_mean) / a2_std
            
            a1_std = np.nan_to_num(a1_std, nan=0.0, posinf=0.0, neginf=0.0)
            a2_std = np.nan_to_num(a2_std, nan=0.0, posinf=0.0, neginf=0.0)
            
            target_gauss_np = self.robust_gaussianize(torch.tensor(target_np), strength=0.6).cpu().numpy()
            target_gauss_np = np.nan_to_num(target_gauss_np, nan=0.0, posinf=0.0, neginf=0.0)
            
            features = np.concatenate([target_gauss_np, a1_std, a2_std], axis=1)
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            
            noise = np.random.normal(0, 1e-6, features.shape)
            features += noise
            
            jitter = 1e-6
            cov_success = False
            
            for _ in range(5):
                try:
                    cov_matrix = np.cov(features.T) + jitter * np.eye(features.shape[1])
                    _ = np.linalg.cholesky(cov_matrix)
                    cov_success = True
                    break
                except np.linalg.LinAlgError:
                    jitter *= 10 
            
            if not cov_success:
                return (torch.tensor(1/3).to(a1.device), 
                        torch.tensor(1/3).to(a1.device), 
                        torch.tensor(1/3).to(a1.device),
                        normality_loss,
                        (0.0, 0.0, 0.0, 0.0))
            
            try:
                if len(target.shape) == 1:
                    dm = 1
                else:
                    dm = target.shape[1]
                dx = a1_std.shape[1]
                dy = a2_std.shape[1]
                
                pid_results = exact_gauss_tilde_pid(cov_matrix, dm, dx, dy)
                uix, uiy, ri, si = pid_results[5:9]
                
                uix = max(0, uix)
                uiy = max(0, uiy)
                ri = max(0, ri)
                si = max(0, si)
            except Exception as e:
                logging.error(f"PID computation error: {str(e)}")
                return (torch.tensor(1/3).to(a1.device), 
                        torch.tensor(1/3).to(a1.device), 
                        torch.tensor(1/3).to(a1.device),
                        normality_loss,
                        (0.0, 0.0, 0.0, 0.0))
            
            total = uix + uiy + ri + si
            if total < eps:
                return (torch.tensor(1/3).to(a1.device), 
                    torch.tensor(1/3).to(a1.device), 
                    torch.tensor(1/3).to(a1.device),
                    normality_loss,
                    (0.0, 0.0, 0.0, 0.0))
            
            # Bernoulli distribution
            bernoulli_sample = np.random.binomial(1, 0.5)
            if bernoulli_sample == 1:
                w1 = torch.tensor((uix + ri) / total).to(a1.device)
                w2 = torch.tensor(uiy / total).to(a1.device)
            else:
                w1 = torch.tensor(uix / total).to(a1.device)
                w2 = torch.tensor((uiy + ri) / total).to(a1.device)
            w3 = torch.tensor(si / total).to(a1.device)
            
            return w1, w2, w3, normality_loss, (float(uix), float(uiy), float(ri), float(si))
                
        except Exception as e:
            logging.error(f"PID weight computation error: {str(e)}")
            return (torch.tensor(1/3).to(a1.device), 
                    torch.tensor(1/3).to(a1.device), 
                    torch.tensor(1/3).to(a1.device),
                    torch.tensor(0.0).to(a1.device),
                    (0.0, 0.0, 0.0, 0.0))
        
    def fuse_features(self, a1, a2, target=None):
        """
        Fuse features
        """
        # element-wise product
        a3 = a1 * a2
        
        if target is None:
            w1, w2, w3 = self.fusion_weights
            w1 = torch.tensor(w1).to(a1.device)
            w2 = torch.tensor(w2).to(a1.device)
            w3 = torch.tensor(w3).to(a1.device)
            z = w1 * a1 + w2 * a2 + w3 * a3
            return z, torch.tensor(0.0).to(a1.device), None
            
        w1, w2, w3, normality_loss, pid_params = self.compute_pid_weights(a1, a2, target)
        
        # Weighted fusion
        z = w1 * a1 + w2 * a2 + w3 * a3
        
        return z, normality_loss, pid_params
        
    def forward(self, x_dict, target=None):
        """
        Forward propagation
        """
        modal1_data = x_dict['modal1'] if 'modal1' in x_dict else x_dict['fmri']
        modal2_data = x_dict['modal2'] if 'modal2' in x_dict else x_dict['smri']
        
        x1_raw, a1_mean, a1_noise = self.extract_modal1_features(modal1_data)
        x2_raw, a2_mean, a2_noise = self.extract_modal2_features(modal2_data)
        
        losses = {}
        
        # Compute CS divergence
        losses['modal1_lcs'] = self.compute_cs_divergence(a1_mean)
        losses['modal2_lcs'] = self.compute_cs_divergence(a2_mean)
        # Conditional mutual information computation
        device = a1_mean.device
        # CMI computation: I(Z1,X2|X1)
        cmi_loss1 = self.cmi_calculator.compute_conditional_mi(
            latent=a1_noise,
            target=x2_raw, 
            condition=x1_raw 
        )
        # CMI computation: I(Z2,X1|X2)
        cmi_loss2 = self.cmi_calculator.compute_conditional_mi(
            latent=a2_noise, 
            target=x1_raw, 
            condition=x2_raw 
        )
        
        losses['modal1_cmi'] = cmi_loss1
        losses['modal2_cmi'] = cmi_loss2
        
        z, normality_loss, pid_params = self.fuse_features(a1_noise, a2_noise, target)
        losses['normality'] = normality_loss
        
        pred = self.predictor(z)
        
        return pred, losses, pid_params
    
    def compute_cs_divergence(self, z):
        """
        Compute LCS
        """
        M = z.size(0) 
        N = M 
        eps = 1e-8
        true_samples = torch.randn_like(z)
        # Compute κ(z_i^p, z_j^p)
        dist_p = torch.cdist(z, z, p=2) 
        sigma = torch.median(dist_p) + eps
        K_pp = torch.exp(-0.5 * (dist_p / sigma)**2)
        sum_K_pp = torch.sum(K_pp)
        term1 = torch.log(sum_K_pp / (M*M) + eps)
        
        # Compute κ(z_i^q, z_j^q)
        dist_q = torch.cdist(true_samples, true_samples, p=2)
        K_qq = torch.exp(-0.5 * (dist_q / sigma)**2)
        sum_K_qq = torch.sum(K_qq)
        term2 = torch.log(sum_K_qq / (N*N) + eps)
        
        # Compute κ(z_i^p, z_j^q)
        dist_pq = torch.cdist(z, true_samples, p=2)
        K_pq = torch.exp(-0.5 * (dist_pq / sigma)**2)
        sum_K_pq = torch.sum(K_pq)
        term3 = 2 * torch.log(sum_K_pq / (M*N) + eps)
        
        # Compute CS divergence
        cs_div = term1 + term2 - term3
        cs_div = torch.clamp(cs_div, min=0.0)
        
        return cs_div

    def predict(self, x_dict):
        """
        Prediction
        """
        self.eval()
        with torch.no_grad():
            _, a1_mean, _ = self.extract_modal1_features(x_dict['modal1'] if 'modal1' in x_dict else x_dict['fmri'])
            _, a2_mean, _ = self.extract_modal2_features(x_dict['modal2'] if 'modal2' in x_dict else x_dict['smri'])
            z, _, _ = self.fuse_features(a1_mean, a2_mean)
            pred = self.predictor(z)
        return pred
    
    def get_fusion_features(self, x_dict):
        """
        Get fusion features
        """
        self.eval()
        with torch.no_grad():
            _, a1_mean, _ = self.extract_modal1_features(x_dict['modal1'] if 'modal1' in x_dict else x_dict['fmri'])
            _, a2_mean, _ = self.extract_modal2_features(x_dict['modal2'] if 'modal2' in x_dict else x_dict['smri'])
            z, _, _ = self.fuse_features(a1_mean, a2_mean)
        return z
