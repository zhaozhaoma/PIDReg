import torch
import numpy as np

class ConditionalMICalculatorTorch:
    def __init__(self, eta=0.05):
        """
        Initialize conditional mutual information calculator
        """
        self.eta = eta
    
    @staticmethod
    def rbf_dot(x1, x2, sigma):
        """
        Compute RBF kernel matrix
        """
        device = x1.device
        G = torch.sum(x1 ** 2, dim=1)
        H = torch.sum(x2 ** 2, dim=1)
        Q = torch.outer(G, torch.ones(len(H), device=device)) + torch.outer(torch.ones(len(G), device=device), H) - 2 * torch.mm(x1, x2.T)
        return torch.exp(-Q / (2 * sigma ** 2))

    @staticmethod
    def comp_med(x):
        """
        Compute median for kernel width selection
        """
        device = x.device
        d, n = x.shape
        G = torch.sum(x ** 2, dim=0)
        T = G.repeat(n, 1)
        dist2 = T - 2 * torch.mm(x.T, x) + T.T
        mask = torch.tril(torch.ones_like(dist2, device=device))
        dist2 = dist2 - dist2 * mask
        R = dist2.flatten()
        R = R[R > 0]
        
        if len(R) == 0:
            print("Warning: R is empty, returning NaN")
            return torch.tensor(float('nan'), device=device)

        if R.dim() > 0:
            median_val = torch.median(R)
            if isinstance(median_val, tuple):
                median_val = median_val[0]
            else:
                median_val = median_val
        else:
            median_val = torch.tensor(float('nan'), device=device)
        
        return torch.sqrt(0.5 * median_val)

    def conditional_CS_CMI(self, condition, target, latent, sigma1, sigma2, sigma3):
        """
        Compute conditional CS-CMI strictly following the formula in Proposition 1.
        """
        M = self.rbf_dot(condition, condition, sigma1)  
        K = self.rbf_dot(target, target, sigma2)    
        L = self.rbf_dot(latent, latent, sigma3)   
        
        eps = 1e-6
        
        # First term: -2 * log(∑j (∑i Mji) * (∑i Kji*Mji) * (∑i Lji*Mji))
        M_sum_i = torch.sum(M, dim=1)             
        KM_sum_i = torch.sum(K * M, dim=1)          
        LM_sum_i = torch.sum(L * M, dim=1)          
        
        cross_product = M_sum_i * KM_sum_i * LM_sum_i  
        cross_term = torch.sum(cross_product)          
        cross_term = torch.clamp(cross_term, min=eps)  
        term1 = -2 * torch.log(cross_term)
        
        # Second term: log(∑j ((∑i Kji*Lji*Mji) * (∑i Lji)^2))
        KLM_sum_i = torch.sum(K * L * M, dim=1)  
        L_sum_i = torch.sum(L, dim=1)     
        L_sum_i_squared = L_sum_i ** 2 
        
        second_product = KLM_sum_i * L_sum_i_squared 
        second_term = torch.sum(second_product)
        second_term = torch.clamp(second_term, min=eps)
        term2 = torch.log(second_term)
        
        # Third term: log(∑j ((∑i Kji*Lji)^2 * (∑i Lji*Mji)^2 / (∑i Kji*Lji*Mji)))
        KL_sum_i = torch.sum(K * L, dim=1)
        KL_sum_i_squared = KL_sum_i ** 2
        LM_sum_i_squared = LM_sum_i ** 2 
        
        numerator = KL_sum_i_squared * LM_sum_i_squared
        denominator = torch.clamp(KLM_sum_i, min=eps) 
        ratio = numerator / denominator 
        
        third_term = torch.sum(ratio) 
        third_term = torch.clamp(third_term, min=eps)
        term3 = torch.log(third_term)
        
        # Final conditional divergence
        conditional_divergence = term1 + term2 + term3
        conditional_divergence = torch.clamp(conditional_divergence, min=0, max=1e4)
        
        return conditional_divergence

    def compute_conditional_mi(self, latent, target, condition):
        """
        Main function to compute conditional mutual information
        """
        device = latent.device
        
        combined_data = torch.cat([condition, target, latent], dim=1)
        med = self.comp_med(combined_data.T)
        
        if torch.isnan(med):
            return torch.tensor(float('inf'), device=device)
        
        sigma = med * self.eta
        
        cmi = self.conditional_CS_CMI(
            condition=condition, 
            target=target, 
            latent=latent, 
            sigma1=sigma,
            sigma2=sigma,
            sigma3=sigma
        )
        
        if torch.isnan(cmi):
            return torch.tensor(float('inf'), device=device)
        return cmi



