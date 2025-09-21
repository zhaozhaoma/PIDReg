import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import argparse
import json
from tqdm import tqdm
from PIDRegModel import PIDRegModel
from PIDRegTrainer import PIDRegTrainer
from csv_data_loader import load_csv_data, create_csv_dataloaders

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(SCRIPT_DIR, 'data')

def main():
    torch.manual_seed(1234)
    np.random.seed(1234)
    
    parser = argparse.ArgumentParser(description='PIDReg Training')
    parser.add_argument('--data_path', type=str, default=DEFAULT_DATA_DIR, help=f'Path to the directory containing CSV files' )
    parser.add_argument('--result_dir', type=str, default=SCRIPT_DIR, help='Directory where results will be saved')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--n_epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--window_size', type=int, default=5, help='PID parameter sliding window size')
    parser.add_argument('--early_stopping', type=int, default=30, help='Early stopping patience value')
    parser.add_argument('--init_modal1_lambda', type=float, default=4.0, help='Initial parameter for first modality information bottleneck strength')
    parser.add_argument('--init_modal2_lambda', type=float, default=4.0, help='Initial parameter for second modality information bottleneck strength')
    parser.add_argument('--lambda_lr', type=float, default=0.1, help='Learning rate for lambda parameters')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden layer dimension (recommended range: 128-512)')
    parser.add_argument('--latent_dim', type=int, default=64, help='Latent space dimension (recommended range: 32-128)')
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    csv_path = os.path.join(args.data_path, 'Superconductivity.csv')
    print(f"Loading CSV dataset: {csv_path}")
    
    train_data, val_data, test_data, feature_dims, scalers = load_csv_data(
        csv_path=csv_path,
        test_size=0.2,
        val_size=0.1
    )
    
    modal1_dim, modal2_dim = feature_dims
    
    train_loader, val_loader, test_loader = create_csv_dataloaders(
        train_data, val_data, test_data, 
        batch_size=args.batch_size
    )
    
    # Set modality names
    modal1_name = 'Modal1'
    modal2_name = 'Modal2'
    # Create result directory
    result_dir = os.path.join(args.result_dir, "pid_regression")
    os.makedirs(result_dir, exist_ok=True)
    
    # Configuration parameters
    config = {
        'hidden_dim': args.hidden_dim,
        'latent_dim': args.latent_dim,
        'batch_size': args.batch_size,
        'n_epochs': args.n_epochs,
        'learning_rate': 1e-3,
        'lambda_learning_rate': args.lambda_lr,
        'early_stopping_patience': args.early_stopping,
        'result_dir': result_dir,
        'modal1_name': modal1_name,
        'modal2_name': modal2_name
    }
    
    # Create model
    print(f"Creating PID regression model...")
    pid_model = PIDRegModel(
        hidden_dim=config['hidden_dim'],
        latent_dim=config['latent_dim'],
        modal1_dim=modal1_dim,
        modal2_dim=modal2_dim,
        fmri_lambda=args.init_modal1_lambda,
        smri_lambda=args.init_modal2_lambda
    ).to(device)
    
    # Create trainer
    trainer = PIDRegTrainer(config, pid_model)
    trainer.window_size = args.window_size
    
    # Train model
    print("Starting training...")
    print(f"Using learnable lambda parameters with initial values - {modal1_name}: {args.init_modal1_lambda}, {modal2_name}: {args.init_modal2_lambda}")
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config['n_epochs']):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{config['n_epochs']}")
        print(f"{'='*50}")
        
        train_losses = trainer.train_epoch(train_loader, epoch)
        val_losses = trainer.validate(val_loader)
        trainer.pred_scheduler.step(val_losses['pred_loss'])
        
        current_val_loss = val_losses['total_loss']
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            patience_counter = 0
            print(f"\nNew best model! Validation loss: {best_val_loss:.6f}")
            trainer.save_model(
                os.path.join(config['result_dir'], 'best_model.pth'),
                epoch,
                {'val_loss': best_val_loss}
            )
            
        else:
            patience_counter += 1
            print(f"\nNo improvement. Counter: {patience_counter}/{config['early_stopping_patience']}")
            if patience_counter >= config['early_stopping_patience']:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
    
    print("\nLoading best model for testing...")
    trainer.load_model(os.path.join(config['result_dir'], 'best_model.pth'))
    
    if trainer.current_fusion_weights:
        w1, w2, w3 = trainer.current_fusion_weights
        trainer.pid_model.set_fusion_weights(w1, w2, w3)
        print(f"Using saved fusion weights: w1={w1:.4f}, w2={w2:.4f}, w3={w3:.4f}")
    
    print("\nEvaluating model on test set...")
    test_results = trainer.evaluate(test_loader, scalers)
    
    metrics = test_results['metrics']
    
    print(f"\n{'='*50}")
    print(f"PID Regression Model Evaluation Report")
    print(f"{'='*50}")
    
    print(f"Modalities: {modal1_name} (dim={modal1_dim}), {modal2_name} (dim={modal2_dim})")
    
    if trainer.current_fusion_weights:
        print(f"\nFinal Fusion Weights:")
        print(f"  w1 ({modal1_name}): {trainer.current_fusion_weights[0]:.6f}")
        print(f"  w2 ({modal2_name}): {trainer.current_fusion_weights[1]:.6f}")
        print(f"  w3 (Synergy): {trainer.current_fusion_weights[2]:.6f}")
        print(f"  PID Fixed: {trainer.pid_fixed}")
    
    print(f"\nTest Results:")
    print(f"  MSE: {metrics['MSE']:.6f}")
    print(f"  RMSE: {metrics['RMSE']:.6f}")
    print(f"  MAE: {metrics['MAE']:.6f}")
    print(f"  R²: {metrics['R2']:.6f}")
    
    print(f"\nTraining and evaluation completed! Results saved to {config['result_dir']}")

if __name__ == "__main__":
    main()