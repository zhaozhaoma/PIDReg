import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import os

class CustomCSVDataset(Dataset):
    """
    Custom dataset class for CSV data
    """
    def __init__(self, modal1_data, modal2_data, labels):
        """
        Initialize dataset
        """
        self.modal1_data = modal1_data
        self.modal2_data = modal2_data
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        modal1 = self.modal1_data[idx]
        modal2 = self.modal2_data[idx]
        label = self.labels[idx]
        
        return {'modal1': modal1, 'modal2': modal2}, label

def load_csv_data(csv_path, test_size=0.2, val_size=0.1, random_state=42):
    """
    Load CSV data and split into training, validation, and test sets
    """
    print(f"Reading CSV file: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Superconductivity format
    modal1 = df.iloc[:, 0:81].values        # Columns 0-80 for Modal 1
    modal2 = df.iloc[:, 81:167].values      # Columns 81-166 for Modal 2
    labels = df.iloc[:, 167].values         # Column 167 for Target
    
    print(f"Modal1 shape: {modal1.shape}")
    print(f"Modal2 shape: {modal2.shape}")
    print(f"Labels shape: {labels.shape}")
    
    scaler_modal1 = StandardScaler()
    scaler_modal2 = StandardScaler()
    scaler_labels = StandardScaler()
    
    modal1_scaled = scaler_modal1.fit_transform(modal1)
    modal2_scaled = scaler_modal2.fit_transform(modal2)
    labels_scaled = scaler_labels.fit_transform(labels.reshape(-1, 1)).ravel()
    
    X_train_modal1, X_test_modal1, X_train_modal2, X_test_modal2, y_train, y_test = train_test_split(
        modal1_scaled, modal2_scaled, labels_scaled, 
        test_size=test_size, random_state=random_state, stratify=None
    )
    
    if val_size > 0:
        val_size_adjusted = val_size / (1 - test_size)
        X_train_modal1, X_val_modal1, X_train_modal2, X_val_modal2, y_train, y_val = train_test_split(
            X_train_modal1, X_train_modal2, y_train,
            test_size=val_size_adjusted, random_state=random_state, stratify=None
        )
    else:
        X_val_modal1 = X_val_modal2 = y_val = None
    
    train_data = (
        torch.FloatTensor(X_train_modal1),
        torch.FloatTensor(X_train_modal2),
        torch.FloatTensor(y_train)
    )
    
    if val_size > 0:
        val_data = (
            torch.FloatTensor(X_val_modal1),
            torch.FloatTensor(X_val_modal2),
            torch.FloatTensor(y_val)
        )
    else:
        val_data = None
    
    test_data = (
        torch.FloatTensor(X_test_modal1),
        torch.FloatTensor(X_test_modal2),
        torch.FloatTensor(y_test)
    )
    
    scalers = {
        'modal1': scaler_modal1,
        'modal2': scaler_modal2,
        'labels': scaler_labels
    }
    
    feature_dims = (modal1.shape[1], modal2.shape[1])
    
    print(f"Data split completed:")
    print(f"  Training samples: {len(y_train)}")
    if val_size > 0:
        print(f"  Validation samples: {len(y_val)}")
    print(f"  Test samples: {len(y_test)}")
    
    return train_data, val_data, test_data, feature_dims, scalers

def create_csv_dataloaders(train_data, val_data, test_data, batch_size=32, num_workers=4):
    """
    Create data loaders
    """
    train_dataset = CustomCSVDataset(*train_data)
    test_dataset = CustomCSVDataset(*test_data)
    
    if val_data is not None:
        val_dataset = CustomCSVDataset(*val_data)
    else:
        val_dataset = test_dataset
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader