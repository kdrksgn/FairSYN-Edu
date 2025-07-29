import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import shap
from opacus import PrivacyEngine  # For DP-SGD
import uuid
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DiffusionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_steps=1000):
        super(DiffusionModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_steps = num_steps
        
        # Simple MLP architecture for score-based diffusion
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Noise schedule
        self.betas = torch.linspace(0.0001, 0.02, num_steps)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
    
    def forward(self, x, t):
        return self.net(x)  # Predict score/noise
    
    def sample(self, batch_size, device):
        """Generate synthetic samples using reverse diffusion process"""
        x = torch.randn(batch_size, self.input_dim, device=device)
        for t in reversed(range(self.num_steps)):
            t_tensor = torch.full((batch_size,), t, dtype=torch.long, device=device)
            noise = torch.randn_like(x) if t > 0 else 0
            x = (1 / torch.sqrt(self.alphas[t])) * (
                x - (1 - self.alphas[t]) / torch.sqrt(1 - self.alpha_bar[t]) * self(x, t_tensor)
            ) + torch.sqrt(self.betas[t]) * noise
        return x

class AdversarialDebiaser(nn.Module):
    def __init__(self, input_dim, sensitive_dim):
        super(AdversarialDebiaser, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, sensitive_dim),  # Predict sensitive attribute
        )
    
    def forward(self, x):
        return self.net(x)

class FairSYN_Edu:
    def __init__(self, input_dim, sensitive_dim, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = torch.device(device)
        self.generator = DiffusionModel(input_dim).to(self.device)
        self.adversary = AdversarialDebiaser(input_dim, sensitive_dim).to(self.device)
        self.input_dim = input_dim
        self.sensitive_dim = sensitive_dim
        
        # Optimizers
        self.gen_optimizer = optim.Adam(self.generator.parameters(), lr=1e-3)
        self.adv_optimizer = optim.Adam(self.adversary.parameters(), lr=1e-3)
        
        # Privacy parameters for DP-SGD
        self.privacy_engine = PrivacyEngine()
        self.delta = 1e-5
        self.noise_multiplier = 1.0
        self.max_grad_norm = 1.0
    
    def preprocess_data(self, data, categorical_cols, numerical_cols):
        """Preprocess data with one-hot encoding for categorical features and scaling for numerical features"""
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('cat', OneHotEncoder(sparse=False, handle_unknown='ignore'), categorical_cols)
            ])
        processed_data = preprocessor.fit_transform(data)
        return processed_data, preprocessor
    
    def train(self, data, sensitive_data, categorical_cols, numerical_cols, epochs=100, batch_size=128):
        """Train FairSYN-Edu with DP-SGD and adversarial debiasing"""
        # Preprocess data
        processed_data, self.preprocessor = self.preprocess_data(data, categorical_cols, numerical_cols)
        dataset = TensorDataset(
            torch.tensor(processed_data, dtype=torch.float32).to(self.device),
            torch.tensor(sensitive_data.values, dtype=torch.float32).to(self.device)
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Attach privacy engine for DP-SGD
        self.generator, self.gen_optimizer, dataloader = self.privacy_engine.make_private(
            module=self.generator,
            optimizer=self.gen_optimizer,
            data_loader=dataloader,
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=self.max_grad_norm
        )
        
        for epoch in range(epochs):
            gen_loss_total = 0.0
            adv_loss_total = 0.0
            for x, s in dataloader:
                batch_size = x.size(0)
                t = torch.randint(0, self.generator.num_steps, (batch_size,), device=self.device)
                
                # Forward diffusion: add noise
                noise = torch.randn_like(x)
                sqrt_alpha_bar = torch.sqrt(self.generator.alpha_bar[t]).view(-1, 1)
                sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.generator.alpha_bar[t]).view(-1, 1)
                x_noisy = sqrt_alpha_bar * x + sqrt_one_minus_alpha_bar * noise
                
                # Generator loss (score matching)
                predicted_noise = self.generator(x_noisy, t)
                gen_loss = nn.MSELoss()(predicted_noise, noise)
                
                # Adversarial loss (minimize ability to predict sensitive attributes)
                adv_pred = self.adversary(x_noisy.detach())
                adv_loss = nn.BCEWithLogitsLoss()(adv_pred, s)
                
                # Update generator
                self.gen_optimizer.zero_grad()
                gen_loss.backward()
                self.gen_optimizer.step()
                
                # Update adversary
                self.adv_optimizer.zero_grad()
                adv_loss.backward()
                self.adv_optimizer.step()
                
                gen_loss_total += gen_loss.item()
                adv_loss_total += adv_loss.item()
            
            logger.info(f"Epoch {epoch+1}/{epochs}, Gen Loss: {gen_loss_total/len(dataloader):.4f}, "
                       f"Adv Loss: {adv_loss_total/len(dataloader):.4f}")
    
    def generate_synthetic_data(self, num_samples):
        """Generate synthetic data samples"""
        self.generator.eval()
        with torch.no_grad():
            synthetic_data = self.generator.sample(num_samples, self.device)
        synthetic_data = synthetic_data.cpu().numpy()
        
        # Inverse transform to original feature space
        synthetic_data = self.preprocessor.inverse_transform(synthetic_data)
        return pd.DataFrame(synthetic_data, columns=self.preprocessor.get_feature_names_out())
    
    def evaluate_shap(self, synthetic_data, real_data, target_col):
        """Evaluate SHAP values for interpretability"""
        X_real = real_data.drop(columns=[target_col])
        y_real = real_data[target_col]
        X_synthetic = synthetic_data.drop(columns=[target_col])
        
        # Train a simple classifier (e.g., logistic regression)
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression().fit(X_real, y_real)
        
        # Compute SHAP values
        explainer = shap.Explainer(clf, X_real)
        shap_values_real = explainer(X_real)
        shap_values_synthetic = explainer(X_synthetic)
        
        return shap_values_real, shap_values_synthetic

# Example usage
if __name__ == "__main__":
    # Sample data (replace with your dataset, e.g., OULAD)
    data = pd.DataFrame({
        'StudentID': range(1000),
        'Gender': np.random.choice(['Male', 'Female'], 1000),
        'Interactions': np.random.randint(100, 500, 1000),
        'AssessmentScore': np.random.uniform(50, 100, 1000),
        'Pass': np.random.choice([0, 1], 1000)
    })
    
    categorical_cols = ['Gender']
    numerical_cols = ['Interactions', 'AssessmentScore']
    sensitive_cols = ['Gender']
    target_col = 'Pass'
    
    # Initialize FairSYN-Edu
    input_dim = len(categorical_cols) + len(numerical_cols) + 1  # +1 for one-hot encoding
    sensitive_dim = 1  # Binary sensitive attribute (e.g., Gender)
    model = FairSYN_Edu(input_dim=input_dim, sensitive_dim=sensitive_dim)
    
    # Train model
    sensitive_data = data[sensitive_cols]
    model.train(data.drop(columns=['StudentID', target_col]), sensitive_data, 
                categorical_cols, numerical_cols, epochs=100, batch_size=128)
    
    # Generate synthetic data
    synthetic_data = model.generate_synthetic_data(num_samples=1000)
    
    # Evaluate SHAP
    shap_values_real, shap_values_synthetic = model.evaluate_shap(synthetic_data, data, target_col)
    
    # Save synthetic data
    synthetic_data.to_csv('synthetic_data.csv', index=False)
    logger.info("Synthetic data saved to 'synthetic_data.csv'")