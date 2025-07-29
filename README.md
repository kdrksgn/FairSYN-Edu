FairSYN-Edu: A Fairness-Aware and Privacy-Preserving Synthetic Data Generator for Educational Research
Overview
FairSYN-Edu is a diffusion-based synthetic data generation framework designed for educational data, integrating adversarial debiasing, differential privacy (DP-SGD), and SHAP-based interpretability. It optimizes for utility, fairness, and privacy, suitable for high-stakes educational settings.
Installation

Clone the repository:git clone https://github.com/kdrrkagn/fairsynedu.git
cd fairsynedu


Install dependencies:pip install -r requirements.txt



Requirements
See requirements.txt for a complete list of dependencies. Key packages include:

PyTorch 2.0
Opacus
SHAP
Pandas
Scikit-learn

Usage

Prepare your dataset in a tabular format (e.g., CSV) with categorical and numerical features.
Run the main script:python fairSYN_edu.py


The script will train the model, generate synthetic data, and save it to synthetic_data.csv.

Example
import pandas as pd
from fairSYN_edu import FairSYN_Edu

# Load your dataset
data = pd.read_csv('your_dataset.csv')
categorical_cols = ['Gender']
numerical_cols = ['Interactions', 'AssessmentScore']
sensitive_cols = ['Gender']
target_col = 'Pass'

# Initialize and train model
model = FairSYN_Edu(input_dim=len(categorical_cols) + len(numerical_cols) + 1, sensitive_dim=1)
model.train(data.drop(columns=['StudentID', target_col]), data[sensitive_cols], 
            categorical_cols, numerical_cols, epochs=100, batch_size=128)

# Generate synthetic data
synthetic_data = model.generate_synthetic_data(num_samples=1000)
synthetic_data.to_csv('synthetic_data.csv', index=False)

Reproducibility

The code is tested on a workstation with an NVIDIA RTX 3090 GPU (24GB VRAM) and 256GB RAM.
Training takes approximately 1.6 hours per dataset for 100 epochs with batch size 128.
Synthetic data generation is near real-time (<1 sec/sample).

License
This project is licensed under the MIT License.
Citation
If you use FairSYN-Edu, please cite:
Kesgin, K. (2025). FairSYN-Edu: A Fairness-Aware and Privacy-Preserving Synthetic Data Generator for Educational Research.

Contact
For questions, contact Kadir Kesgin at kadir@bandirma.edu.tr.
