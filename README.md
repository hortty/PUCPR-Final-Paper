# TCC: Advanced Framework for Facial Emotion Recognition

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)

This repository contains a robust and modular framework for training, evaluating, and comparing deep learning models for Facial Emotion Recognition (FER). The architecture is designed for experimentation, allowing for easy integration of new datasets and models, and provides powerful features like cross-dataset evaluation, ensemble methods, and comprehensive error analysis.

## ✨ Key Features

### 🎯 Training Strategies
- **Individual Training:** Train separate models for each dataset
- **Merged Training:** Train models on combined datasets
- **Hybrid Mode:** Execute both strategies and compare results

### 📊 Multi-Dataset Support
- Currently supports **FER2013**, **RAF-DB**, and **ExpW**
- Easy integration of new datasets via `data_loader.py`
- Automatic data balancing with Random Oversampling

### 🧠 Multiple Model Architectures
- Pre-configured for **DenseNet121**, **ResNet50**, and **EfficientNet-B0**
- Uses `timm` and `torchvision` for pre-trained weights
- Simple configuration to add new architectures

### 🚀 Intelligent Training Pipeline
- **Pre-trained Model Detection:** Automatically skips training if weights exist
- **Early Stopping:** Prevents overfitting and saves training time
- **Automatic Memory Management:** GPU cache clearing and garbage collection
- **Batch Size Optimization:** Per-model configuration for optimal GPU usage

### 📈 Advanced Evaluation
- **Test-Time Augmentation (TTA):** Improves prediction accuracy
- **Cross-Dataset Evaluation:** Tests generalization across different domains
- **Ensemble Methods:** Combines multiple models (soft/hard voting)
- **Normalized Confusion Matrices:** Shows proportions (0-1) for easy comparison

### 🔍 Comprehensive Error Analysis
- **Per-Class Error Rates:** Visual breakdown with color coding
- **Top Confused Pairs:** Identifies critical misclassifications
- **Error Distribution Matrix:** Heatmap of confusion patterns
- **Statistical Summary:** Best/worst classes and critical confusions

### 📁 Organized Output Structure
- **Automatic PDF Generation:** High-quality plots ready for academic papers
- **Structured Directories:** Separate folders for each output type
- **CSV Export:** Results table for easy integration with LaTeX/Excel
- **Timestamped Logs:** Detailed execution history

## 🗂️ Project Structure

```
.
├── main.py                 # Main execution script - unified pipeline
├── config.py               # Central configuration (models, datasets, hyperparameters)
├── data_loader.py          # Dataset loading, splitting, and balancing
├── model_utils.py          # Model creation and ensemble logic
├── training.py             # Training and evaluation loops
├── utils.py                # Plotting, logging, and error analysis utilities
│
├── fer2013/                # FER2013 dataset directory
├── rafdb/                  # RAF-DB dataset directory
├── expw/                   # Expression in-the-Wild dataset directory
│
└── saidas/                 # Output directory (auto-generated)
    ├── execution_log_*.txt
    ├── comprehensive_results.csv
    ├── saved_models/
    ├── class_distributions/
    ├── confusion_matrices/
    │   └── TTA/
    ├── error_analysis/
    ├── cross_dataset/
    └── ensemble_results/
```

## 📦 Setup and Installation

### 1. Clone the Repository

```bash
git clone https://github.com/MatheusVMariussi/FER-Framework.git
cd TCC-FER
```

### 2. Install Dependencies

Create a `requirements.txt` file with the following content:

```txt
torch
torchvision
pandas
scikit-learn
matplotlib
seaborn
Pillow
timm
imbalanced-learn
numpy
```

Then install:

```bash
pip install -r requirements.txt
```

### 3. Dataset Structure

This framework expects the following directory structure:

#### **FER2013:**
```
./fer2013/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── sad/
│   ├── surprise/
│   └── neutral/
└── test/
    ├── angry/
    └── ...
```

#### **RAF-DB:**
```
./rafdb/DATASET/
├── train/
│   ├── 1/  (surprise)
│   ├── 2/  (fear)
│   ├── 3/  (disgust)
│   ├── 4/  (happy)
│   ├── 5/  (sad)
│   ├── 6/  (angry)
│   └── 7/  (neutral)
└── test/
    └── ...
```

#### **Expression in-the-Wild (ExpW):**
```
./expw/Expw-F/
├── angry/
├── disgust/
├── fear/
├── happy/
├── sad/
├── surprise/
└── neutral/
```

## 🚀 How to Use

### 1. Configure Your Experiment

Open `config.py` and customize your settings:

#### **Choose Training Strategy:**
```python
# Train only individual models (one per dataset)
TRAINING_STRATEGY = 'individual'

# Train only merged model (all datasets combined)
TRAINING_STRATEGY = 'merged'

# Train both and compare (RECOMMENDED)
TRAINING_STRATEGY = 'both'
```

#### **Select Active Datasets:**
```python
ACTIVE_DATASETS = {
    'RAF-DB': 'load_rafdb',
    'ExpW': 'load_expw',
    'FER2013': 'load_fer2013',  # Comment out to deactivate
}
```

#### **Configure Models:**
```python
MODEL_CONFIG = {
    'densenet121': {'batch_size': 96},
    'resnet50': {'batch_size': 128},
    'efficientnet_b0': {'batch_size': 128},
    # Add more models here
}
```

#### **Output Settings:**
```python
PLOT_FORMAT = 'pdf'      # 'pdf' or 'png'
NORMALIZE_CM = True      # Normalize confusion matrices (0-1)
```

#### **Hyperparameters:**
```python
EPOCHS = 100             # Max epochs (early stopping may stop sooner)
LEARNING_RATE = 0.001
PATIENCE = 5             # Early stopping patience
```

### 2. Run the Pipeline

```bash
python main.py
```

The script will automatically:
- ✅ Load and balance datasets
- ✅ Train models (or load existing weights)
- ✅ Evaluate with and without TTA
- ✅ Generate confusion matrices
- ✅ Perform error analysis
- ✅ Cross-dataset evaluation
- ✅ Ensemble evaluation
- ✅ Export comprehensive results table

### 3. Review the Outputs

All results are saved to `saidas/` with the following structure:

```
saidas/
├── execution_log_20241025_143022.txt       # Detailed log
├── comprehensive_results.csv               # Main results table
│
├── saved_models/                           # Trained model weights
│   ├── RAF-DB_densenet121.pth
│   ├── merged_densenet121.pth
│   └── ...
│
├── class_distributions/                    # Dataset distributions
│   ├── RAF-DB_dist_original.pdf
│   ├── RAF-DB_dist_balanced.pdf
│   └── merged_train_distribution.pdf
│
├── confusion_matrices/
│   └── TTA/                                # All TTA confusion matrices
│       ├── RAF-DB_densenet121_cm_TTA.pdf
│       ├── merged_densenet121_RAF-DB_cm_TTA.pdf
│       └── ...
│
├── error_analysis/                         # Detailed error analysis
│   ├── RAF-DB_densenet121_error_analysis.pdf
│   ├── merged_densenet121_RAF-DB_error_analysis.pdf
│   └── ...
│
├── cross_dataset/                          # Generalization tests
│   ├── cross_dataset_individual.pdf
│   └── cross_dataset_merged.pdf
│
└── ensemble_results/                       # Ensemble evaluations
    ├── ensemble_individual_RAF-DB_cm.pdf
    ├── ensemble_individual_RAF-DB_error.pdf
    ├── ensemble_merged_RAF-DB_cm.pdf
    └── ensemble_merged_RAF-DB_error.pdf
```

## 📊 Understanding the Outputs

### **Confusion Matrix (Normalized)**
Shows prediction accuracy as proportions (0-1):
- **Diagonal:** Correct predictions (higher = better)
- **Off-diagonal:** Confusions between classes
- Values sum to 1.0 per row (true label)

### **Error Analysis Plot**
Four-panel visualization:
1. **Top:** Per-class error rates (color-coded by severity)
2. **Middle:** Top 10 most confused class pairs
3. **Bottom-Left:** Error distribution heatmap
4. **Bottom-Right:** Statistical summary

### **Cross-Dataset Evaluation**
Heatmap showing model generalization:
- **Rows:** Training dataset
- **Columns:** Test dataset
- **Cells:** Accuracy (darker = better)
- **Blue boxes:** Same dataset (expected high accuracy)

### **Comprehensive Results Table (CSV)**
Complete results in tabular format:
- Section 1: Individual model accuracies
- Section 2: Merged model accuracies
- Section 3: Individual ensemble results
- Section 4: Merged ensemble results

 **⭐ If this framework helped your research, please consider starring the repository!**
