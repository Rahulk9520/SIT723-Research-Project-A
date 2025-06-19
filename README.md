# Transformers for COVID-19 Detection Using Chest X-ray Images

**SIT723 Research Project-A**  
*Deakin University*  
*Rahul Kumar, Master of Applied Artificial Intelligence (Professional)*

---

## Table of Contents

- [Project Overview](#project-overview)
- [Motivation](#motivation)
- [Research Objectives](#research-objectives)
- [Methodology](#methodology)
- [Data Sources](#data-sources)
- [Installation & Requirements](#installation--requirements)
- [Usage](#usage)
- [Results & Evaluation](#results--evaluation)
- [Directory Structure](#directory-structure)
- [Reproducibility](#reproducibility)
- [Limitations & Future Work](#limitations--future-work)
- [Acknowledgements](#acknowledgements)
- [Citations](#citations)
- [Contact](#contact)

---

## Project Overview

This repository contains the code, data processing scripts, and documentation for the research project: **"Transformers for COVID-19 detection for the multiclass classification problem utilizing X-ray images."** The project explores advanced deep learning models, specifically Vision Transformers (ViT) and their convolutional variants (CvT), for accurate and efficient classification of COVID-19, Pneumonia, and Normal cases from chest X-ray images.

---

## Motivation

The COVID-19 pandemic has strained healthcare systems worldwide, emphasizing the need for rapid, accurate, and accessible diagnostic tools. While RT-PCR remains the gold standard, it is resource-intensive and time-consuming. Chest X-rays (CXR) are widely available and can provide visual evidence for diagnosis. Recent advances in deep learning, especially transformers, offer the potential to improve diagnostic accuracy and speed, overcoming limitations of traditional CNNs in capturing global image context.

---

## Research Objectives

- **Develop** an advanced deep learning model using Transformers for multiclass classification (COVID-19, Pneumonia, Normal) from CXR images.
- **Investigate** architectural innovations (e.g., convolutional token embedding, convolutional projection) to enhance model performance and efficiency.
- **Benchmark** the proposed models against state-of-the-art CNN and transformer-based approaches.
- **Evaluate** model explainability and robustness using appropriate metrics and visualization techniques.
- **Facilitate** reproducibility and future research by providing clear documentation and code.

---

## Methodology

1. **Literature Review:**  
   Surveyed recent advances in deep learning for medical imaging, focusing on transformer architectures and their application to COVID-19 detection.

2. **Model Architecture:**  
   - Implemented Convolutional Vision Transformers (CvT) that combine convolutional layers with transformer blocks to capture both local and global features.
   - Compared with standard CNNs and other transformer variants (ViT, DeiT, Swin Transformer, etc.).

3. **Data Preparation:**  
   - Utilized publicly available datasets (see [Data Sources](#data-sources)).
   - Performed data cleaning, normalization, augmentation (resizing, flipping, rotation, etc.), and split into training/validation/test sets.

4. **Training & Optimization:**  
   - Models trained using PyTorch with AdamW optimizer and cosine learning rate scheduling.
   - Hyperparameter tuning (batch size, learning rate, epochs, etc.) and regularization strategies applied.

5. **Evaluation:**  
   - Metrics: Accuracy, Precision, Recall, F1-score, Specificity, MCC, AUC.
   - Explainability: Attention maps, Grad-CAM visualizations.

6. **Comparison:**  
   - Benchmarked against published results and state-of-the-art methods.

---

## Data Sources

- **COVID-19 Radiography Database** ([Kaggle](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database))
  - 1200 COVID-19 positive, 1341 Normal, 1345 Viral Pneumonia images.
- **Other Datasets Used for Benchmarking:**
  - COVIDx CXR-2
  - CheXpert
  - SIIM-FISABIO-RSNA COVID-19
  - Additional datasets as referenced in the thesis.

---

## Installation & Requirements

### Prerequisites:
- Python 3.7+
- PyTorch
- scikit-learn
- pandas, numpy
- matplotlib, seaborn (for visualization)
- tensorboardX (for logging)
- Jupyter Notebook (recommended for experimentation)

### Installation:

**Clone the repository:**
- `git clone https://github.com/Rahulk9520/CvT-main.git`
- `cd CvT-main`

**Install dependencies:**
- `pip install -r requirements.txt`

*For Google Cloud Platform (GCP) usage, ensure you have access and the appropriate SDKs installed.*

---

## Usage

**Data Preparation:**
- Download the datasets as described in [Data Sources](#data-sources).
- Organize data into `train`, `validation`, and `test` folders as per the provided scripts.

**Model Training:**
`python train.py --config configs/cvt_config.yaml`

**Evaluation:**
`python evaluate.py --model_path checkpoints/best_model.pth`

**Visualization:**
- Use Jupyter notebooks in the `notebooks/` directory for data exploration and attention map visualization.

---

## Results & Evaluation

- **Best Model:** CvT-W24 (Wide) achieved an overall accuracy of **84.12%** on the COVID-19 Radiography Database.
- **Performance Metrics:**  
  - Accuracy, Precision, Recall, F1-score, Specificity, MCC, AUC (see `results/` for detailed tables and plots).
- **Explainability:**  
  - Attention maps and Grad-CAM visualizations indicate that the model focuses on clinically relevant regions in the X-rays.
- **Comparison:**  
  - Outperformed several existing CNN and transformer-based models on the same datasets.

---

## Directory Structure
```
CvT-main/
│
├── data/ # Data folders and download scripts
├── notebooks/ # Jupyter notebooks for EDA and visualization
├── src/ # Source code for models, training, evaluation
├── configs/ # YAML configuration files
├── results/ # Output metrics, plots, logs
├── requirements.txt # Python dependencies
├── README.md # This file
└── LICENSE

```

## Reproducibility

- All experiments are logged with configuration files and random seeds for reproducibility.
- Detailed instructions for reproducing results are provided in the `notebooks/` and `src/` directories.
- For GCP-based training, see `cloud/README.md`.

---

## Limitations & Future Work

- **Dataset Size:** Current models are trained and validated on limited datasets; larger, more diverse datasets may further improve generalizability.
- **Clinical Validation:** Further testing in real-world clinical settings is required.
- **Model Extensions:** Future work includes integrating neural architecture search (NAS) for model optimization and expanding to other imaging modalities (CT, MRI).

---

## Acknowledgements

- Supervisor: **Prof. Bahareh Nakisa**
- Unit Chair: **Prof. Yong Xiang**
- Thesis Advisors: Dr. Asef Nazari, Dr. Duc Thanh Nguyen, Dr. Chandan Karmakar, Dr. Wei Luo
- [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) (Kaggle)
- Open-source contributors and the Deakin University online library

---

## Citations

If you use this code or data, please cite:

Rahul Kumar. (2022). Transformers for COVID-19 detection for the multiclass classification problem utilizing X-ray images. SIT723 Research Project-A, Deakin University.


See also the [thesis PDF](Rahul-Thesis_merged.pdf) for a full bibliography.

---

## Contact

**Author:** Rahul Kumar  
**Email:** kumar.rahul226@gmail.com  
**LinkedIn:** [rk95-dataquasar](https://www.linkedin.com/in/rk95-dataquasar/)

---


