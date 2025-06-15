# 🧠 NeuroCare: CNN–LSTM for EEG Seizure Detection

> Hybrid deep learning architecture for real-time seizure detection from EEG signals using the BIDS-SEINA dataset.

---

## 📌 Overview

NeuroCare is a hybrid **1D CNN–BiLSTM** model designed to detect epileptic seizures from EEG signals. This model addresses the limitations of existing models in handling **temporal dynamics**, **class imbalance**, and **real-world noise** in EEG data.

---

## 🧪 Key Features

- Hybrid **CNN + BiLSTM** architecture
- **Focal loss** to improve sensitivity to seizures
- **Data augmentation** with sliding windows and synthetic generation
- Trained and evaluated on **BIDS-SEINA EEG dataset**
- Evaluation focused on **PR-AUC** and **F1-score** due to class imbalance
- Implemented in **TensorFlow 2.x**, optimized with **TFRecord**

---

## 🧰 Requirements

- Python 3.8+
- TensorFlow 2.12
- NumPy
- Pandas
- SciPy
- Matplotlib
- Seaborn
- Pyedflib
- Tqdm
- Scikit Learn
- Weights & Biases (optional)

---

## 🧬 Dataset

The BIDS-SEINA dataset includes:
- EEG recordings from 14 adult epilepsy patients (in .edf format with corresponding labels in .tsv files)
- 512 Hz sampling rate, 10–20 electrode placement
- Severe class imbalance (1 seizure to 59 background ratio)

> ⚠️ Dataset must be downloaded separately from its official repository.[ Click here to download BIDS-SEINA dataset.](https://paperswithcode.com/dataset/bids-siena-scalp-eeg-database)

---

## 🧼 Preprocessing Steps

1. **Butterworth bandpass filter** (0.5–30 Hz)
2. **Notch filter** for 50/60 Hz powerline noise
3. **Z-score normalization** per EEG channel
4. **Epoch segmentation** (1–10s, 50% overlap)
5. **TFRecord serialization** for TensorFlow training

---

## 🧠 Model Architecture

![architecture_cnnLstm (2)1](https://github.com/user-attachments/assets/1bf9a644-ff76-4578-8810-33872243a927)
```text
Input (Timestamps x Channels)
│
├── 1D Convolution Layers (Conv1D + BatchNorm + ReLU)
├── MaxPooling + Dropout
│
├── Bidirectional LSTM Layers
│
├── Fully Connected Layers
│
└── Output (Softmax: Seizure / Non-Seizure)
```
![methodology2 drawio (3)](https://github.com/user-attachments/assets/9a8d2249-141c-4224-9b51-ffe542f53d9b)

## 📈 Results Summary

The model was evaluated on the BIDS-SEINA EEG dataset, which has a severe class imbalance (1 seizure to 59 background segments). Below is a summary of the model’s performance:

| **Metric**      | **Value** |
|------------------|-----------|
| Accuracy         | 88.94%    |
| Seizure F1 Score | 0.13      |
| ROC-AUC          | 0.686     |
| PR-AUC           | 0.279     |

### 🔬 Key Observations

- Removing **LSTM layers** led to a **−38% drop in seizure F1 score**
- Replacing **cross-entropy** with **focal loss** improved seizure F1 by **+38%**
- Removing **data augmentation** dropped F1 by **−23%**

> ⚠️ Note: High accuracy may be misleading in imbalanced datasets. PR-AUC and F1-score are more informative for seizure detection performance.


