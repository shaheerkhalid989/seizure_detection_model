import json
import numpy as np
from pathlib import Path
from config.settings import Config
from typing import Dict, List
import json
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    auc
)
from sklearn.utils.multiclass import unique_labels

class ReportGenerator:
    def __init__(self, config: Config):
        self.config = config
        self.class_mapping = config.data.event_mapping
        self.inverse_mapping = config.data.inverse_mapping
        self.report_dir = config.paths.output_dir 
        # / "detailed_reports"
        self.report_dir.mkdir(parents=True, exist_ok=True)
        
    def save_report(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Save comprehensive evaluation reports with visualizations"""
        try:
            # Convert to numpy arrays if they're tensors
            y_true = y_true.numpy() if isinstance(y_true, tf.Tensor) else y_true
            y_pred = y_pred.numpy() if isinstance(y_pred, tf.Tensor) else y_pred
            
            
            # Get class names from config
            classes = unique_labels(y_true, y_pred)
            class_names = [self.inverse_mapping[c] for c in classes]
            
            # 1. Classification Report
            report = classification_report(
                y_true, 
                y_pred,
                labels=classes,
                target_names=class_names,
                zero_division=0,  # Handle undefined metrics
                output_dict=True
            )
            
            # Save text report
            with open(self.config.paths.output_dir / "classification_report.txt", "w") as f:
                f.write(classification_report(y_true, y_pred, target_names=class_names))
            
            # 2. Confusion Matrix
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=class_names, yticklabels=class_names)
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title("Confusion Matrix")
            plt.savefig(self.config.paths.output_dir / "confusion_matrix.png")
            plt.close()
            
            # 3. ROC-AUC Score (for binary and multiclass)
            if len(class_names) == 2:  # Binary classification
                roc_auc = roc_auc_score(y_true, y_pred)
            else:  # Multiclass
                roc_auc = roc_auc_score(y_true, y_pred, multi_class="ovr")
            
            # 4. Precision-Recall Curve
            precision, recall, _ = precision_recall_curve(y_true, y_pred)
            pr_auc = auc(recall, precision)
            
            # 5. Save metrics to JSON
            metrics = {
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "classification_report": report,
                "confusion_matrix": cm.tolist()
            }
            
            with open(self.config.paths.output_dir / "metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)
            
            print(f"Reports saved to {self.config.paths.output_dir}")
            
        except Exception as e:
            print(f"Error generating reports: {str(e)}")
            raise
            
    def save_events(self, events: list, filename: str):
        """Save detected events to JSON"""
        path = self.config.paths.output_dir / filename
        with path.open('w') as f:
            json.dump(events, f, indent=2)