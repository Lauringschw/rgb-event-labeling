import numpy as np
from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv(Path(__file__).parent.parent / '.env')
base = Path(os.getenv("RESULTS_DIR"))

num = 3 # 1, 2, 3
metadata = np.load(base / f"rq{num}_results.npy", allow_pickle=True).item()

def classification_report_from_cm(cm):
    """Compute classification report from confusion matrix."""
    n_classes = cm.shape[0]
    precision = np.zeros(n_classes)
    recall = np.zeros(n_classes)
    f1_score = np.zeros(n_classes)
    support = cm.sum(axis=1)
    
    for i in range(n_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - (tp + fp + fn)
        
        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0
    
    # Macro averages
    macro_precision = precision.mean()
    macro_recall = recall.mean()
    macro_f1 = f1_score.mean()
    
    # Weighted averages
    weighted_precision = (precision * support).sum() / support.sum()
    weighted_recall = (recall * support).sum() / support.sum()
    weighted_f1 = (f1_score * support).sum() / support.sum()
    
    return precision, recall, f1_score, support, macro_precision, macro_recall, macro_f1, weighted_precision, weighted_recall, weighted_f1

for window, results in metadata.items():
    print(f"\nWindow: {window}")
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"Validation Accuracy: {results['val_accuracy']:.4f}")
    print("Confusion Matrix:")
    print(results['confusion_matrix'])
    
    # Compute classification report
    cm = results['confusion_matrix']
    precision, recall, f1_score, support, macro_p, macro_r, macro_f1, weighted_p, weighted_r, weighted_f1 = classification_report_from_cm(cm)
    
    print("\nClassification Report:")
    print("              precision    recall  f1-score   support")
    classes = ["rock", "paper", "scissors"]
    for i in range(len(precision)):
        print(f"    {classes[i]}    {precision[i]:.4f}    {recall[i]:.4f}    {f1_score[i]:.4f}    {support[i]}")
    print(f"   macro avg    {macro_p:.4f}    {macro_r:.4f}    {macro_f1:.4f}    {support.sum()}")
    print(f"weighted avg    {weighted_p:.4f}    {weighted_r:.4f}    {weighted_f1:.4f}    {support.sum()}")

