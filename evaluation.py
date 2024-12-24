import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def set_seed(seed=42):
    """
    Set random seed for reproducibility.

    Args:
        seed (int): The seed value to set (default is 42).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate_model(model, test_loader, device, num_classes, class_to_idx):
    """
    Evaluate the model on the test dataset and display metrics in percentages.

    Args:
        model (torch.nn.Module): Trained model.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        device (torch.device): Device to perform computations on.
        num_classes (int): Number of classes in the dataset.
        class_to_idx (dict): Mapping of class names to indices.

    Returns:
        dict: Evaluation metrics including accuracy, classification report, and confusion matrix.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Get predictions
            logits = model.classify(inputs)
            preds = torch.argmax(logits, dim=1)

            # Collect predictions and labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute accuracy
    accuracy = accuracy_score(all_labels, all_preds) * 100
    print(f"Accuracy: {accuracy:.2f}%")

    # Generate classification report
    target_names = list(class_to_idx.keys())
    report = classification_report(
        all_labels,
        all_preds,
        target_names=target_names,
        zero_division=0,
        output_dict=True
    )
    print("\nClassification Report (in percentages):")
    for class_name, metrics in report.items():
        if class_name not in ["accuracy", "macro avg", "weighted avg"]:
            print(
                f"{class_name}: Precision: {metrics['precision'] * 100:.2f}%, Recall: {metrics['recall'] * 100:.2f}%, F1-Score: {metrics['f1-score'] * 100:.2f}%")
    print(f"\nMacro Avg F1-Score: {report['macro avg']['f1-score'] * 100:.2f}%")
    print(f"Weighted Avg F1-Score: {report['weighted avg']['f1-score'] * 100:.2f}%")

    # Generate confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    # Return metrics for further use
    return {
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": conf_matrix
    }