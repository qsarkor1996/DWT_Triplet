import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from config import (
    DATA_PATH, PROCESSED_DATA_DIR, DWT_DATA_DIR, TRIPLETS_DIR, VISUALIZATION_DIR,
    SNR_DB, DWT_WAVELET, DWT_LEVEL, NUM_TRIPLETS, EMBEDDING_DIM, NUM_CLASSES,
    MARGIN, LEARNING_RATE, BATCH_SIZE, NUM_EPOCHS_TRIPLET, NUM_EPOCHS_CLASSIFICATION
)
from modules.utils import load_mat_data, save_data, min_max_scaling, split_data, save_class_data
from modules.awgn import add_awgn_to_training_data
from modules.dwt import apply_dwt_to_noisy_training_data
from modules.visualization import (
    visualize_tsne_embeddings,
    plot_sample_signals,
    plot_confusion_matrix,
    visualize_dwt_signals
)
from modules.dataset import TripletDataset, ClassificationDataset
from models.siamese_cnn import SiameseCNN
from models.training import train_siamese_with_triplet_loss, fine_tune_for_classification, TripletLoss
from evaluation import evaluate_model, set_seed


def generate_classification_data(data_dict):
    """
    Prepare classification data and labels from the training data.

    Args:
        data_dict (dict): Dictionary with class names as keys and samples as values.

    Returns:
        tuple: (classification_data, classification_labels, class_to_idx)
    """
    classification_data = []
    classification_labels = []
    class_to_idx = {class_name: idx for idx, class_name in enumerate(data_dict.keys())}

    for class_name, samples in data_dict.items():
        for sample in samples:
            classification_data.append(sample)
            classification_labels.append(class_to_idx[class_name])

    return np.array(classification_data), np.array(classification_labels), class_to_idx


def main():
    # Ensure reproducibility
    set_seed(42)

    # Directory for saving results
    RESULTS_DIR = os.path.join(VISUALIZATION_DIR, "results")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Device Configuration
    print("Configuring device...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Step 1: Load Data
    print("Loading data...")
    mat_data = load_mat_data(DATA_PATH)
    my_data = mat_data["myData"]  # Adjust this based on your .mat file structure
    data = np.abs(my_data["LTF"][0, 0])  # Example structure
    labels = my_data["label"][0, 0]

    # Step 2: Normalize and Split Data
    print("Normalizing data...")
    data = min_max_scaling(data)
    slices = {
        'txa1a': (0, 350),
        'txb2a': (351, 701),
        'txc3a': (773, 1123),
        'txd4a': (1233, 1583),
        'txe5a': (1622, 1972),
        'txf6a': (2024, 2374),
        'txg7a': (2477, 2827),
        'txh8a': (2915, 3265),
        'txi9a': (3365, 3715),
        'txj10a': (3760, 4110),
        'txk11a': (4224, 4574),
    }
    class_data = split_data(data, slices)
    train_data, test_data = {}, {}
    for class_name, samples in class_data.items():
        train, test = train_test_split(samples, test_size=0.2, random_state=42)
        train_data[class_name] = train
        test_data[class_name] = test

    # Step 3: Add AWGN
    print("Adding AWGN...")
    noisy_train_data = add_awgn_to_training_data(train_data, SNR_DB)

    # Step 4: Apply DWT
    print("Applying DWT...")
    dwt_data = apply_dwt_to_noisy_training_data(noisy_train_data, wavelet=DWT_WAVELET, level=DWT_LEVEL)

    # Visualize DWT Signals
    print("Visualizing DWT applied signals for a single sample...")
    signal = noisy_train_data['txa1a'][0]  # Visualize the first signal from 'txa1a'
    visualize_dwt_signals(signal, wavelet=DWT_WAVELET, output_dir=RESULTS_DIR, signal_idx=0)

    # Step 5: Create Triplets
    print("Creating triplets...")
    triplets, triplet_labels = [], []
    class_names = list(dwt_data.keys())

    for _ in range(NUM_TRIPLETS):
        anchor_class = np.random.choice(class_names)
        positive_class = anchor_class
        negative_class = np.random.choice([cls for cls in class_names if cls != anchor_class])

        anchor_idx, positive_idx = np.random.choice(len(dwt_data[anchor_class]), size=2, replace=False)
        negative_idx = np.random.choice(len(dwt_data[negative_class]))

        triplets.append((
            dwt_data[anchor_class][anchor_idx][0],  # Anchor
            dwt_data[positive_class][positive_idx][0],  # Positive
            dwt_data[negative_class][negative_idx][0],  # Negative
        ))
        triplet_labels.append(anchor_class)

    # Save Triplets
    save_data(triplets, os.path.join(TRIPLETS_DIR, "triplets.npy"))
    save_data(triplet_labels, os.path.join(TRIPLETS_DIR, "labels.npy"))

    # Step 6: Prepare Datasets and Dataloaders
    print("Preparing datasets...")
    triplet_dataset = TripletDataset(triplets, triplet_labels)
    triplet_loader = DataLoader(triplet_dataset, batch_size=BATCH_SIZE, shuffle=True)

    classification_data, classification_labels, class_to_idx = generate_classification_data(train_data)
    classification_dataset = ClassificationDataset(classification_data, classification_labels)
    classification_loader = DataLoader(classification_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_data, test_labels, _ = generate_classification_data(test_data)
    test_dataset = ClassificationDataset(test_data, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Step 7: Initialize Model
    print("Initializing model...")
    model = SiameseCNN(num_classes=NUM_CLASSES, embedding_dim=EMBEDDING_DIM).to(device)
    triplet_criterion = TripletLoss(margin=MARGIN)
    classification_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Step 8: Train with Triplet Loss
    print("Training with Triplet Loss...")
    for epoch in range(NUM_EPOCHS_TRIPLET):
        triplet_loss = train_siamese_with_triplet_loss(
            model, triplet_loader, optimizer, triplet_criterion, device
        )
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS_TRIPLET}, Triplet Loss: {triplet_loss:.4f}")

    # Step 9: Fine-tune for Classification
    print("Fine-tuning for classification...")
    for epoch in range(NUM_EPOCHS_CLASSIFICATION):
        classification_loss = fine_tune_for_classification(
            model, classification_loader, optimizer, classification_criterion, device
        )
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS_CLASSIFICATION}, Classification Loss: {classification_loss:.4f}")

    # Step 10: Evaluate Model
    print("Evaluating the model...")
    metrics = evaluate_model(model, test_loader, device, NUM_CLASSES, class_to_idx)

    # Save confusion matrix visualization
    print("Saving confusion matrix...")
    plot_confusion_matrix(metrics["confusion_matrix"], list(class_to_idx.keys()), RESULTS_DIR)

    # Step 11: Visualize t-SNE
    print("Visualizing embeddings...")
    visualize_tsne_embeddings(model, test_loader, device, class_to_idx, RESULTS_DIR)

    # Step 12: Visualize sample signals
    print("Visualizing sample signals...")
    plot_sample_signals(data, indices=[0, 10, 20], output_dir=RESULTS_DIR)


if __name__ == "__main__":
    main()