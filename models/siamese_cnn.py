import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseCNN(nn.Module):
    """
    Siamese CNN model for triplet loss and classification.

    Args:
        num_classes (int): Number of output classes for classification.
        embedding_dim (int): Dimension of the embedding space.
    """
    def __init__(self, num_classes=11, embedding_dim=128):
        super(SiameseCNN, self).__init__()

        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )

        # Embedding layer
        self.embedding_layer = nn.Sequential(
            nn.Linear(128, embedding_dim),
            nn.Dropout(0.5)  # Dropout for regularization
        )

        # Classification head
        self.classification_head = nn.Linear(embedding_dim, num_classes)

    def forward_once(self, x):
        """
        Forward pass for one branch of the Siamese network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, sequence_length).

        Returns:
            torch.Tensor: Normalized embedding of shape (batch_size, embedding_dim).
        """
        features = self.feature_extractor(x)
        embedding = self.embedding_layer(features)
        embedding = F.normalize(embedding, p=2, dim=1)  # L2 normalize embeddings
        return embedding

    def forward(self, anchor, positive, negative):
        """
        Forward pass for triplet input (anchor, positive, negative).

        Args:
            anchor (torch.Tensor): Anchor input tensor.
            positive (torch.Tensor): Positive input tensor.
            negative (torch.Tensor): Negative input tensor.

        Returns:
            tuple: Triplet embeddings (anchor, positive, negative).
        """
        anchor_embedding = self.forward_once(anchor)
        positive_embedding = self.forward_once(positive)
        negative_embedding = self.forward_once(negative)
        return anchor_embedding, positive_embedding, negative_embedding

    def classify(self, x):
        """
        Classification forward pass.

        Args:
            x (torch.Tensor): Input tensor for classification.

        Returns:
            torch.Tensor: Logits for classification.
        """
        embedding = self.forward_once(x)
        logits = self.classification_head(embedding)
        return logits