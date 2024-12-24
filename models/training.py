import torch
import torch.nn.functional as F


class TripletLoss(torch.nn.Module):
    """
    Triplet Loss module with margin.

    Args:
        margin (float): Margin for the triplet loss (default is 1.0).
    """

    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        """
        Compute the triplet loss.

        Args:
            anchor (torch.Tensor): Anchor embeddings.
            positive (torch.Tensor): Positive embeddings.
            negative (torch.Tensor): Negative embeddings.

        Returns:
            torch.Tensor: Triplet loss value.
        """
        pos_distance = F.pairwise_distance(anchor, positive, p=2)
        neg_distance = F.pairwise_distance(anchor, negative, p=2)
        loss = F.relu(pos_distance - neg_distance + self.margin)
        return loss.mean()


def train_siamese_with_triplet_loss(model, train_loader, optimizer, criterion, device):
    """
    Train the Siamese network using triplet loss.

    Args:
        model (torch.nn.Module): Siamese network model.
        train_loader (torch.utils.data.DataLoader): DataLoader for triplet training data.
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        criterion (torch.nn.Module): Triplet loss criterion.
        device (torch.device): Device to train the model on.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    total_loss = 0

    for batch_idx, (anchor, positive, negative, _) in enumerate(train_loader):
        # Move data to the appropriate device
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

        # Forward pass
        anchor_embed, positive_embed, negative_embed = model(anchor, positive, negative)

        # Compute triplet loss
        loss = criterion(anchor_embed, positive_embed, negative_embed)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def fine_tune_for_classification(model, train_loader, optimizer, criterion, device):
    """
    Fine-tune the Siamese network for classification.

    Args:
        model (torch.nn.Module): Siamese network model.
        train_loader (torch.utils.data.DataLoader): DataLoader for classification training data.
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        criterion (torch.nn.Module): Classification loss criterion (e.g., CrossEntropyLoss).
        device (torch.device): Device to train the model on.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    total_loss = 0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        # Move data to the appropriate device
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass for classification
        logits = model.classify(inputs)
        loss = criterion(logits, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)