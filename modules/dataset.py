import torch
from torch.utils.data import Dataset

class TripletDataset(Dataset):
    """
    Dataset class for triplet data (anchor, positive, negative).

    Args:
        triplets (list of tuples): Each tuple contains (anchor, positive, negative).
        labels (list): Class labels corresponding to each triplet.
    """
    def __init__(self, triplets, labels):
        self.triplets = triplets
        self.labels = labels

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor, positive, negative = self.triplets[idx]
        label = self.labels[idx]

        # Convert to PyTorch tensors
        anchor = torch.tensor(anchor, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        positive = torch.tensor(positive, dtype=torch.float32).unsqueeze(0)
        negative = torch.tensor(negative, dtype=torch.float32).unsqueeze(0)
        return anchor, positive, negative, label


class ClassificationDataset(Dataset):
    """
    Dataset class for classification data.

    Args:
        data (numpy.ndarray): Input data array (samples x features).
        labels (numpy.ndarray): Corresponding labels for each sample.
    """
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_data = torch.tensor(self.data[idx], dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return input_data, label