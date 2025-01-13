import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe
        self.texts = dataframe['text'].tolist()
        self.labels = dataframe['labels'].tolist()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        return text, torch.tensor(label, dtype=torch.long)