import torch
from tqdm import tqdm
import random
import matplotlib.pyplot as plt

def train_one_epoch(model, tokenizer, data_loader, optim, lr_scheduler, device):
    """
    Train the model for one epoch.
    """
    # track results
    all_true_labels = []
    all_predicted_labels = []
    cumulative_loss = 0

    # Set the model to training mode
    model.train()

    # Loop through each batch in DataLoader
    for text_batch, label_batch in tqdm(data_loader, total=len(data_loader), desc="Batch Training Progress"):
        # Store true labels for evaluation
        all_true_labels += label_batch.flatten().tolist()

        # Tokenize text and move to device
        encoded_text = tokenizer(text_batch, padding=True, truncation=True, return_tensors="pt").to(device)
        label_batch = label_batch.to(device)

        # set gradients to zero before doing backprop
        model.zero_grad()

        # Forward pass
        output = model(**encoded_text, labels=label_batch)
        batch_loss = output.loss
        logits = output.logits
        cumulative_loss += batch_loss.item()

        # backprop
        batch_loss.backward()
        # update weights
        optim.step()
        # update learning rate
        lr_scheduler.step()

        # store predictions
        logits = logits.detach().cpu().numpy()
        all_predicted_labels += logits.argmax(axis=-1).flatten().tolist()

    # Compute average loss for the epoch
    epoch_loss = cumulative_loss / len(data_loader)

    return all_true_labels, all_predicted_labels, epoch_loss


def validate_one_epoch(model, tokenizer, data_loader, device):
    """
    Validate the model for one epoch.
    """
    # Tracking results
    all_true_labels = []
    all_predicted_labels = []
    cumulative_loss = 0

    # Set the model to evaluation mode
    model.eval()

    # Loop through each batch in the DataLoader
    for text_batch, label_batch in tqdm(data_loader, total=len(data_loader), desc="Batch Validating Progress"):
        # Store true labels for evaluation
        all_true_labels += label_batch.flatten().tolist()

        # Tokenize text and move to device
        encoded_text = tokenizer(text_batch, padding=True, truncation=True, return_tensors="pt").to(device)
        label_batch = label_batch.to(device)

        # Disable gradient calculation for validation
        with torch.no_grad():
            output = model(**encoded_text, labels=label_batch)
            batch_loss = output.loss
            logits = output.logits
            cumulative_loss += batch_loss.item()

            # Store predictions
            logits = logits.detach().cpu().numpy()
            predictions = logits.argmax(axis=-1).flatten().tolist()
            all_predicted_labels += predictions

    # Compute average loss for the epoch
    epoch_loss = cumulative_loss / len(data_loader)

    return all_true_labels, all_predicted_labels, epoch_loss
