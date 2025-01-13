import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from finetune_utils import train_one_epoch, validate_one_epoch, plot_attention
from transformers import AdamW, get_linear_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup, get_constant_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
from transformers import GPT2ForSequenceClassification, GPT2Config, GPT2Tokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from TextDataset import TextDataset
import os
import sys
from datetime import datetime

time_now = datetime.now().strftime('%Y%m%d_%H%M%S')
# logging
log_filename = f"{time_now}_training_log.txt"
sys.stdout = open(log_filename, 'w')

################################
########## PARAMETERS ##########
################################

model_name = "distilbert/distilgpt2"
n_labels = 28
batch_size = 32
epochs = 25
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
patience = 5

###################################
######### Data processing #########
###################################

# set the column names for pd dfs
columns = ['text', 'labels', 'id']

# Load the datasets from tsv files (downloaded from https://github.com/google-research/google-research/tree/master/goemotions/data)
train_df = pd.read_csv('data/splits/train.tsv', sep='\t', header=None, names=columns)
dev_df = pd.read_csv('data/splits/dev.tsv', sep='\t', header=None, names=columns)
test_df = pd.read_csv('data/splits/test.tsv', sep='\t', header=None, names=columns)

# remove rows with more than 1 label
filtered_df_train = train_df[train_df['labels'].apply(len) == 1]
filtered_df_test = test_df[test_df['labels'].apply(len) == 1]
filtered_df_dev = dev_df[dev_df['labels'].apply(len) == 1]
filtered_df_train['labels'] = filtered_df_train['labels'].astype(int)
filtered_df_test['labels'] = filtered_df_test['labels'].astype(int)
filtered_df_dev['labels'] = filtered_df_dev['labels'].astype(int)

# Initialize the datasets
train_dataset = TextDataset(filtered_df_train)
valid_dataset = TextDataset(filtered_df_dev)
test_dataset = TextDataset(filtered_df_test)

# Initialize the dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

###########################
########## MODEL ##########
###########################
# get the model config object
model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path=model_name, num_labels=n_labels, output_attentions=True)

# get the GPT2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path=model_name)

# add padding on the left side of sequences
tokenizer.padding_side = "left"

# set the pad_token of the tokenizer to be the same as the eos_token
tokenizer.pad_token = tokenizer.eos_token

# Initialize the GPT-2 model for sequence classification 
model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name, config=model_config)

# Adjust the model’s embedding layer to match the vocabulary size of the tokenizer
model.resize_token_embeddings(len(tokenizer))

model.config.pad_token_id = model.config.eos_token_id

# move model to device
model.to(device) # GPUs go brrr

#######################################
####### Pre-training Accuracy #########
#######################################

test_true_labels, test_predicted_labels, test_loss = validate_one_epoch(
    model, tokenizer, test_dataloader, device
)

test_accuracy = accuracy_score(test_true_labels, test_predicted_labels)
test_precision = precision_score(test_true_labels, test_predicted_labels, average='weighted')
test_recall = recall_score(test_true_labels, test_predicted_labels, average='weighted')
test_f1 = f1_score(test_true_labels, test_predicted_labels, average='weighted')

print(f"Pretrained Model Test Accuracy: {test_accuracy:.5f}")
print(f"Pretrained Model Test Precision: {test_precision:.5f}")
print(f"Pretrained Model Test Recall: {test_recall:.5f}")
print(f"Pretrained Model Test F1-score: {test_f1:.5f}")

###########################
######## TRAINING #########
###########################

# Initialize the Adam optimizer 
# https://huggingface.co/docs/transformers/en/main_classes/optimizer_schedules#transformers.AdamW
# Triggers a deprecation warning, but works for now 
optimizer = AdamW(model.parameters(), 
                  lr = 1e-5, # default is 0.001 
                  betas = (0.9, 0.999), # default is (0.9, 0.999)
                  eps = 1e-8) # default is 1e-6 

# needed for the scheduler
total_steps = len(train_dataloader) * epochs

# Define a learning rate scheduler with a linear warmup and decay schedule
# https://huggingface.co/docs/transformers/en/main_classes/optimizer_schedules#transformers.SchedulerType
#scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps) 
# scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
scheduler = get_constant_schedule_with_warmup(optimizer,num_warmup_steps=0)

# Store training and validation loss, accuracy, precision, recall, and F1-score
metrics_loss = {'training_loss': [], 'validation_loss': []}
metrics_accuracy = {'training_accuracy': [], 'validation_accuracy': []}
metrics_precision = {'training_precision': [], 'validation_precision': []}
metrics_recall = {'training_recall': [], 'validation_recall': []}
metrics_f1 = {'training_f1-score': [], 'validation_f1-score': []}

#track best model
best_validation_accuracy = 0.0
best_model_state_dict = None
no_improvement_counter = 0

# Training loop over all epochs
for current_epoch in tqdm(range(epochs), desc="Training Epochs Progress"):
    print(f"Epoch {current_epoch + 1}/{epochs}")
    
    # Training 
    train_true_labels, train_predicted_labels, training_loss = train_one_epoch(
        model, tokenizer, train_dataloader, optimizer, scheduler, device
    )
    training_accuracy = accuracy_score(train_true_labels, train_predicted_labels)
    training_precision = precision_score(train_true_labels, train_predicted_labels, average='weighted')
    training_recall = recall_score(train_true_labels, train_predicted_labels, average='weighted')
    training_f1 = f1_score(train_true_labels, train_predicted_labels, average='weighted')
    
    # Validation
    validation_true_labels, validation_predicted_labels, validation_loss = validate_one_epoch(
        model, tokenizer, valid_dataloader, device
    )

    validation_accuracy = accuracy_score(validation_true_labels, validation_predicted_labels)
    validation_precision = precision_score(validation_true_labels, validation_predicted_labels, average='weighted')
    validation_recall = recall_score(validation_true_labels, validation_predicted_labels, average='weighted')
    validation_f1 = f1_score(validation_true_labels, validation_predicted_labels, average='weighted')

    # Log metrics
    print(f"train loss: {training_loss:.5f}, validation loss: {validation_loss:.5f}, train accuracy: {training_accuracy:.5f}, validation accuracy: {validation_accuracy:.5f}")
    print(f"train precision: {training_precision:.5f}, validation precision: {validation_precision:.5f}")
    print(f"train recall: {training_recall:.5f}, validation recall: {validation_recall:.5f}")
    print(f"train F1-score: {training_f1:.5f}, validation F1-score: {validation_f1:.5f}")

    # Append metrics to tracking dictionaries
    metrics_loss['training_loss'].append(training_loss)
    metrics_loss['validation_loss'].append(validation_loss)
    metrics_accuracy['training_accuracy'].append(training_accuracy)
    metrics_accuracy['validation_accuracy'].append(validation_accuracy)
    metrics_precision['training_precision'].append(training_precision)
    metrics_precision['validation_precision'].append(validation_precision)
    metrics_recall['training_recall'].append(training_recall)
    metrics_recall['validation_recall'].append(validation_recall)
    metrics_f1['training_f1-score'].append(training_f1)
    metrics_f1['validation_f1-score'].append(validation_f1)
    
    # Save the best model based on validation accuracy
    if validation_accuracy > best_validation_accuracy:
        best_validation_accuracy = validation_accuracy
        best_model_state_dict = model.state_dict()
        no_improvement_counter = 0
    else:
        no_improvement_counter += 1
        if no_improvement_counter >= patience: 
            print(f"Training stopped early after {current_epoch} epochs.")
            break

# Load the best model
model.load_state_dict(best_model_state_dict)
# save the best model 
model_name = f"{time_now}_gpt2_finetuned.pth"
# Save the model state_dict with the unique name
torch.save(model.state_dict(), model_name)

###########################
######## TESTING ##########
###########################

# Test the best model
test_true_labels, test_predicted_labels, test_loss = validate_one_epoch(
    model, tokenizer, test_dataloader, device
)

test_accuracy = accuracy_score(test_true_labels, test_predicted_labels)
test_precision = precision_score(test_true_labels, test_predicted_labels, average='weighted')
test_recall = recall_score(test_true_labels, test_predicted_labels, average='weighted')
test_f1 = f1_score(test_true_labels, test_predicted_labels, average='weighted')

print(f"Test Accuracy: {test_accuracy:.5f}")
print(f"Test Precision: {test_precision:.5f}")
print(f"Test Recall: {test_recall:.5f}")
print(f"Test F1-score: {test_f1:.5f}")

###########################
######## PLOTTING #########
###########################
os.makedirs("plots", exist_ok=True)

# Plot the metrics and save the figures
metrics_to_plot = {
    'Loss': metrics_loss,
    'Accuracy': metrics_accuracy,
    'Precision': metrics_precision,
    'Recall': metrics_recall,
    'F1-Score': metrics_f1
}

for metric_name, metric_dict in metrics_to_plot.items():
    plt.figure(figsize=(12, 6))
    plt.plot(metric_dict['training_' + metric_name.lower()], label=f'Training {metric_name}', marker='o')
    plt.plot(metric_dict['validation_' + metric_name.lower()], label=f'Validation {metric_name}', marker='o')
    plt.title(f'{metric_name} vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True)
    # Save the plot
    plot_filename = f"plots/{time_now}_{metric_name}_vs_Epochs.png"
    plt.savefig(plot_filename)
    plt.close()  # Close the figure to free memory
    print(f"Saved plot for {metric_name} at {plot_filename}")

print(model)