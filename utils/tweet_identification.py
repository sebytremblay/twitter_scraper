import numpy as np
import torch
from torch.utils.data import Dataset
from utils import data_cleaning as dc

# Create a custom dataset class
class TweetDataset(Dataset):
    """A custom dataset class for tweets."""
    
    def __init__(self, tweets, labels, tokenizer, max_len):
        self.tweets = tweets
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, index):
        tweet = self.tweets[index]
        label = self.labels[index]

        encoding = self.tokenizer.encode_plus(
            tweet,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'tweet_text': tweet,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def train_epoch(model, data_loader, optimizer, device, scheduler=None):
    """Train the model for one epoch.
    
    Args:
        model (torch.nn.Module): the model to train.
        data_loader (torch.utils.data.DataLoader): the data loader.
        optimizer (torch.optim.Optimizer): the optimizer to use.
        device (torch.device): the device to run the model on.
        scheduler (torch.optim.lr_scheduler._LRScheduler): the scheduler to use.
        
    Returns:
        float: the accuracy of the model on the data.
        float: the average loss on the data."""
    model = model.train()
    losses = []
    correct_predictions = 0

    for d in data_loader:
        input_ids = d['input_ids'].to(device)
        attention_mask = d['attention_mask'].to(device)
        labels = d['label'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        logits = outputs.logits

        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)

def eval_model(model, data_loader, device):
    """Evaluate the model on the data.

    Args:
        model (torch.nn.Module): the model to evaluate.
        data_loader (torch.utils.data.DataLoader): the data loader.
        device (torch.device): the device to run the model on.

    Returns:
        float: the accuracy of the model on the data.
        float: the average loss on the data.
    """
    model = model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d['input_ids'].to(device)
            attention_mask = d['attention_mask'].to(device)
            labels = d['label'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            logits = outputs.logits

            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)