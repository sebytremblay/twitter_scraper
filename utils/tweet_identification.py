import numpy as np
import torch
import re
import pandas as pd
import openai
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
        
def forward_model(model, input_ids, attention_mask, labels, device):
    """Ensures the tensors are on the correct device before the forward pass.

    Args:
        model (torch.nn.Module): the model to forward pass.
        input_ids (torch.Tensor): the input ids.
        attention_mask (torch.Tensor): the attention mask.
        labels (torch.Tensor): the labels.
        device (torch.device): the device to run the model on.

    Returns:
        torch.Tensor: the model outputs.
    """
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    if labels is not None:
        labels = labels.to(device)
        return model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    else:
        return model(input_ids=input_ids, attention_mask=attention_mask)

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
        # Move the data to the device
        input_ids = d['input_ids']
        attention_mask = d['attention_mask']
        labels = d['label']
        
        # Reset the gradients
        optimizer.zero_grad()

        # Forward pass, handles moving tensors to device
        outputs = forward_model(model, input_ids, attention_mask, labels, device)
        loss = outputs.loss
        logits = outputs.logits

        # Get the predicted labels
        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        # Backpropagation
        loss.backward()
        optimizer.step()
        
        # Update the scheduler
        if scheduler is not None:
            scheduler.step(loss)

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
            input_ids = d['input_ids']
            attention_mask = d['attention_mask']
            labels = d['label']

            # Forward pass, handles moving tensors to device
            outputs = forward_model(model, input_ids, attention_mask, labels, device)
            loss = outputs.loss
            logits = outputs.logits

            # Get the predicted labels
            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)

def classify_tweets(tweets, model, tokenizer, label_encoder, device, max_len=64, batch_size=32):
    """Classify a batch of tweets using the model.
    
    Args:
        tweets (list of str): the tweets to classify.
        model (torch.nn.Module): the model to use.
        tokenizer (transformers.PreTrainedTokenizer): the tokenizer to use.
        label_encoder (sklearn.preprocessing.LabelEncoder): the label encoder.
        device (torch.device): the device to run the model on.
        max_len (int): the maximum length of the tokenized sequence.
        batch_size (int): the batch size for processing.
        
    Returns:
        list of str: the predicted labels."""
    all_preds = []

    for i in range(0, len(tweets), batch_size):
        batch = tweets[i:i + batch_size]
        
        # Tokenize the batch
        encoding = tokenizer(batch, add_special_tokens=True, max_length=max_len, return_token_type_ids=False, 
                             padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt')
        
        inputs = {k: v.to(device) for k, v in encoding.items()}
        with torch.no_grad():
            outputs = model(**inputs)

        # Get the predicted labels
        _, preds = torch.max(outputs.logits, dim=1)
        preds = label_encoder.inverse_transform(preds.cpu().numpy())
        all_preds.extend(preds)

    return all_preds

def is_valid_ticker(ticker, valid_tickers):
    """Determines whether a stock ticker is valid.

    Args:
        ticker (str): The stock ticker to validate.
        valid_tickers (set): A set of valid stock tickers (uppercase).
        
    Returns:
        bool: True if the ticker is valid, False otherwise.
    """
    return ticker.upper() in valid_tickers

def find_ticker(tweet, valid_tickers):
    """Finds all unique stock tickers in a tweet.

    Args:
        tweet (str): The tweet to search for tickers.
        valid_tickers (set): A set of valid stock tickers (uppercase).
        
    Returns:
        list: All unique tickers in the tweet.
    """
    # Find potential stock tickers in the tweet
    pattern = r"\$[A-Za-z]{1,5}"
    potential_tickers = re.findall(pattern, tweet)
    
    # Remove the $ from the tickers and convert to uppercase
    tickers = [ticker[1:].upper() if ticker.startswith('$') else ticker.upper() for ticker in potential_tickers]
    
    # Filter out invalid tickers
    unique_tickers = list(set(tickers))
    return [ticker for ticker in unique_tickers if is_valid_ticker(ticker, valid_tickers)]

def get_ticker_strings(tweet, valid_tickers, spam_theshold=0.1):
    """Determines all unique stock tickers in a tweet as a string.

    Args:
        tweet (str): The tweet to assign tickers.
        valid_tickers (set): A set of valid stock tickers (uppercase).
        spam_theshold (float): The ratio of tickers to text to consider spam.
        
    Returns:
        str: A string of all unique tickers in the tweet separated by a hyphen.
    """
    tickers = find_ticker(tweet, valid_tickers)
    
    # Calculate the ratio of tickers to tweet length
    ticker_count = len(tickers)
    tweet_word_count = len(tweet.split())
    ticker_to_text_ratio = ticker_count / tweet_word_count
    
    # Adjust the spam threshold dynamically based on the number of tickers
    dynamic_threshold = spam_theshold * (1 + ticker_count / 25)
    
    # Avoid tweets with no tickers or spammed tickers
    if len(tickers) == 0 or (ticker_count > 2 and ticker_to_text_ratio > dynamic_threshold):
        return pd.NA
    else:
        return '-'.join(tickers)
