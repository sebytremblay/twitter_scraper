import pandas as pd
import string
import re

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def preprocess_tweet(tweet, lemmatizer=WordNetLemmatizer(),
                     keep_urls=True, keep_mentions=True, keep_stock_tickers=True):
    """Apply NLP preprocessing to a given tweet, preserving stock tickers, URLs, and mentions.

    Args:
        tweet (str): The tweet to be preprocessed.
        lemmatizer (WordNetLemmatizer): The lemmatizer to use.
        keep_urls (bool): Whether to keep URLs in the tweet.
        keep_mentions (bool): Whether to keep mentions in the tweet.
        

    Returns:
        str: The preprocessed tweet.
    """
    patterns = {
        'ticker': r'\$\w+',
        'url': r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
        'mention': r'@\w+',
    }
    placeholder_map = {}

    # Replace special patterns with placeholders
    for key, pattern in patterns.items():
        for i, match in enumerate(re.findall(pattern, tweet)):
            placeholder = f"__{key}{i}__"
            placeholder_map[placeholder] = match
            tweet = tweet.replace(match, placeholder)

    # Processing steps: tokenize, lower, remove stopwords, remove punctuation, and lemmatize
    tokens = word_tokenize(tweet.lower())
    stop_words = set(stopwords.words('english'))
    punctuation = set(string.punctuation + '“”‘’—')

    filtered_tokens = [
        lemmatizer.lemmatize(token) if token not in placeholder_map else token
        for token in tokens
        if token not in stop_words and (token in placeholder_map or not any(char in punctuation for char in token))
    ]

    # Restore placeholders with original content
    final_text = ' '.join(placeholder_map.get(token, token) for token in filtered_tokens)
    
    return final_text

def init_df(force_data_reload, raw_data_path, processed_data_path, data_size, tweet_col):
    """Initializes dataframe from file path. Re-runs processing if processed file does not exist or re-process flag is set.

    Args:
        force_data_reload (_type_): _description_
        raw_data_path (_type_): _description_
        processed_data_path (_type_): _description_
        data_size (_type_): _description_
        tweet_col (_type_): _description_

    Raises:
        ValueError: _description_
        ValueError: _description_
        
    Return:
        pd.DataFrame: the loaded dataframe.
    """
    try:
        # If force_data_regeneration is set, force an exception to reload the data
        if force_data_reload:
            print('Forcing data regeneration.')
            raise ValueError('Forcing data regeneration.')
        
        # Load the preprocessed data if it exists
        df = pd.read_csv(processed_data_path)
        
        # If dataframe is not expected size, reload the data
        if data_size != -1 and len(df) > data_size:
            df = df.sample(n=data_size)
        elif data_size != -1 and len(df) < data_size:    
            print('Preprocessed file is not the expected size. Reloading data.')
            raise ValueError('Preprocessed file is not the expected size.')
        
        print('Preprocessed file found and loaded.')
    except (FileNotFoundError, ValueError):
        # Load raw data
        df = pd.read_csv(raw_data_path)
        
        # Ensure no duplicates
        orig_len = len(df)
        df = df.drop_duplicates()
        print(f"Removed {orig_len - len(df)} duplicates")

        # Preprocess the tweet column
        df[tweet_col] = df[tweet_col].apply(lambda x: preprocess_tweet(x))
        
        # Save the preprocessed file for reuse
        df.to_csv(processed_data_path)
        print('Processing complete and saved to disk.')
        
        # Sample data accordingly
        df = df.sample(n=data_size)

    # Display the preprocessed dataframe
    pd.set_option('display.max_colwidth', None)
    print(f"Dataframe shape: {df.shape}")
    return df