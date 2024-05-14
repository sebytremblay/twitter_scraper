import string
import re
import pandas as pd

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy.sparse import hstack
from textblob import TextBlob
from utils import stock_pricing as sp
from utils import data_cleaning as dc

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

def prepare_features(df, text_columns, categorical_columns, numeric_columns, target_column, vectorizers, encoder, scaler, training_data=False):
    """Prepare the features for training and evaluation.

    Args:
        df (pd.DataFrame): The input DataFrame.
        text_columns (list): The names of the text columns.
        categorical_columns (list): The names of the categorical columns.
        numeric_columns (list): The names of the numeric columns.
        target_column (str): The name of the target column.
        vectorizers (list of TfidfVectorizer): A list of fitted text vectorizers, one for each text column.
        encoder (OneHotEncoder): The fitted categorical encoder.
        scaler (StandardScaler): The fitted numeric scaler.
        training_data (bool): Whether the data is for training or evaluation.
    
    Returns:
        X: The input features.
        y: The target variable.
    """
    # Fit the vectorizers, encoder, and scaler if this is training data
    if training_data:
        for text_column, vectorizer in zip(text_columns, vectorizers):
            vectorizer.fit(df[text_column].astype('U'))
        encoder.fit(df[categorical_columns])
        scaler.fit(df[numeric_columns])

    # Transform the text features
    text_features = [vectorizer.transform(df[text_column].astype('U')) for text_column, vectorizer in zip(text_columns, vectorizers)]
    if text_features:
        text_features = hstack(text_features)

    # Transform the categorical and numeric features
    categorical_features = encoder.transform(df[categorical_columns])
    numeric_features = scaler.transform(df[numeric_columns])

    # Concatenate all features
    X = hstack([text_features, categorical_features, numeric_features])

    # Extract the target variable
    y = df[target_column].values
    
    return X, y


def run_preprocessing(raw_data_path, processed_data_path, 
                      timestamp_col, symbol_col, 
                      database_size, 
                      raw_text_columns, preprocessed_text_columns, 
                      lemmatizer=WordNetLemmatizer()):
    """Preprocess the raw data and save the processed data to a new file.

    Args:
        raw_data_path (str): The path to the raw data file.
        processed_data_path (str): The path to save the processed data.
        timestamp_col (str): The name of the timestamp column.
        symbol_col (str): The name of the symbol column.
        database_size (int): The desired size of the resulting database.
        raw_text_columns (list): The names of the raw text columns.
        preprocessed_text_columns (list): The names of the preprocessed text columns.
        lemmatizer (WordNetLemmatizer): The lemmatizer to use.
        
    Returns:
        pd.DataFrame: The preprocessed data.
    """
    # Load dataset with stock data
    df = sp.preprocess_nasdaq_df(raw_data_path, timestamp_col, symbol_col, database_size)
    
    for raw_col, processed_col in zip(raw_text_columns, preprocessed_text_columns):
        # Add sentiment column with TextBlob if it doesn't exist
        df[f'{raw_col}_polarity'] = df[raw_col].apply(lambda tweet: TextBlob(tweet).sentiment.polarity if pd.notna(tweet) else None)
        df[f'{raw_col}_subjectivity'] = df[raw_col].apply(lambda tweet: TextBlob(tweet).sentiment.subjectivity if pd.notna(tweet) else None)
        
        # Apply preprocessing to the raw column
        df[processed_col] = df[raw_col].apply(lambda tweet: dc.preprocess_tweet(tweet, lemmatizer) if pd.notna(tweet) else None)
        
    # Save the preprocessed data
    df.to_csv(processed_data_path, index=False)
    print('File preprocessing completed and saved.')
    
    return df