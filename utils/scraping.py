import pandas as pd
import os

def save_tweets(tweets, ticker, save_path="./"):
    """Save the tweets to a CSV file.

    Args:
        tweets (list): a list of twint.tweet objects.
        ticker (str): the stock ticker.
        save_path (str): the path where the produced CSV file will be saved.
    """
    # Extract the relevant information from each tweet
    cleaned_tweets = [{'id': tweet.id_str,
                    'url': tweet.url,
                    'symbols': ticker[1:] if ticker.startswith('$') else pd.NA,
                    'user': tweet.user.username,
                    'verifiedUser': tweet.user.blue or tweet.user.verified,
                    'quotedUser': tweet.quotedTweet.user.username if tweet.quotedTweet else pd.NA,
                    'verifiedQuotedUser': (tweet.quotedTweet.user.blue or tweet.quotedTweet.user.verified) if tweet.quotedTweet else pd.NA,
                    'timestamp': tweet.date,
                    'rawContent': tweet.rawContent,
                    'quotedContent': tweet.quotedTweet.rawContent if tweet.quotedTweet else pd.NA,
                    'retweetCount': tweet.retweetCount,
                    'likeCount': tweet.likeCount}
                    for tweet in tweets]

    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(cleaned_tweets)
    
    # Save the DataFrame to a CSV file
    if len(df) == 0:
        print("No tweets to save!")
        return
    df = df.sort_values(by=['timestamp'], ascending=False)
    save_to_csv(save_path, df)
    
def save_to_csv(file_path, df):
    """Save the DataFrame to a CSV file, appending to the file if it already exists.
    
    Args:
        file_path (str): the path to the CSV file.
        df (pd.DataFrame): the DataFrame to save."""
    if os.path.exists(file_path):
        # Read the existing data from the file
        existing_df = pd.read_csv(file_path)
        
        # Combine the data and remove duplicates
        combined_df = pd.concat([existing_df, df])
        final_df = combined_df.drop_duplicates()
        
        # Save the combined DataFrame back to the file
        final_df.to_csv(file_path, index=False)
    else: 
        # If the file does not exist, save the DataFrame directly
        df.to_csv(file_path, index=False)
        
def combine_csv_files(file_paths, output_path, column_names=['id']):
    """Combine multiple CSV files into a single file.
    
    Args:
        file_paths (list): a list of file paths to the CSV files.
        output_path (str): the path to save the combined CSV file.
        column_names (list): a list of column names to check for duplicates."""
    # Load all the dataframes
    dataframes = []
    for file in file_paths:
        df = pd.read_csv(file)
        print(f"Loaded {len(df)} rows from {file}")
        dataframes.append(df)
        
    # Combine the dataframes
    combined_df = pd.concat(dataframes)
    
    # Drop duplicates
    original_len = len(combined_df)
    combined_df = combined_df.drop_duplicates(subset=column_names)
    print(f"Dropped {original_len - len(combined_df)} duplicates.")
        
    # Save the combined dataframe to a new file
    combined_df.to_csv(output_path, index=False)