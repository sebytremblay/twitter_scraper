import yfinance as yf
import pandas as pd
import pandas_market_calendars as mcal

def is_trading_day(date, nyse=mcal.get_calendar('NYSE')):
    """Check if the given date is a trading day for the NYSE.

    Args:
        date (str): The date to check in 'YYYY-MM-DD' format.
        nyse (pandas_market_calendars.exchange_calendar.ExchangeCalendar): The NYSE calendar.
        
    Returns:
        bool: True if the date is a trading day, False otherwise.
    """
    # Use the date parameter to create a date range where the start and end are the same date
    schedule = nyse.schedule(start_date=date, end_date=date)
    
    # If the schedule DataFrame is empty, the date is not a trading day
    return not schedule.empty

def create_df(ticker_lst):
    """Builds a dictionary of stock dataframes for the given list of tickers.

    Args:
        ticker_lst (list): A list of stock tickers.

    Returns:
        dict: A dictionary of stock dataframes, where the keys are the stock tickers.
    """
    ticker_dfs = {}
    for ticker in ticker_lst:
        # Get stock data
        stock = yf.Ticker(ticker)
        stock_d = stock.history(period="10y")
        
        # Add stock data to dictionary
        ticker_dfs[ticker] = stock_d
        ticker_dfs[ticker].reset_index(inplace=True)
    
    return ticker_dfs

def most_recent_stock_date(tweet_date, when, nyse):
    """
    Returns the most recent trading day given the conditions.
    
    Args:
        tweet_date (pandas.Timestamp): The date of the tweet.
        when (str): The condition to check.
        nyse (pandas_market_calendars.exchange_calendar.ExchangeCalendar): The NYSE calendar.
        
    Returns:
        str: The most recent trading day.
    """
    if when == 'day before':
        day = tweet_date - pd.Timedelta(days=1)
        while not is_trading_day(day.strftime('%Y-%m-%d'), nyse):
            day = day - pd.Timedelta(days=1)
        return day.strftime('%Y-%m-%d')
    elif when == 'day of':
        day = tweet_date
        while not is_trading_day(day.strftime('%Y-%m-%d'), nyse):
            day = day - pd.Timedelta(days=1)
        return day.strftime('%Y-%m-%d')
    elif when == 'day after':
        day = tweet_date + pd.Timedelta(days=1)
        while not is_trading_day(day.strftime('%Y-%m-%d'), nyse):
            day = day + pd.Timedelta(days=1)
        return day.strftime('%Y-%m-%d')

def add_price_data(stock_dict, tweet_df, symbols_col):
    """Add stock price data to the tweet DataFrame.

    Args:
        stock_dict (dict): A dictionary of stock dataframes.
        tweet_df (pandas.DataFrame): The tweet DataFrame.
        symbols_col (str): The name of the column containing the stock symbols.

    Returns:
        pandas.DataFrame: The tweet DataFrame with stock price data added.
    """
    # Load the NYSE calendar
    nyse = mcal.get_calendar('NYSE')
    
    # Track dropped rows
    rows_to_drop = []
                
    for index, row in tweet_df.iterrows():
        # Get the stock data for the ticker in the tweet
        stock_df = stock_dict.get(row[symbols_col], None)
        
        # Extract the date string
        tweet_date = row['timestamp']
        day_before = most_recent_stock_date(tweet_date, 'day before', nyse)
        day_of = most_recent_stock_date(tweet_date, 'day of', nyse)
        day_after = most_recent_stock_date(tweet_date, 'day after', nyse)
        
        # Filter the DataFrame based on the date string
        filtered_df_before = stock_df.loc[stock_df['Date'] == day_before]
        filtered_df_of = stock_df.loc[stock_df['Date'] == day_of]
        filtered_df_after = stock_df.loc[stock_df['Date'] == day_after]
        
        # Drop the row if any of the filtered DataFrames are empty
        if filtered_df_before.empty or filtered_df_of.empty or filtered_df_after.empty:
            rows_to_drop.append(index)
            continue

        # Extract the price values
        price_day_before = filtered_df_before['Close'].values[0]
        price_day_of = filtered_df_of['Close'].values[0]
        price_day_after = filtered_df_after['Close'].values[0]

        # Update the tweet DataFrame
        tweet_df.at[index, 'Price Day Before Tweet'] = price_day_before
        tweet_df.at[index, 'Price Day of Tweet'] = price_day_of
        tweet_df.at[index, 'Price Day After Tweet'] = price_day_after
        
    # Drop rows with missing values
    tweet_df = tweet_df.drop(rows_to_drop)
    print(f"Finished adding price data. Dropped {len(rows_to_drop)} rows.")  
      
    return tweet_df

def preprocess_nasdaq_df(raw_file_path, timestamp_col, symbols_col, size=-1):     
    """Loads the stockerbot-export.csv file and preprocesses the data.
    
    Args:
        raw_file_path (str): The path to the original CSV file.
        size (int): The number of rows to load from the CSV file.
        timestamp_col (str): The name of the column containing the timestamp.
        symbols_col (str): The name of the column containing the stock symbols.

    Returns:
        pandas.DataFrame: The preprocessed data.
    """
    # Load the data frame  
    df = pd.read_csv(raw_file_path, on_bad_lines='skip')
    print(f"Loaded {len(df)} rows from the CSV file.")
    
    # If size is provided, subset the dataframe
    if size > 0:
        df = df.sample(size)
    
    # Drop rows with missing values
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
    df = df.dropna(subset=[timestamp_col])

    # Access date components for each timestamp
    df['month'] = df[timestamp_col].dt.month
    df['day'] = df[timestamp_col].dt.day
    df['year'] = df[timestamp_col].dt.year

    # Add new columns to the DataFrame
    df['Price Day Before Tweet'] = pd.NA
    df['Price Day of Tweet'] = pd.NA
    df['Price Day After Tweet'] = pd.NA

    # Create a dictionary of stock dataframes
    ticker_lst = df[symbols_col].unique().tolist()
    stock_data = create_df(ticker_lst)
    
    # Merge the stock data with the tweet data
    return add_price_data(stock_data, df, symbols_col)