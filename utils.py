import pandas as pd

def load_data(file_path):
    """
    Load CSV data and parse dates.
    """
    return pd.read_csv(file_path, parse_dates=['InvoiceDate'])

def save_results(df, file_path):
    """
    Save DataFrame results to CSV.
    """
    df.to_csv(file_path, index=False)
    print(f"Results saved to {file_path}")
