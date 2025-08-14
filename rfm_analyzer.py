import pandas as pd
from datetime import datetime

def calculate_rfm(data: pd.DataFrame, current_date=None):
    """
    Calculate Recency, Frequency, and Monetary scores for each customer.
    """
    if current_date is None:
        current_date = datetime.now()

    rfm_df = data.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (current_date - x.max()).days,
        'InvoiceNo': 'count',
        'TotalAmount': 'sum'
    }).reset_index()

    rfm_df.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

    # Assign scores (simple quantile-based)
    rfm_df['R_Score'] = pd.qcut(rfm_df['Recency'], 4, labels=[4, 3, 2, 1])
    rfm_df['F_Score'] = pd.qcut(rfm_df['Frequency'].rank(method="first"), 4, labels=[1, 2, 3, 4])
    rfm_df['M_Score'] = pd.qcut(rfm_df['Monetary'], 4, labels=[1, 2, 3, 4])

    # Combine into one RFM score
    rfm_df['RFM_Segment'] = rfm_df[['R_Score', 'F_Score', 'M_Score']].astype(str).sum(axis=1)
    rfm_df['RFM_Score'] = rfm_df[['R_Score', 'F_Score', 'M_Score']].sum(axis=1).astype(int)

    return rfm_df
