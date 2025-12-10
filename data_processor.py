import pandas as pd
import numpy as np
from datetime import datetime
import os

def fetch_data(csv_file_path):
    """Fetch data from CSV file."""
    try:
        #Check if the file exists
        if not os.path.exists(csv_file_path):
            print(f"CSV file not Found: {csv_file_path}")
            return None 
        #Read CSV file
        data = pd.read_csv(csv_file_path)
        print(f"Succesfully Loaded {len(data)} transactions from {csv_file_path}")
        return data 
    except Exception as e:
        print(f"Error loading data from csv: {str(e)}")
        return None 
    
def preprocess_data(data):
        """ Preprocess the transaction data with consistency checks"""
        if data is None: 
            print("No data available. Please Fetch data first.")
            return None 
        
        print("Starting data preprocessing...")

        df= data.copy()

        total_duplicates = df.duplicated().sum()
        transaction_id_duplicates = df['TransactionID'].duplicated().sum()
        print(f"Total duplicate rows found: {total_duplicates}")
        print(f"Duplicate TransactionID found: {transaction_id_duplicates}")

        #Droping TransactionID duplicates
        df= df.drop_duplicates(subset = ['TransactionID'])

        unique_customers = df['CustomerID'].nunique()
        print(f"Unique Customers: {unique_customers}")

        #Handling missing values
        df = df.dropna()

        #Ensure Proper Datatime
        df['TransactionDate'] = pd.to_datetime(df['TransactionDate'], errors='coerce')
        print("Converted TransactionDate to datetime format.")
        return df

def calculate_rfm_metrics(data):
    """ Calculate RFM Metrics"""
    if data is None: 
        print("No data available. Please Fetch and Preprocess data first.")
        return None
    
    reference_date = data['TransactionDate'].max() + pd.Timedelta(days=1)

    rfm_df = data.groupby('CustomerID').agg({
    'TransactionDate': lambda x: (reference_date - x.max()).days,
    'TransactionID': 'count',
    'TransactionAmount': 'sum'
    }).reset_index()

    rfm_df.columns = ['CustomerID','Recency', 'Frequency', 'Monetary']

    print ("Adding Customer Demographics")


    customer_demographics = data.groupby('CustomerID').agg({
    'CustomerDOB': 'first',
    'CustGender': 'first',
    'CustAccountBalance': 'last'
    }).reset_index()

    customer_demographics.columns = ['CustomerID', 'DOB', 'Gender', 'AccountBalance']

    rfm_df = rfm_df.merge(customer_demographics, on = 'CustomerID', how = 'left')
    return rfm_df 