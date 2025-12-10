import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()

def calculate_rfm_scores(rfm_data):

    if rfm_data is None:
        print ("No RFM data available for scoring.")
        return None 
    print("Calculating RFM Scores...")

    rfm_data['R_Score'] = pd.qcut(rfm_data['Recency'], q=5, labels=[5, 4, 3, 2, 1])

    rfm_data['F_Score']= pd.qcut(rfm_data['Frequency'], q=5, labels=[1,2,3,4,5])

    #### Monetary: Higher Money = Better
    rfm_data['M_Score'] = pd.qcut(rfm_data['Monetary'], q =5, labels = [1,2,3,4,5])

    print("Combining and Calculating RFM_Score and RFM_Group")

    rfm_data['rfm_score'] = rfm_data['R_Score'].astype(int) + rfm_data['F_Score'].astype(int) + rfm_data['M_Score'].astype(int)
    rfm_data['rmf_group'] = rfm_data['R_Score'].astype(str) + rfm_data['F_Score'].astype(str) + rfm_data['M_Score'].astype(str)

    print("RFM Scoring Completed.")
    print(f"Average RFM Score: {rfm_data['rfm_score'].mean():.2f}") 

    return rfm_data 

def prepare_rfm_data_for_clustering(rfm_data):
    """ Prepare RFM data for Clustering by Normalization and Scaling"""
    if rfm_data is None:
        print("No RFM data available for normalizing and scaling.")
        return None 
    
    print("Preparing RFM data for Clustering by Normalization...")

    ## Extract RFM Values
    rfm_clustering = rfm_data[['Recency', 'Frequency', 'Monetary']].copy()

    ## Log Transformation to reduce skewness
    rfm_clustering['Recency'] = np.log1p(rfm_clustering['Recency'])
    rfm_clustering['Frequency'] = np.log1p(rfm_clustering['Frequency'])
    rfm_clustering['Monetary'] = np.log1p(rfm_clustering['Monetary']) 

    ## Scaling using StandardScaler
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_clustering)
    rfm_scaled_df = pd.DataFrame(rfm_scaled, columns = ['Recency', 'Frequency', 'Monetary']) 

    print("RFM Data Normalization and Scaling Completed.")
    return rfm_scaled_df 



    