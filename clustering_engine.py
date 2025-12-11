import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

#Global variables
kmeans_model = None
rfm_data = None 

def find_optimal_clusters(rfm_scaled_df): 
    """Determine the optimal number of clusters"""
    wcss = []
    silhouette_scores = []
    cluster_range = range(2, 11)

    for k in cluster_range:
        kmeans = KMeans(n_clusters = k, random_state = 42, n_init = 10)
        kmeans.fit(rfm_scaled_df)
        wcss.append(kmeans.inertia_)

  # For the Silhouette Score (moved inside the loop)
        score = silhouette_score(rfm_scaled_df, kmeans.labels_)
        silhouette_scores.append(score)

    best_k = cluster_range[np.argmax(silhouette_scores)] 

    return {
        'wcss': wcss,
        'silhouette_scores': silhouette_scores,
        'optimal_k': best_k,
        'cluster_range': cluster_range ,
        'best_silhouette_score': max(silhouette_scores)
    }

def apply_clustering(rfm_scaled_df, rfm_data, optimal_k=4):
    """Apply KMeans Clustering cleanly without global variables"""

    if optimal_k is None:
        results = find_optimal_clusters(rfm_scaled_df)
        optimal_k = results['optimal_k']
    
    kmeans = KMeans(n_clusters=optimal_k, random_state= 42, n_init=10)
    rfm_data = rfm_data.copy()
    rfm_data['Cluster'] = kmeans.fit_predict(rfm_scaled_df)

    print(rfm_data.head())

    ### Converting RFM Scores to INTs

    rfm_data['R_Score'] = rfm_data['R_Score'].astype(int)
    rfm_data['F_Score'] = rfm_data['F_Score'].astype(int)
    rfm_data['M_Score'] = rfm_data['M_Score'].astype(int)

    cluster_analysis = rfm_data.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean',
        'R_Score': 'mean',
        'F_Score': 'mean',
        'M_Score': 'mean',
        'CustomerID': 'count',
        'AccountBalance': 'mean',
        'Gender': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown'
    }).round(2)

    cluster_analysis.columns = ['Avg_Recency', 'Avg_Frequency', 'Avg_Monetary',
                            'Avg_R_Score', 'Avg_F_Score', 'Avg_M_Score',
                            'Customer_Count', 'Avg_Account_Balance',
                            'Gender_Mode']
    
    return cluster_analysis, rfm_data, optimal_k 

def assign_cluster_name(stats): 
    """Assign Descriptive Names to Clusters based on RFM Statistics"""    
    recency = stats['Avg_Recency']
    frequency = stats['Avg_Frequency']
    monetary = stats['Avg_Monetary']

  #### Recency Thresholds

    if recency >365:
        if monetary >  20000:   ### ie havent purchased in a year
            return "High Value Dormant Customers"
        else:
            return "Dormant Low Value Customers"
    elif recency > 180:  #### Havent purchased in 6-12 months
        if monetary > 20000:
            return "High Value Inactive Customers"
        else:
            return "Inactive Low Value Customers"

#### Active Customers (Recent Transaction)

    if frequency > 10 and monetary > 30000:
        return "Active VIP Customers"
    elif frequency > 8 and monetary < 10000:
        return "Active Low Value Customers"
    else:
        return "Regular Customers"


def assign_cluster_names(rfm_df, cluster_analysis): 
    rfm_df = rfm_df.copy()
    print(rfm_df.head())
    rfm_df['Cluster_Name'] = rfm_df['Cluster'].map(lambda x: assign_cluster_name(cluster_analysis.loc[x]))
    
    return rfm_df 
