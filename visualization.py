import pandas as pd
import matplotlib.pyplot as plt
import numpy as np




def generate_cluster_profiles(rfm_df):
    """ Generate Summarized Cluster Profile From RFM DataFrame"""
    cluster_profiles = rfm_df.groupby('Cluster_Name').agg({
        'Recency': ['mean', 'std'],
        'Frequency': ['mean', 'std'],
        'Monetary': ['mean', 'std'],
        'CustomerID': 'count',
        'AccountBalance': 'mean',
    }).round(2)

    cluster_profiles.columns = ['Avg_Recency', 'Std_Recency', 'Avg_Frequency',
                           'Std_Frequency', 'Avg_Monetary', 'Std_Monetary',
                           'Customer_Count', 'Avg_Account_Balance']

    ####Percentage Representation
    cluster_profiles ['Percentage'] = (cluster_profiles['Customer_Count']/len(rfm_df)*100).round(2)
    return cluster_profiles 

def plot_segmentation_distribution(rfm_df): 
    """ Generate Pie Chart for Customer Segmentation Distribution"""
    segment_counts = rfm_df['Cluster_Name'].value_counts()
    plt.figure(figsize = (10, 8))
    explode = [0.05] * len(segment_counts)

    plt.pie(segment_counts.values, labels= segment_counts.index,
        autopct= lambda p : f'{p:.2f}%\n({int(p*sum(segment_counts.values)/100):,})',
        startangle=90,
        explode = explode,
        shadow = True
        )
    plt.title('RFM CUSTOMER SEGMENTATION DISTRIBUTION', fontsize= 16, fontweight='bold')
    plt.axis('equal') 
    return plt

def plot_avg_values_by_segment(cluster_profiles):
    """ Generate Bar Charts for Average RFM Values by Segment"""
    plt.figure(figsize=(10,5))
    rfm_metrics = ['Avg_Recency', 'Avg_Frequency', 'Avg_Monetary']
    x_pos = np.arange(len(cluster_profiles))
    width = 0.25

    for i, metric in enumerate(rfm_metrics):
        values = cluster_profiles [metric].values
        if metric == 'Avg_Monetary':
            values = values /1000 ###Thousands

        plt.bar(x_pos + i*width, values, width, label = metric.replace('Avg_', ''), alpha =0.8)

    plt.xlabel('Customer Segment')
    plt.ylabel('Values (Monetary in thousands)')
    plt.title('Average RFM Values by Segment')
    plt.xticks(x_pos + width, cluster_profiles.index, rotation = 45)
    plt.legend()
    plt.grid(True, alpha = 0.3)
    plt.tight_layout()
    return plt

