import streamlit as st
import pandas as pd
import numpy as np
from data_processor import fetch_data, preprocess_data, calculate_rfm_metrics
from rfm_analyzer import calculate_rfm_scores, prepare_rfm_data_for_clustering
from clustering_engine import apply_clustering, assign_cluster_names
from visualization import generate_cluster_profiles, plot_segmentation_distribution, plot_avg_values_by_segment
import openai
from dotenv import load_dotenv
load_dotenv(override=True)
import os 

def main():
    st.set_page_config(
        page_title = "Bank Trust RFM Analysis",
        page_icon = "🏦",
        layout = "wide",
        initial_sidebar_state = "expanded"
    )

    st.title("🏦 Bank Trust RFM Customer Segmentation Analysis")
    st.markdown ("""
    This application performs RFM (Recency, Frequency, Monetary) 
            analysis on Bank Trust customer transaction data to segment 
            customers based on their transaction behavior. 
            The analysis includes data preprocessing, RFM metric calculation, 
            scoring, clustering, and visualization of customer segments.
    """)


    ### Run the analysis pipeline

    raw_data = fetch_data("data/Bank_Trust_Dataset.csv")
    processed_data = preprocess_data(raw_data) 

    rfm_data = calculate_rfm_metrics(processed_data)

    ## Calculate RFM Scores from rfm_analyzer.py
    rfm_data = calculate_rfm_scores(rfm_data)

    ## Fix: Send only numeric columns for scaling from rfm_analyzer.py
    rfm_scaled_df = prepare_rfm_data_for_clustering(rfm_data) 

    ## Apply Clustering from clustering_engine.py
    cluster_analysis, rfm_data, _ = apply_clustering(rfm_scaled_df, rfm_data)
    clustered_data = assign_cluster_names(rfm_data, cluster_analysis)

    cluster_profile = generate_cluster_profiles(clustered_data) 

    ### Create Tabs for Visualization

    tab1, tab2, tab3 = st.tabs (["RFM Visualizations", "Customer Categories", "Recommendations"])

    with tab1: 
        col1, col2 = st.columns(2)
        with col1: 
            st.pyplot(plot_segmentation_distribution(clustered_data))
            st.pyplot(plot_avg_values_by_segment(cluster_profile))
        with col2:
            st.pyplot(plot_avg_values_by_segment(cluster_profile))

    with tab2:
        def get_segmented_customers(rfm_df): 
            base_columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary', 
                            'R_Score', 'F_Score', 'M_Score', 'rfm_score', 
                            'rmf_group', 'Cluster_Name'] 
            
            available_columns = [col for col in base_columns if col in rfm_df.columns]
            segmented_customers = rfm_df[available_columns].copy()
            segmented_customers.reset_index(drop = True, inplace =True)
            return segmented_customers 
        
        segmented_df = get_segmented_customers(clustered_data)
        st.dataframe(segmented_df)

        csv_data = segmented_df.to_csv(index = False)
        st.download_button(
            label = "Download Segmented Customers as CSV",
            data = csv_data,
            file_name = "rfm_segmented_customers.csv",
            mime = "text/csv"
        )

    with tab3: 
        st.subheader("Recommendations")
        prompt = f"""
        Based on this RFM cluster profile data: {cluster_profile.to_dict()}

        Give recommendations for marketing strategies to target each customer segment effectively.
        Provide specific strategies for at least three different customer segments.
        """

        client = openai.OpenAI(
            api_key=os.getenv("api_key"), 
            base_url="https://api.grog.com/openai/v1"
        )
        response = client.responses.create(
            model = "openai/gpt-oss-120b", 
            input = prompt,
        )
        st.markdown(response.output_text)

if __name__ == "__main__":
    main() 
