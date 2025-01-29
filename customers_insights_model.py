# import the require libraries needed
import streamlit as st  # for model UI and deployment
from openai import OpenAI  # for open ai api access
from sklearn.cluster import KMeans  # for clustering analysis
from sklearn.preprocessing import (
    StandardScaler,
)  # to ensure that numerical features are on the same scale
import matplotlib.pyplot as plt  # for visualization
import pandas as pd  # for data analyis


# create a class : for resuabilty and code extension
class CustomerSegmentationApp:
    # creating a function for constructor to instantiate the class
    def __init__(self):
        self.data = None
        self.clustered_data = None
        self.cluster_centers = None
        self.client = self.get_openai_client()

    def get_openai_client(self):
        """_summary_: Initialize and return the OpenAI API client.
        _return_: OpenAI API client
        """
        api_key = st.secrets["secrets"]["OPENAI_API_KEY"]
        if not api_key:
            st.error("No OpenAI key found, Please set your API key. ")
            return None
        return OpenAI(api_key=api_key)

    # Load data, data pipeline that pick from azure blob or upload csv from machine
    def load_data(self):
        """_summary_:Handle file upload and load customer data.
        _return_: the loaded data
        """
        upload_file = st.file_uploader(
            "upload your telecom customer data (default: CSV)", type="csv"
        )
        if upload_file:
            self.data = pd.read_csv(
                upload_file
            )  # convert csv to dataframe and save in data
            st.write("Data Preview")
            st.write(self.data.head())
        else:
            st.warning("Please upload a csv file.")

        return self.data

    # Preprocess data
    def preprocess_data(self):
        """_summary_: scale the numerical values, encode categorical and fill null values.
        _return_: preprocess data
        """
        if self.data is not None:
            categorical_features = ["Gender", "Region"]
            self.data = pd.get_dummies(
                self.data, columns=categorical_features, drop_first=True
            )
            self.data.fillna(self.data.median(), inplace=True)
            # features engineering
            self.data["Recency"] = 30 - self.data["LastPurchaseDays"]
            self.data["Frequency"] = self.data["CallsMade"] / 30
            self.data["Monetary"] = self.data["MonthlySpending"]

            # scalling numerical features
            scaler = StandardScaler()
            features = ["Recency", "Frequency", "Monetary", "DataUsageGB"]
            self.data[features] = scaler.fit_transform(self.data[features])
            st.success("Data Preprocessed Successfully.")
        return self.data

    # cluster Analysis
    def cluster_data(self):
        """_summary_: Perform customer segmentation using"""
        if self.data is not None:
            st.subheader("Customer Segmentation")
            n_clusters = st.slider(
                "Select Number of Clusters", min_value=2, max_value=10, value=3
            )
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            self.data["Cluster"] = kmeans.fit_predict(
                self.data[["Recency", "Frequency", "Monetary", "DataUsageGB"]]
            )

            # Visualization of clusters
            plt.figure(figsize=(10, 6))
            plt.scatter(
                self.data["Monetary"],
                self.data["Recency"],
                c=self.data["Cluster"],
                cmap="viridis",
            )
            plt.title("Customer Segmentation")
            plt.xlabel("Recency")
            plt.ylabel("Monetary")
            st.pyplot(plt)

            st.success("Clustering completed.")
            self.clustered_data = self.data
            self.clustered_data.to_csv("clustered_data.csv", index=False)
            # self.cluster_centers = kmeans.cluster_centers_
        else:
            st.warning("Preprocessed data is required for clustering")
        return self.clustered_data

    # Generate insights
    def generate_cluster_insights(self):
        """_summary_:Generate insights for each cluster using openAI"""
        if self.client and self.clustered_data is not None:
            system_prompt = """
                You are a Telecommunication Customer Insights Analyst. You are tasked with analyzing Customer Clusters
                for actionable insights.
                
                Give a concise and detailed response with the specified output format below in an easy-to-understand manner
                for the Marketing, Sales Team to act on.
                
                Output in Markdown
                1. Demographic Insights
                2. Customer Behaviour Analysis
                3. Tailor Marketing Strategies
                4. Product and Pricing Strategies
                
                Strictly stick to the output and format it in Markdown for each cluster.
                
                """
            cluster_summary = (
                self.clustered_data.groupby("Cluster").mean().reset_index()
            )
            for cluster_id in cluster_summary["Cluster"]:
                cluster_data = cluster_summary.loc[
                    cluster_summary["Cluster"] == cluster_id
                ]
                prompt = f"Analyze the following customer data for Cluster{cluster_id}: {cluster_data.to_dict()}"

                # Generative AI API call for Insights
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.7,
                    top_p=1,
                    max_tokens=600,
                )
                insights = response.choices[0].message.content
                st.write(f"Cluster {cluster_id} Insights:")
                st.write(insights)
            else:
                st.warning("Client or clustered data not available")
