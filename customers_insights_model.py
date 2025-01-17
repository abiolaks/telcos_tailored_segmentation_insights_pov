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
        api_key = st.secrets.get("OPENAI_API_KEY")
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
