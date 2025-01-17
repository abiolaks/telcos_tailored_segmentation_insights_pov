import streamlit as st
from customers_insights_model import CustomerSegmentationApp


def main():
    app = CustomerSegmentationApp()

    st.title("Customer Value Management")
    st.header("Customer Segmentation and Insights Use Case")

    st.subheader("Data Loading")
    if st.button("Load Data"):
        data = app.load_data()
        data = app.preprocess_data(data)
        st.divider()
        st.subheader("Clustering and Customer Segment Analysis")
        app.cluster_data(data)

    st.divider()

    # st.subheader("Data Preprocessing")
    # if st.button("Preprocess Data"):
    #   app.preprocess_data()

    # st.divider()

    # st.subheader("Clustering and Customer Segment Analysis")
    # if st.button("Clusters Plot"):
    #   app.cluster_data()

    # st.divider()

    st.subheader("Cluster Analysis")
    if st.button("Generate Cluster Insights"):
        app.generate_cluster_insights()


if __name__ == "__main__":
    main()
