# Step 3: Clustering and Analysis
st.subheader("📈 Clustering and Customer Segment Analysis")
if st.button("Run Clustering"):
    with st.spinner("Running clustering algorithm..."):
        app.cluster_data(num_clusters=num_clusters)
    st.success("Clustering completed!")
    st.write("Cluster Visualization:")
    st.pyplot(app.plot_clusters())  # Display the 3D plot

# Step 4: Generate Insights
st.subheader("🔍 Cluster Insights")
if st.button("Generate Insights"):
    if app.clustered_data is not None:  # Check if clustering has been performed
        with st.spinner("Generating insights using LLMs..."):
            insights = app.generate_cluster_insights()

        if insights:  # Check if insights are generated
            st.success("Insights generated successfully!")
            for i, insight in enumerate(insights):
                with st.expander(f"Cluster {i+1} Insights"):
                    st.markdown(insight)  # Display insights in markdown format
        else:
            st.warning("No insights were generated. Please check the logs for errors.")
    else:
        st.error("Clustered data is not available. Please run clustering first.")
