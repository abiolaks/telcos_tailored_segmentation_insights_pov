# telcos_tailored_segmentation_insights_pov
Building a customer segmentation with tailored insights

## Business Problem
Customer is interest in the use case of Customer Value Management (CVM) to see how AI can improve outbound campaigns and cross-selling efforts to increase conversion rates. Right now most of the campaigns are run digitally but the conversion rate is pretty low so how can Analytics/AI can help them to understand which segment are they targeting, what is the right product for this segment and how can I increase my conversion rate by knowing these metrics. 9Mobile is looking to enhance the impact of the campaigns by getting detailed AI driven insights to target right customer and generate more revenue.

## Business understanding
CVM focuses on maximizing customer lifetime value through personalized engagement, offers, and services. So, the solution should help businesses understand and enhance customer value leveraging machine learning to understand customer segment and behaviour while leveraging generative for personalized insight and campigns.

End-to-End Solution for Customer Value Management (CVM) Using Azure and Generative AI

Objective
Build a scalable, AI-driven CVM system to maximize customer lifetime value (CLV) through personalized engagement, predictive analytics, and generative AI-powered interactions.

## Data Sources
Data Sources: CRM (Dynamics 365), transaction logs, social media, IoT devices.

## Data Understanding
Synthetic data was generated having the folowing features for the Proof of concept.
* monthly bills
* tenure_months
* calls_

### Tech Stack
* Microsoft Fabric- Data Science workloads
* Azure machine learning
* Azure OpenAi
* Power Bi
* Azure Container

## Project Pipeline
* Data Ingestion ---> Data Preprocessing ---> Feature Engineering ---> Clustering ----> Generative Insight ---> model deployment ----> Monitoring and Evaluation