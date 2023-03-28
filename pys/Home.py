import streamlit as st
import streamlit.components.v1 as components
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import os
from PIL import Image

st.set_page_config(page_title="Home", page_icon="💰", )

selected = option_menu(menu_title=None,
                       options=["Home", "Visualization", "Model"],
                       default_index=0,
                       icons=["house","clipboard-data","cloud-fill"],
                       orientation="horizontal")

# menu = ["Home", "Exploratory Data Analysis", "Machine Learning Model", "About"]

# choice = st.sidebar.selectbox(label = "Menu", options = menu, index = 0)

# if choice == "Home":

if selected == "Home":

    st.title("Introduction")

    st.markdown("""
    
    
    In this machine learning project, we are aiming to predict whether clients will invest in a fixed deposit or not. To do this, we will be using various data points related to the clients, such as their education level, current job, marital status, and bank balance, among others.

    To start, we will gather a dataset that includes information about clients who have invested in fixed deposits and those who have not. We will then preprocess the data to ensure that it is in a format suitable for analysis, including handling missing values, converting categorical variables into numerical ones, and normalizing numerical data.

    After preprocessing the data, we will explore and visualize it to gain insights into the relationships between different variables and the target variable. This exploration may reveal patterns or correlations that can inform the selection of features for the model.

    Next, we will select a machine learning algorithm to build a predictive model. There are many algorithms to choose from, such as logistic regression, decision trees, and support vector machines. We will evaluate different algorithms and select the one that performs best on our data.

    Once we have selected an algorithm, we can train the model on our data and evaluate its performance using metrics such as accuracy, precision, recall, and F1 score. If the model performs well, we can deploy it to make predictions on new data, such as a new client's information, and use these predictions to inform business decisions around marketing and sales efforts for fixed deposits.
    
    """)

    st.title("Objective")

    st.title("Source")

    st.title("Tecnologies involved")


elif selected == "Visualization":

    st.title("Power BI")

    components.html('<iframe title="Visualizaciones" width="700" height="455" src="https://app.powerbi.com/view?r=eyJrIjoiZjEzMjA0YzAtNTUwYi00Y2Y0LTliODEtZThkZjIyOTdlYTFhIiwidCI6ImY5N2FjMzMyLWQ2NzktNDFkMS1hNWIzLTU1MjgyNTdlMGQ3ZSIsImMiOjh9" frameborder="0" allowFullScreen="true"></iframe>', height=1024)
    
    st.title("Exploratory Data Analysis")

else:

    st.title("Pipelines")

    bdt_image                 = Image.open("../resources/azure_snapshots/boosted_decision_tree.png")
    full_pipeline_image       = Image.open("../resources/azure_snapshots/full_pipeline.png")
    prediction_pipeline_image = Image.open("../resources/azure_snapshots/prediction_pipeline.png")

    st.image(image = full_pipeline_image,       caption = "Azure Pipeline With 5 Different Models")
    st.image(image = prediction_pipeline_image, caption = "Azure Pipeline Structured For Deployment")
    st.image(image = bdt_image,                 caption = "Boosted Decision Tree Metrics")

    df = pd.read_csv("../resources/metricas.csv")
    
    st.dataframe(df)

    st.title("Example clients")



