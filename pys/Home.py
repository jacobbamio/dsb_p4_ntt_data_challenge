import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import os




st.set_page_config(page_title="Home", page_icon="💰")

st.sidebar.success("patata")

# menu = ["Home", "Exploratory Data Analysis", "Machine Learning Model", "About"]

# choice = st.sidebar.selectbox(label = "Menu", options = menu, index = 0)

# if choice == "Home":


st.title("Fixed deposit Predictor")

st.markdown("""In this machine learning project, we are aiming to predict whether clients will invest in a fixed deposit or not. To do this, we will be using various data points related to the clients, such as their education level, current job, marital status, and bank balance, among others.

To start, we will gather a dataset that includes information about clients who have invested in fixed deposits and those who have not. We will then preprocess the data to ensure that it is in a format suitable for analysis, including handling missing values, converting categorical variables into numerical ones, and normalizing numerical data.

After preprocessing the data, we will explore and visualize it to gain insights into the relationships between different variables and the target variable. This exploration may reveal patterns or correlations that can inform the selection of features for the model.

Next, we will select a machine learning algorithm to build a predictive model. There are many algorithms to choose from, such as logistic regression, decision trees, and support vector machines. We will evaluate different algorithms and select the one that performs best on our data.

Once we have selected an algorithm, we can train the model on our data and evaluate its performance using metrics such as accuracy, precision, recall, and F1 score. If the model performs well, we can deploy it to make predictions on new data, such as a new client's information, and use these predictions to inform business decisions around marketing and sales efforts for fixed deposits.""")


st.markdown("""Model""")

