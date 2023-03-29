import streamlit as st
import streamlit.components.v1 as components
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import os
from PIL import Image
import functions
import joblib

st.set_page_config(page_title="Home", page_icon="💰", )

if 'scaler' not in st.session_state:
    st.session_state.scaler = joblib.load("../resources/x_scaler.pkl")

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

    st.markdown("""The objective of this machine learning project is to predict whether clients will invest in a fixed deposit or not, using various data points related to the clients such as education level, job, marital 
    status, and bank balance, among others. The project involves gathering and preprocessing the data, exploring and visualizing it to gain insights, selecting a suitable machine learning algorithm, training and evaluating 
    the model, and ultimately using the model to make predictions on new data. The aim is to inform business decisions around marketing and sales efforts for fixed deposits based on the model's predictions.""")

    st.title("Source")

    st.markdown("""The source of this project was provided by NTT data as a group test during a masterclass. We decided to go for it not just becouse it was a good practice but to prove ourselves that we are capable of doing a real world machine learning task.""")

    st.title("Tecnologies involved")

    col_1, col_2 = st.columns(2)

    col_1.markdown('🐍 Python')
    col_1.markdown('🐼 Pandas')
    col_1.markdown('🤖 Scikitlearn')
    col_1.markdown('☁️ Azure Machine Learning Studio')
    col_2.markdown('〽️ Power BI')
    col_2.markdown('🍃 MongoDB')
    col_2.markdown('🌌 Azure Cosmos DB')
    col_2.markdown('👑 Streamlit')


elif selected == "Visualization":

    st.title("Power BI")

    components.html('<iframe title="Visualizaciones" width="700" height="455" src="https://app.powerbi.com/view?r=eyJrIjoiZjEzMjA0YzAtNTUwYi00Y2Y0LTliODEtZThkZjIyOTdlYTFhIiwidCI6ImY5N2FjMzMyLWQ2NzktNDFkMS1hNWIzLTU1MjgyNTdlMGQ3ZSIsImMiOjh9" frameborder="0" allowFullScreen="true"></iframe>', height=1024)
    
    st.title("Exploratory Data Analysis")

else:
    st.title("Pipelines")

    st.markdown("""Once the Data was cleaned, we started trying different algorithms to see which one fitted for our data. """)

    df = pd.read_csv("../resources/metricas.csv")
    
    with st.expander(label = "Models tried", expanded = False):
        st.dataframe(df)

    bdt_image                 = Image.open("../resources/azure_snapshots/boosted_decision_tree.png")
    full_pipeline_image       = Image.open("../resources/azure_snapshots/full_pipeline.png")
    prediction_pipeline_image = Image.open("../resources/azure_snapshots/prediction_pipeline.png")


    st.markdown("""Then we got to versions of the model, one done by coding and the other one using Azure machine learning studio. You will be able to check either of them below.
    Here an schema of the models trial:""")

    with st.expander(label = "Azure in action", expanded = False):
        st.image(image = full_pipeline_image,       caption = "Azure Pipeline With 5 Different Models")
    
    st.markdown("""After that, it was time to try it with some predictions. here a guide of how it works:""")
    
    with st.expander(label = "Get the prediction", expanded = False):
        st.image(image = prediction_pipeline_image, caption = "Azure Pipeline Structured For Deployment")

    st.markdown("""Only one thing left, get the metrics to check how our model works, it turned out pretty well:""")

    with st.expander(label = "Metrics", expanded = False):    
        st.image(image = bdt_image,                 caption = "Boosted Decision Tree Metrics")

    st.markdown("""There is a lot of work in betwen those steps like cleaning the data, decidig what to do with nans and outliers or validating and tuning the model but we only wanted to show you the best part""")

    title, models = st.columns(2)

    title.title("Model selection")

    rb_models = models.radio(" ", ["Handmade model", "Azure model"])

    st.title("Example clients")

    clients_col_1,clients_col_2,clients_col_3,clients_col_4,clients_col_5 = st.columns(5)

    btc_c1  = clients_col_1.button(label = "Client 1")
    btc_c2  = clients_col_2.button(label = "Client 2")
    btc_c3  = clients_col_3.button(label = "Client 3")
    btc_c4  = clients_col_4.button(label = "Client 4")
    btc_c5  = clients_col_5.button(label = "Client 5")
    btc_c6  = clients_col_1.button(label = "Client 6")
    btc_c7  = clients_col_2.button(label = "Client 7")
    btc_c8  = clients_col_3.button(label = "Client 8")
    btc_c9  = clients_col_4.button(label = "Client 9")
    btc_c10 = clients_col_5.button(label = "Client 10")


    if rb_models == "Handmade model":

        if btc_c1:

            model_input  = functions.cosmos_request(1)

            if model_input == None:

                st.warning("Try different clients slower, or the API won't provide more results")

            else:
                
                scaled_input = functions.scale_model_input(model_input, st.session_state.scaler)
                result       = functions.handmade_model_request(scaled_input)

                if "n\'t" in result:

                    st.error(result)
                
                else:

                    st.success(result)

        elif btc_c2:

            model_input  = functions.cosmos_request(2)

            if model_input == None:

                st.warning("Try different clients slower, or the API won't provide more results")

            else:
            
                scaled_input = functions.scale_model_input(model_input, st.session_state.scaler)
                result       = functions.handmade_model_request(scaled_input)

                if "n\'t" in result:

                    st.error(result)
                
                else:

                    st.success(result)

        elif btc_c3:

            model_input  = functions.cosmos_request(3)

            if model_input == None:

                st.warning("Try different clients slower, or the API won't provide more results")

            else:
            
                scaled_input = functions.scale_model_input(model_input, st.session_state.scaler)
                result       = functions.handmade_model_request(scaled_input)

                if "n\'t" in result:

                    st.error(result)
                
                else:

                    st.success(result)

        elif btc_c4:

            model_input  = functions.cosmos_request(4)

            if model_input == None:

                st.warning("Try different clients slower, or the API won't provide more results")

            else:
            
                scaled_input = functions.scale_model_input(model_input, st.session_state.scaler)
                result       = functions.handmade_model_request(scaled_input)

                if "n\'t" in result:

                    st.error(result)
                
                else:

                    st.success(result)

        elif btc_c5:

            model_input  = functions.cosmos_request(5)

            if model_input == None:

                st.warning("Try different clients slower, or the API won't provide more results")

            else:
            
                scaled_input = functions.scale_model_input(model_input, st.session_state.scaler)
                result       = functions.handmade_model_request(scaled_input)

                if "n\'t" in result:

                    st.error(result)
                
                else:

                    st.success(result)

        elif btc_c6:

            model_input  = functions.cosmos_request(6)

            if model_input == None:

                st.warning("Try different clients slower, or the API won't provide more results")

            else:
            
                scaled_input = functions.scale_model_input(model_input, st.session_state.scaler)
                result       = functions.handmade_model_request(scaled_input)

                if "n\'t" in result:

                    st.error(result)
                
                else:

                    st.success(result)

        elif btc_c7:

            model_input  = functions.cosmos_request(7)

            if model_input == None:

                st.warning("Try different clients slower, or the API won't provide more results")

            else:
            
                scaled_input = functions.scale_model_input(model_input, st.session_state.scaler)
                result       = functions.handmade_model_request(scaled_input)

                if "n\'t" in result:

                    st.error(result)
                
                else:

                    st.success(result)

        elif btc_c8:

            model_input  = functions.cosmos_request(8)

            if model_input == None:

                st.warning("Try different clients slower, or the API won't provide more results")

            else:
            
                scaled_input = functions.scale_model_input(model_input, st.session_state.scaler)
                result       = functions.handmade_model_request(scaled_input)

                if "n\'t" in result:

                    st.error(result)
                
                else:

                    st.success(result)

        elif btc_c9:

            model_input  = functions.cosmos_request(9)

            if model_input == None:

                st.warning("Try different clients slower, or the API won't provide more results")

            else:

                scaled_input = functions.scale_model_input(model_input, st.session_state.scaler)
                result       = functions.handmade_model_request(scaled_input)

                if "n\'t" in result:

                    st.error(result)
                
                else:

                    st.success(result)

        elif btc_c10:

            model_input  = functions.cosmos_request(10)

            if model_input == None:

                st.warning("Try different clients slower, or the API won't provide more results")

            else:
            
                scaled_input = functions.scale_model_input(model_input, st.session_state.scaler)
                result       = functions.handmade_model_request(scaled_input)

                if "n\'t" in result:

                    st.error(result)
                
                else:

                    st.success(result)

    elif rb_models == "Azure model":

        if btc_c1:

            model_input  = functions.cosmos_request(1)

            if model_input == None:

                st.warning("Try different clients slower, or the API won't provide more results")

            else:
                
                scaled_input = functions.scale_model_input(model_input, st.session_state.scaler)
                result       = functions.mls_model_request(scaled_input)

                if "n\'t" in result:

                    st.error(result)
                
                else:

                    st.success(result)

        elif btc_c2:

            model_input  = functions.cosmos_request(2)

            if model_input == None:

                st.warning("Try different clients slower, or the API won't provide more results")

            else:
            
                scaled_input = functions.scale_model_input(model_input, st.session_state.scaler)
                result       = functions.mls_model_request(scaled_input)

                if "n\'t" in result:

                    st.error(result)
                
                else:

                    st.success(result)

        elif btc_c3:

            model_input  = functions.cosmos_request(3)

            if model_input == None:

                st.warning("Try different clients slower, or the API won't provide more results")

            else:
            
                scaled_input = functions.scale_model_input(model_input, st.session_state.scaler)
                result       = functions.mls_model_request(scaled_input)

                if "n\'t" in result:

                    st.error(result)
                
                else:

                    st.success(result)

        elif btc_c4:

            model_input  = functions.cosmos_request(4)

            if model_input == None:

                st.warning("Try different clients slower, or the API won't provide more results")

            else:
            
                scaled_input = functions.scale_model_input(model_input, st.session_state.scaler)
                result       = functions.mls_model_request(scaled_input)

                if "n\'t" in result:

                    st.error(result)
                
                else:

                    st.success(result)

        elif btc_c5:

            model_input  = functions.cosmos_request(5)

            if model_input == None:

                st.warning("Try different clients slower, or the API won't provide more results")

            else:
            
                scaled_input = functions.scale_model_input(model_input, st.session_state.scaler)
                result       = functions.mls_model_request(scaled_input)

                if "n\'t" in result:

                    st.error(result)
                
                else:

                    st.success(result)

        elif btc_c6:

            model_input  = functions.cosmos_request(6)

            if model_input == None:

                st.warning("Try different clients slower, or the API won't provide more results")

            else:
            
                scaled_input = functions.scale_model_input(model_input, st.session_state.scaler)
                result       = functions.mls_model_request(scaled_input)

                if "n\'t" in result:

                    st.error(result)
                
                else:

                    st.success(result)

        elif btc_c7:

            model_input  = functions.cosmos_request(7)

            if model_input == None:

                st.warning("Try different clients slower, or the API won't provide more results")

            else:
            
                scaled_input = functions.scale_model_input(model_input, st.session_state.scaler)
                result       = functions.mls_model_request(scaled_input)

                if "n\'t" in result:

                    st.error(result)
                
                else:

                    st.success(result)

        elif btc_c8:

            model_input  = functions.cosmos_request(8)

            if model_input == None:

                st.warning("Try different clients slower, or the API won't provide more results")

            else:
            
                scaled_input = functions.scale_model_input(model_input, st.session_state.scaler)
                result       = functions.mls_model_request(scaled_input)

                if "n\'t" in result:

                    st.error(result)
                
                else:

                    st.success(result)

        elif btc_c9:

            model_input  = functions.cosmos_request(9)

            if model_input == None:

                st.warning("Try different clients slower, or the API won't provide more results")

            else:

                scaled_input = functions.scale_model_input(model_input, st.session_state.scaler)
                result       = functions.mls_model_request(scaled_input)

                if "n\'t" in result:

                    st.error(result)
                
                else:

                    st.success(result)

        elif btc_c10:

            model_input  = functions.cosmos_request(10)

            if model_input == None:

                st.warning("Try different clients slower, or the API won't provide more results")

            else:
            
                scaled_input = functions.scale_model_input(model_input, st.session_state.scaler)
                result       = functions.mls_model_request(scaled_input)

                if "n\'t" in result:

                    st.error(result)
                
                else:

                    st.success(result)





