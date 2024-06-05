import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Title of the app
st.title('Credit Card Fraud Detection')

# Section to upload the dataset
st.header('Upload Dataset')
uploaded_file = st.file_uploader("./creditcard.csv", type="csv")

if uploaded_file is not None:
    # Load the dataset
    credit_card_data = pd.read_csv(uploaded_file)
    
    # Display dataset information
    st.header('Dataset Information')
    st.write(credit_card_data.head())
    st.write(credit_card_data.tail())
    st.write(credit_card_data.info())
    
    # Checking the number of missing values in each column
    st.subheader('Missing Values')
    st.write(credit_card_data.isnull().sum())
    
    # Distribution of legit transactions & fraudulent transactions
    st.subheader('Class Distribution')
    st.write(credit_card_data['Class'].value_counts())
    
    # Data preprocessing
    st.header('Data Preprocessing')
    legit = credit_card_data[credit_card_data.Class == 0]
    fraud = credit_card_data[credit_card_data.Class == 1]
    
    st.write('Legit transactions:', legit.shape)
    st.write('Fraudulent transactions:', fraud.shape)
    
    # Under-sampling
    st.subheader('Under-Sampling')
    legit_sample = legit.sample(n=492)
    new_dataset = pd.concat([legit_sample, fraud], axis=0)
    st.write(new_dataset['Class'].value_counts())
    
    # Splitting the data into Features & Targets
    X = new_dataset.drop(columns='Class', axis=1)
    Y = new_dataset['Class']
    
    # Split the data into Training data & Testing Data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
    
    st.write('Training data shape:', X_train.shape)
    st.write('Testing data shape:', X_test.shape)
    
    # Model Training
    st.header('Model Training')
    model = LogisticRegression()
    model.fit(X_train, Y_train)
    
    # Model Evaluation
    st.header('Model Evaluation')
    X_train_prediction = model.predict(X_train)
    training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
    st.write('Accuracy on Training data:', training_data_accuracy)
    
    X_test_prediction = model.predict(X_test)
    test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
    st.write('Accuracy score on Test Data:', test_data_accuracy)
