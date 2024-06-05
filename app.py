import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

# Load the dataset
def load_data():
    df = pd.read_csv('./creditcard.csv')  # Replace with your dataset path
    return df

# Train model
def train_model(df):
    X = df.drop('Class', axis=1)
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    return y_test, y_pred

# Main Streamlit App
def main():
    st.title('Credit Card Fraud Detection')
    
    # Load data
    df = load_data()
    
    # Show dataset
    st.subheader('Dataset')
    st.write(df.head())
    
    # Train model and get predictions
    y_test, y_pred = train_model(df)
    
    # Show metrics
    st.subheader('Confusion Matrix')
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", ax=ax)
    st.pyplot(fig)
    
    st.subheader('Classification Report')
    report = classification_report(y_test, y_pred, output_dict=True)
    st.write(report)
    
    # Additional visualization
    st.subheader('Data Distribution')
    fig, ax = plt.subplots()
    sns.countplot(x='Class', data=df, ax=ax)
    st.pyplot(fig)

if __name__ == '__main__':
    main()
