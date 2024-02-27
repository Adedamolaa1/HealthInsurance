import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
import seaborn as sns
data = pd.read_csv('insurance.csv')
df = data.copy()

scaler = StandardScaler()
encoder = LabelEncoder()

for column in df.select_dtypes(include = ['object']):
    df[column] = encoder.fit_transform(df[column])

st.markdown("<h1 style = 'text-align: center; font-family: helvetica; color: #1F4172;>INSURANCE FORECAST PROJECT</h1>", unsafe_allow_html=True)
st.markdown("<h4 style = 'margin: -30px; color: #F11A7B; text-align: center; font-family: cursive'>BUILT BY AHMED --CODE</h4>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
st.image('pngwing.com (11).png', width = 360, use_column_width=True)
st.markdown("<br>", unsafe_allow_html= True)
st.markdown("<p> The Regression Prediction of Health Insurance Forecast Project aims to predict medical insurance costs for individuals based on various independent variables. Using regression analysis, the project seeks to model the relationship between factors such as age, BMI, smoking status, and geographic region with medical charges. By leveraging machine learning techniques, the project aims to develop an accurate predictive model to assist insurers in estimating healthcare expenses and individuals in planning for medical costs. The project involves data preprocessing, model training, and evaluation to ensure robust predictions. Ultimately, it addresses the need for precise forecasting in the healthcare industry to facilitate informed decision-making and cost-effective planning.</p>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
st.dataframe(data, use_container_width=True)

st.sidebar.image('pngwing.com (12).png', caption = 'Welcome Enrollee!')
st.write('Feature Input')

enrollee_age = st.sidebar.number_input('Age', data['age'].min(), data['age'].max())
gender = st.sidebar.selectbox('Gender', data['sex'].unique())
enrollee_bmi = st.sidebar.number_input('Body_Mass_Index', data['bmi'].min(), data['bmi'].max())
children = st.sidebar.number_input('No_of_Children', data['children'].min(), data['children'].max())
smokers = st.sidebar.selectbox('Smokers', data['smoker'].unique())
Regions = st.sidebar.selectbox('Region', data['region'].unique())

try:
    new_gender = encoder.transform([gender])
    new_smokers = encoder.transform([smokers])
    new_regions = encoder.transform([Regions])
except ValueError:
    new_gender = 0
    new_smokers = 0
    new_regions = 0

new_input = pd.DataFrame({'age':[enrollee_age], 'sex':[new_gender], 'bmi': [enrollee_bmi], 'children':[children],
                          'smoker': [new_smokers], 'region': [new_regions] })

st.dataframe(new_input)

model = joblib.load('rff.pkl')
Predictor = st.button('Press to predict')

if Predictor:
    prediction = model.predict(new_input)
    st.success(f"The predicted medical cost for your insurance is {prediction}")
    st.balloons()




