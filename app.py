import streamlit as st
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split


data = pd.read_csv('Cleaned car.csv')

lr = pickle.load(open('LinearRegressionModel.pkl', 'rb'))

st.write("# 🚗 Welcome to Car Predictor 🚗")

st.markdown("<br>", unsafe_allow_html=True)

st.write("This app predicts the price of a car you wan to sell. Try filling the details below")

st.markdown("<br>", unsafe_allow_html=True)


# def car_predictor():

# col1, col2, col3, col4, col5 = st.columns(5)

companies = sorted(data['company'].unique())
car_models = sorted(data['name'].unique())
year = sorted(data['year'].unique(), reverse=True)
fuel_type = sorted(data['fuel_type'].unique())
km_drive = sorted(data['kms_driven'].unique())


    # def predictor
# with col1:

# st.header("Enter Car company")
st.write("*Search for Companies*")
company = st.selectbox(" ",companies)
company = str(company)

st.markdown("<br>", unsafe_allow_html=True)

# with col2:
st.write("*Serch for name of car*")
name = st.selectbox(" ",car_models)
name = str(name)

# with col3:
st.write("*Search for year*")
car_year = st.selectbox(" ",year)

# with col4:
st.write("*Fuel Type*")
fuel = st.selectbox(" ",fuel_type)
fuel = str(fuel)

# with col5:
st.write("*km to be driven*")
km = st.selectbox(" ",km_drive)


if st.button("predict"):
    price_val = lr.predict(pd.DataFrame([[name, company, car_year, km, fuel]],
                                        columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']))
    price_val = int(price_val)

    st.write(f" The price of {name} is : ")
    st.write(" ## ", price_val)







