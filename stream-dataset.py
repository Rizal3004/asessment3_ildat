import joblib
import streamlit as st

import os

model_path = os.path.join('ILDAT', 'diabetes_predict.sav')
diabetes_model = joblib.load(open(model_path, 'rb'))


# Judul aplikasi
st.title('Prediksi Diabetes')

# Input fitur dari pengguna
col1, col2 = st.columns(2)

with col1:
    Pregnancies = st.text_input('Jumlah Kehamilan')
    BloodPressure = st.text_input('Tekanan Darah')
    Insulin = st.text_input('Insulin')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function')

with col2:
    Glucose = st.text_input('Glukosa')
    SkinThickness = st.text_input('Ketebalan Kulit')
    BMI = st.text_input('Indeks Massa Tubuh (BMI)')
    Age = st.text_input('Usia')

# Button prediksi
if st.button('Prediksi Diabetes'):
    # Lakukan prediksi dengan model
    diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])

    # Tampilkan hasil prediksi
    if diab_prediction[0] == 1:
        diab_diagnosis = 'Pasien terkena Diabetes'
    else:
        diab_diagnosis = 'Pasien tidak terkena Diabetes'
    
    st.success(diab_diagnosis)
