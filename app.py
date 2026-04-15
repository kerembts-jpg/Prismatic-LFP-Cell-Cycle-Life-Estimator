import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Page Configuration
st.set_page_config(page_title="AION Battery ML Predictor", page_icon="⚡")

# Logo
try:
    st.image("logo.png", width=200)
except:
    pass

# 1. MODEL TRAINING ON MASSIVE DATASET
@st.cache_resource
def train_model():
    # GitHub'a yüklediğin o 47.000 satırlık YENİ csv dosyası
    df = pd.read_csv('aion_massive_dataset.csv')
    
    X = df[['Resistance_mOhm', 'Test_CRate', 'Test_DoD', 'Test_SoH']]
    y = df['Datasheet_Cycle']
    
    # Veri seti çok büyük olduğu için %10 test verisi yeterli
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    # Model (n_jobs=-1 ile sunucunun tüm gücünü kullanır)
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # MAE (Hata Marjini) Hesaplama
    y_pred = model.predict(X_test)
    mae = int(mean_absolute_error(y_test, y_pred))
    
    return model, mae

# Modeli Yükle
with st.spinner('Loading massive ML dataset...'):
    model, error_margin = train_model()

# 2. USER INTERFACE
st.title("AION Engineering: Battery ML Predictor")
st.markdown("Advanced cycle life estimation using a Random Forest model trained on 47,000+ data points.")

# Sidebar Inputs
st.sidebar.header("Test Parameters")
res = st.sidebar.number_input("Internal Resistance (mΩ)", value=0.50, min_value=0.10, max_value=2.00, step=0.01)
crate = st.sidebar.number_input("Test Current (C-Rate)", value=1.0, min_value=0.1, max_value=2.5, step=0.1)
dod = st.sidebar.slider("DoD (Depth of Discharge %)", 50, 100, 100)
soh = st.sidebar.slider("SoH (Health Target %)", 60, 100, 80)

if st.sidebar.button("Predict Cycle Life"):
    # Tahmin için veriyi hazırla
    input_df = pd.DataFrame([[res, crate, dod, soh]], 
                            columns=['Resistance_mOhm', 'Test_CRate', 'Test_DoD', 'Test_SoH'])
    
    # Pürüzsüz ML Tahmini
    final_pred = model.predict(input_df)[0]
    
    # Alt ve Üst Sınır
    lower = max(0, int(final_pred - error_margin))
    upper = int(final_pred + error_margin)
    
    # SONUÇ
    st.divider()
    st.subheader("Results")
    st.success(f"### 🎯 Estimated Cycle Life: {lower} - {upper} Cycles")
    st.info(f"Model confidently predicted based on {res}mΩ resistance, {crate}C rate, {dod}% DoD, and {soh}% SoH target.")

else:
    st.info("Adjust parameters and click 'Predict' to see the ML estimated cycle life range.")

st.divider()
st.caption("Developed by AION Engineering | Powered by Random Forest ML")
