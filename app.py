import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Page Configuration
st.set_page_config(page_title="AION LFP Cell Cycle Life Estimator", page_icon="⚡")


try:
    st.image("logo.png", width=400) 
except:
    
    pass

# 1. MODEL TRAINING (Optimized with Cache)
@st.cache_resource
def train_model():
    
    df = pd.read_csv('aion_battery_tests.csv')
    X = df[['Resistance_mOhm', 'Test_CRate', 'Test_DoD', 'Test_SoH']]
    y = df['Datasheet_Cycle']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Calculate Mean Absolute Error (MAE)
    y_pred = model.predict(X_test)
    mae = int(mean_absolute_error(y_test, y_pred))
    return model, mae

model, error_margin = train_model()

# 2. USER INTERFACE
st.title("AION Engineering: LFP Cycle Life Estimator")
st.markdown("Predict battery cycle life based on specific test conditions and internal resistance.")

# Sidebar Inputs
st.sidebar.header("Input Parameters")
res = st.sidebar.number_input("Internal Resistance (mΩ)", value=0.50, min_value=0.15, max_value=1.00, step=0.05)
crate = st.sidebar.number_input("Test Current (C-Rate)", value=1.0, min_value=0.5, max_value=1.0, step=0.5)
dod = st.sidebar.number_input("DoD (Depth of Discharge %)", value=100.0, min_value=80.0, max_value=100.0, step=5.0)
soh = st.sidebar.number_input("SoH (Health Target %)", value=80.0, min_value=70.0, max_value=80.0, step=5.0)

if st.sidebar.button("Predict"):
    # Prepare Input
    input_df = pd.DataFrame([[res, crate, dod, soh]], 
                            columns=['Resistance_mOhm', 'Test_CRate', 'Test_DoD', 'Test_SoH'])
    
    # ML Prediction
    raw_pred = model.predict(input_df)[0]
    
    # Resistance Correction Rule: 0.15 -> +1%, 0.5 -> 0%, 1.0 -> -1.5%
    modifier = np.interp(res, [0.15, 0.5, 1.0], [0.10, 0.0, -0.15])
    final_pred = raw_pred * (1.0 + modifier)
    
    # Calculate Range
    lower = max(0, int(final_pred - final_pred*0.04))
    upper = int(final_pred + final_pred*0.04)
    
    # SIMPLIFIED OUTPUT
    st.divider()
    st.subheader("Results")
    st.success(f"### 🎯 Estimated Cycle Life: {lower} - {upper} Cycles")
    st.caption(f"Based on {res}mΩ resistance, {crate}C rate, {dod}% DoD, and {soh}% SoH target.")

else:
    st.info("Adjust the parameters on the sidebar and click 'Predict'.")

st.divider()
st.caption("Developed by AION Engineering")
