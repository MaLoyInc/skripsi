import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier

st.title("💳 Prediksi Transaksi Penipuan (Fraud Detection)")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("Clean Dataset/fraudTrain_dataset_cleaned.csv")

df = load_data()

st.markdown("Gunakan aplikasi ini untuk memprediksi apakah suatu transaksi berisiko penipuan.")

# Pisahkan target dan fitur
target_col = 'is_fraud'
categorical_cols = df.select_dtypes(include='object').columns.tolist()
X = df.drop(columns=[target_col])
y = df[target_col]

# Encoder untuk fitur kategorikal
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X_encoded = X.copy()
X_encoded[categorical_cols] = encoder.fit_transform(X[categorical_cols])

# Simpan encoder
with open("ordinal_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Simpan scaler
with open("fraud_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Simpan model
with open("fraud_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Formulir Input
st.subheader("📋 Formulir Data Transaksi")
with st.form("fraud_form"):
    user_input = {}
    for col in X.columns:
        if col in categorical_cols:
            user_input[col] = st.selectbox(f"{col}", df[col].dropna().unique())
        else:
            user_input[col] = st.number_input(f"{col}", value=float(df[col].median()))
    submitted = st.form_submit_button("🔍 Prediksi")

# Prediksi
if submitted:
    input_df = pd.DataFrame([user_input])
    input_df[categorical_cols] = encoder.transform(input_df[categorical_cols])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"🚨 Transaksi ini berpotensi penipuan! (Probabilitas: {prob:.2%})")
    else:
        st.success(f"✅ Transaksi ini aman. (Probabilitas penipuan: {prob:.2%})")
