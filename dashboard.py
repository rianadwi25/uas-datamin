# -*- coding: utf-8 -*-
"""
Dashboard Prediksi Harga Rumah
Dashboard Interaktif Streamlit untuk Analisis Real Estate
"""

import streamlit as st
import numpy as np
import pandas as pd

# Set matplotlib backend before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi Halaman
st.set_page_config(
    page_title="Dashboard Prediksi Harga Rumah",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Modern Light Theme
st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Main Background */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem 1rem;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Content Container */
    .block-container {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
    }
    
    /* Title Styling */
    h1 {
        color: #667eea !important;
        font-weight: 700 !important;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    h2 {
        color: #764ba2 !important;
        font-weight: 600 !important;
    }
    
    h3 {
        color: #667eea !important;
        font-weight: 600 !important;
    }
    
    p, span, div {
        color: #2d3748 !important;
    }
    
    /* Metric Cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        color: white !important;
    }
    
    [data-testid="stMetric"] label {
        color: rgba(255, 255, 255, 0.9) !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
    }
    
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: white !important;
        font-weight: 700 !important;
        font-size: 1.8rem !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Radio Buttons */
    .stRadio > label {
        color: white !important;
        font-weight: 600 !important;
    }
    
    /* Input Fields */
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        padding: 0.5rem;
        transition: all 0.3s ease;
        color: #2d3748 !important;
    }
    
    .stNumberInput label,
    .stSelectbox label {
        color: #2d3748 !important;
        font-weight: 500 !important;
    }
    
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2);
    }
    
    /* Slider */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* DataFrames */
    [data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
    }
    
    /* Success/Info/Warning/Error Messages */
    .stSuccess, .stInfo, .stWarning, .stError {
        border-radius: 10px;
        padding: 1rem;
        font-weight: 500;
    }
    
    /* Divider */
    hr {
        margin: 2rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
    }
    
    /* Cards for sections */
    .stat-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        margin: 1rem 0;
    }
    
    /* Text in content */
    .stMarkdown, .stText {
        color: #2d3748 !important;
    }
    
    /* Dataframe text */
    [data-testid="stDataFrame"] {
        color: #2d3748 !important;
    }
    
    /* Sidebar Info Box */
    .css-1544g2n {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 1rem;
        backdrop-filter: blur(10px);
    }
    </style>
    """, unsafe_allow_html=True)

# Judul dengan gradient
st.markdown("""
    <h1 style='text-align: center; font-size: 3rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800;'>
    🏠 Dashboard Prediksi Harga Rumah
    </h1>
    """, unsafe_allow_html=True)
st.markdown("---")

# Sidebar
st.sidebar.markdown("## ⚙️ Navigasi")
page = st.sidebar.radio("Pilih Halaman:", 
                        ["📊 Tampilan Data", 
                         "📈 Analisis Eksplorasi", 
                         "🤖 Performa Model",
                         "🔮 Prediksi Harga"],
                        label_visibility="collapsed")

# Fungsi Load Data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('real_estate_dataset2.csv')
        return df
    except:
        st.error("⚠️ File 'real_estate_dataset2.csv' tidak ditemukan!")
        st.info("Upload file CSV Anda atau letakkan di direktori yang sama dengan script ini.")
        return None

# Fungsi Train Model
@st.cache_resource
def train_model(df):
    X = df[['Square_Feet', 'Num_Bedrooms', 'Num_Bathrooms',
            'Num_Floors', 'Year_Built', 'Has_Garden', 'Has_Pool', 
            'Garage_Size', 'Location_Score', 'Distance_to_Center']]
    y = df['Price']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    return model, X_train, X_test, y_train, y_test, y_train_pred, y_test_pred

# Load data
df = load_data()

if df is not None:
    # Train model
    model, X_train, X_test, y_train, y_test, y_train_pred, y_test_pred = train_model(df)
    
    # HALAMAN 1: TAMPILAN DATA
    if page == "📊 Tampilan Data":
        st.header("📊 Tampilan Data")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Data", f"{len(df):,}")
        with col2:
            st.metric("Jumlah Fitur", len(df.columns) - 1)
        with col3:
            st.metric("Harga Rata-rata", f"Rp{df['Price'].mean():,.0f}")
        with col4:
            st.metric("Data Kosong", df.isnull().sum().sum())
        
        st.markdown("---")
        
        # # Tampilkan dataset
        # st.subheader("📋 Contoh Dataset")
        # st.dataframe(df.head(10), use_container_width=True)
        
        # # Info dataset
        # col1, col2 = st.columns(2)
        
        # with col1:
        #     st.subheader("📊 Ringkasan Statistik")
        #     st.dataframe(df.describe(), use_container_width=True)
        
        # with col2:
        #     st.subheader("🔍 Tipe Data & Informasi")
        #     buffer = []
        #     buffer.append(f"Total Baris: {len(df):,}")
        #     buffer.append(f"Total Kolom: {len(df.columns)}")
        #     buffer.append("\nDetail Kolom:")
        #     for col in df.columns:
        #         non_null = df[col].count()
        #         dtype = df[col].dtype
        #         buffer.append(f"  • {col}: {dtype} ({non_null:,} non-null)")
        #     st.text("\n".join(buffer))
        
        # Missing values
        st.subheader("❓ Analisis Data Kosong")
        missing = df.isnull().sum()
        if missing.sum() > 0:
            fig, ax = plt.subplots(figsize=(10, 4))
            missing[missing > 0].plot(kind='bar', ax=ax, color='#667eea')
            ax.set_title('Data Kosong per Kolom', fontweight='bold')
            ax.set_ylabel('Jumlah')
            st.pyplot(fig)
        else:
            st.success("✅ Tidak ada data kosong dalam dataset!")
    
    # HALAMAN 2: ANALISIS EKSPLORASI
    elif page == "📈 Analisis Eksplorasi":
        st.header("📈 Analisis Data Eksplorasi")
        
        df_clean = df.dropna()
        
        # Distribusi harga
        st.subheader("💰 Distribusi Harga")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(df_clean['Price'], bins=30, color='#667eea', edgecolor='white', alpha=0.8)
        ax.set_xlabel('Harga', fontweight='bold')
        ax.set_ylabel('Frekuensi', fontweight='bold')
        ax.set_title('Distribusi Harga Rumah', fontweight='bold', fontsize=14)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # Matriks Korelasi
        st.subheader("🔗 Matriks Korelasi")
        numeric_cols = ['Square_Feet', 'Num_Bedrooms', 'Num_Bathrooms',
                       'Num_Floors', 'Year_Built', 'Garage_Size', 
                       'Location_Score', 'Distance_to_Center', 'Price']
        
        fig, ax = plt.subplots(figsize=(12, 8))
        correlation = df_clean[numeric_cols].corr()
        sns.heatmap(correlation, annot=True, cmap='RdPu', center=0,
                   square=True, linewidths=1, fmt='.2f', ax=ax,
                   cbar_kws={"shrink": 0.8})
        ax.set_title('Matriks Korelasi - Variabel Rumah', fontweight='bold', fontsize=14)
        st.pyplot(fig)
        
        # Fitur vs Harga
        st.subheader("📊 Fitur vs Harga")
        feature_select = st.selectbox(
            "Pilih fitur untuk dibandingkan dengan Harga:",
            ['Square_Feet', 'Num_Bedrooms', 'Num_Bathrooms',
             'Num_Floors', 'Year_Built', 'Garage_Size', 
             'Location_Score', 'Distance_to_Center']
        )
        
        feature_labels = {
            'Square_Feet': 'Luas (kaki persegi)',
            'Num_Bedrooms': 'Jumlah Kamar Tidur',
            'Num_Bathrooms': 'Jumlah Kamar Mandi',
            'Num_Floors': 'Jumlah Lantai',
            'Year_Built': 'Tahun Dibangun',
            'Garage_Size': 'Ukuran Garasi',
            'Location_Score': 'Skor Lokasi',
            'Distance_to_Center': 'Jarak ke Pusat Kota'
        }
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(df_clean[feature_select], df_clean['Price'], 
                  alpha=0.5, color='#764ba2', edgecolor='white', s=50)
        ax.set_xlabel(feature_labels[feature_select], fontweight='bold')
        ax.set_ylabel('Harga', fontweight='bold')
        ax.set_title(f'{feature_labels[feature_select]} vs Harga', fontweight='bold', fontsize=14)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # Feature importance (coefficients)
        st.subheader("⭐ Tingkat Kepentingan Fitur (Koefisien)")
        coef_df = pd.DataFrame({
            'Fitur': X_train.columns,
            'Koefisien': model.coef_
        }).sort_values('Koefisien', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#667eea' if x > 0 else '#f093fb' for x in coef_df['Koefisien']]
        ax.barh(coef_df['Fitur'], coef_df['Koefisien'], color=colors, alpha=0.8)
        ax.set_xlabel('Nilai Koefisien', fontweight='bold')
        ax.set_title('Koefisien Fitur (Dampak terhadap Harga)', fontweight='bold', fontsize=14)
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax.grid(True, alpha=0.3, axis='x')
        st.pyplot(fig)
    
    # HALAMAN 3: PERFORMA MODEL
    elif page == "🤖 Performa Model":
        st.header("🤖 Performa Model")
        
        # Metrik
        train_r2 = r2_score(y_train, y_train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)
        
        test_r2 = r2_score(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📚 Metrik Training")
            st.metric("Skor R²", f"{train_r2:.4f}")
            st.metric("RMSE", f"Rp{train_rmse:,.2f}")
            st.metric("MAE", f"Rp{train_mae:,.2f}")
        
        with col2:
            st.subheader("🎯 Metrik Testing")
            st.metric("Skor R²", f"{test_r2:.4f}")
            st.metric("RMSE", f"Rp{test_rmse:,.2f}")
            st.metric("MAE", f"Rp{test_mae:,.2f}")
        
        # Interpretasi model
        st.markdown("---")
        st.subheader("💡 Interpretasi Model")
        if test_r2 >= 0.9:
            st.success("🌟 Sangat Baik! Model sangat akurat dalam memprediksi harga rumah.")
        elif test_r2 >= 0.7:
            st.info("👍 Baik! Model cukup baik dalam memprediksi harga rumah.")
        elif test_r2 >= 0.5:
            st.warning("⚠️ Cukup! Model memiliki kemampuan prediksi yang memadai.")
        else:
            st.error("❌ Kurang Baik! Model perlu ditingkatkan.")
        
        # Actual vs Predicted
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Harga Aktual vs Prediksi")
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(y_test, y_test_pred, alpha=0.6, color='#667eea', edgecolor='white', s=50)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                   color='#764ba2', linestyle='--', lw=3, label='Prediksi Sempurna')
            ax.set_xlabel('Harga Aktual', fontweight='bold')
            ax.set_ylabel('Harga Prediksi', fontweight='bold')
            ax.set_title('Harga Aktual vs Prediksi (Data Test)', fontweight='bold', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        with col2:
            st.subheader("📉 Plot Residual")
            residuals = y_test - y_test_pred
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(y_test_pred, residuals, alpha=0.6, color='#f093fb', edgecolor='white', s=50)
            ax.axhline(y=0, color='#764ba2', linestyle='--', lw=3)
            ax.set_xlabel('Harga Prediksi', fontweight='bold')
            ax.set_ylabel('Residual', fontweight='bold')
            ax.set_title('Plot Residual', fontweight='bold', fontsize=12)
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        # Distribusi residual
        st.subheader("📊 Distribusi Residual")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(residuals, bins=30, color='#667eea', alpha=0.7, edgecolor='white')
        ax.set_xlabel('Residual', fontweight='bold')
        ax.set_ylabel('Frekuensi', fontweight='bold')
        ax.set_title('Distribusi Residual', fontweight='bold', fontsize=14)
        ax.axvline(x=0, color='#764ba2', linestyle='--', lw=3)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # Persamaan regresi
        st.markdown("---")
        st.subheader("📐 Persamaan Regresi")
        equation = f"**Harga = {model.intercept_:,.2f}**"
        for feature, coef in zip(X_train.columns, model.coef_):
            sign = "+" if coef >= 0 else "-"
            equation += f" **{sign} ({abs(coef):,.2f} × {feature})**"
        st.markdown(equation)
    
    # HALAMAN 4: PREDIKSI HARGA
    elif page == "🔮 Prediksi Harga":
        st.header("🔮 Prediksi Harga Rumah")
        st.markdown("**Masukkan fitur-fitur rumah untuk mendapatkan prediksi harga**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            square_feet = st.number_input("🏘️ Luas (kaki persegi)", 
                                         min_value=50, max_value=5000, value=1500)
            num_bedrooms = st.number_input("🛏️ Jumlah Kamar Tidur", 
                                          min_value=1, max_value=10, value=3)
            num_bathrooms = st.number_input("🚿 Jumlah Kamar Mandi", 
                                           min_value=1, max_value=10, value=2)
            num_floors = st.number_input("🏢 Jumlah Lantai", 
                                        min_value=1, max_value=5, value=2)
            year_built = st.number_input("📅 Tahun Dibangun", 
                                        min_value=1900, max_value=2025, value=2000)
        
        with col2:
            has_garden = st.selectbox("🌳 Punya Taman?", [0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")
            has_pool = st.selectbox("🏊 Punya Kolam?", [0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")
            garage_size = st.number_input("🚗 Ukuran Garasi (kaki persegi)", 
                                         min_value=0, max_value=1000, value=400)
            location_score = st.slider("📍 Skor Lokasi", 
                                      min_value=0.0, max_value=10.0, value=7.5, step=0.1)
            distance_to_center = st.slider("🗺️ Jarak ke Pusat Kota (km)", 
                                          min_value=0.0, max_value=50.0, value=5.0, step=0.5)
        
        if st.button("🎯 Prediksi Harga", type="primary"):
            input_data = pd.DataFrame({
                'Square_Feet': [square_feet],
                'Num_Bedrooms': [num_bedrooms],
                'Num_Bathrooms': [num_bathrooms],
                'Num_Floors': [num_floors],
                'Year_Built': [year_built],
                'Has_Garden': [has_garden],
                'Has_Pool': [has_pool],
                'Garage_Size': [garage_size],
                'Location_Score': [location_score],
                'Distance_to_Center': [distance_to_center]
            })
            
            prediction = model.predict(input_data)[0]
            
            st.markdown("---")
            st.markdown(f"""
                <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 15px; text-align: center; color: white; box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);'>
                    <h2 style='color: white !important; margin: 0;'>💰 Harga Prediksi</h2>
                    <h1 style='color: white !important; font-size: 3rem; margin: 0.5rem 0;'>Rp{prediction:,.2f}</h1>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Tampilkan ringkasan input
            st.subheader("📋 Ringkasan Input")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**🏘️ Luas:** {square_feet:,} kaki²")
                st.write(f"**🛏️ Kamar Tidur:** {num_bedrooms}")
                st.write(f"**🚿 Kamar Mandi:** {num_bathrooms}")
                st.write(f"**🏢 Lantai:** {num_floors}")
            with col2:
                st.write(f"**📅 Tahun Dibangun:** {year_built}")
                st.write(f"**🌳 Punya Taman:** {'Ya' if has_garden else 'Tidak'}")
                st.write(f"**🏊 Punya Kolam:** {'Ya' if has_pool else 'Tidak'}")
                st.write(f"**🚗 Ukuran Garasi:** {garage_size} kaki²")
            with col3:
                st.write(f"**📍 Skor Lokasi:** {location_score}/10")
                st.write(f"**🗺️ Jarak ke Pusat:** {distance_to_center} km")
            
            # Range harga
            st.markdown("---")
            st.subheader("📊 Estimasi Rentang Harga")
            margin = prediction * 0.1
            st.write(f"**Rentang Estimasi:** Rp{prediction - margin:,.2f} - Rp{prediction + margin:,.2f}")
            st.caption("± 10% margin berdasarkan ketidakpastian model")

# Info sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 📖 Tentang")
st.sidebar.info(
    "Dashboard ini menyediakan analisis komprehensif dan prediksi "
    "harga rumah menggunakan model Linear Regression."
)
st.sidebar.markdown("### 👨‍💻 Developer")
st.sidebar.text("Dibuat dengan ❤️ menggunakan Streamlit")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #667eea; font-weight: 500;'>"
    "Dashboard Prediksi Harga Rumah © 2024 | Powered by Machine Learning"
    "</div>",
    unsafe_allow_html=True
)