# -*- coding: utf-8 -*-
"""
House Price Prediction Dashboard
Streamlit Interactive Dashboard for Real Estate Analysis
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

# Page Configuration
st.set_page_config(
    page_title="House Price Prediction Dashboard",
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
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
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
    
    h2, h3 {
        color: #764ba2 !important;
        font-weight: 600 !important;
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
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        margin: 1rem 0;
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

# Title with gradient
st.markdown("""
    <h1 style='text-align: center; font-size: 3rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800;'>
    🏠 House Price Prediction Dashboard
    </h1>
    """, unsafe_allow_html=True)
st.markdown("---")

# Sidebar
st.sidebar.markdown("## ⚙️ Navigation")
page = st.sidebar.radio("Select Page:", 
                        ["📊 Data Overview", 
                         "📈 Exploratory Analysis", 
                         "🤖 Model Performance",
                         "🔮 Price Prediction"],
                        label_visibility="collapsed")

# Load Data Function
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('real_estate_dataset2.csv')
        return df
    except:
        st.error("⚠️ File 'real_estate_dataset2.csv' tidak ditemukan!")
        st.info("Upload file CSV Anda atau letakkan di directory yang sama dengan script ini.")
        return None

# Train Model Function
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
    
    # PAGE 1: DATA OVERVIEW
    if page == "📊 Data Overview":
        st.header("📊 Data Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            st.metric("Features", len(df.columns) - 1)
        with col3:
            st.metric("Avg Price", f"${df['Price'].mean():,.0f}")
        with col4:
            st.metric("Missing Values", df.isnull().sum().sum())
        
        st.markdown("---")
        
        # Display dataset
        st.subheader("📋 Dataset Sample")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Dataset info
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Statistical Summary")
            st.dataframe(df.describe(), use_container_width=True)
        
        with col2:
            st.subheader("🔍 Data Types & Info")
            buffer = []
            buffer.append(f"Total Rows: {len(df):,}")
            buffer.append(f"Total Columns: {len(df.columns)}")
            buffer.append("\nColumn Details:")
            for col in df.columns:
                non_null = df[col].count()
                dtype = df[col].dtype
                buffer.append(f"  • {col}: {dtype} ({non_null:,} non-null)")
            st.text("\n".join(buffer))
        
        # Missing values
        st.subheader("❓ Missing Values Analysis")
        missing = df.isnull().sum()
        if missing.sum() > 0:
            fig, ax = plt.subplots(figsize=(10, 4))
            missing[missing > 0].plot(kind='bar', ax=ax, color='#667eea')
            ax.set_title('Missing Values per Column', fontweight='bold')
            ax.set_ylabel('Count')
            st.pyplot(fig)
        else:
            st.success("✅ No missing values in the dataset!")
    
    # PAGE 2: EXPLORATORY ANALYSIS
    elif page == "📈 Exploratory Analysis":
        st.header("📈 Exploratory Data Analysis")
        
        df_clean = df.dropna()
        
        # Price distribution
        st.subheader("💰 Price Distribution")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(df_clean['Price'], bins=30, color='#667eea', edgecolor='white', alpha=0.8)
        ax.set_xlabel('Price', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title('Distribution of House Prices', fontweight='bold', fontsize=14)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # Correlation Matrix
        st.subheader("🔗 Correlation Matrix")
        numeric_cols = ['Square_Feet', 'Num_Bedrooms', 'Num_Bathrooms',
                       'Num_Floors', 'Year_Built', 'Garage_Size', 
                       'Location_Score', 'Distance_to_Center', 'Price']
        
        fig, ax = plt.subplots(figsize=(12, 8))
        correlation = df_clean[numeric_cols].corr()
        sns.heatmap(correlation, annot=True, cmap='RdPu', center=0,
                   square=True, linewidths=1, fmt='.2f', ax=ax,
                   cbar_kws={"shrink": 0.8})
        ax.set_title('Correlation Matrix - Housing Variables', fontweight='bold', fontsize=14)
        st.pyplot(fig)
        
        # Feature vs Price
        st.subheader("📊 Features vs Price")
        feature_select = st.selectbox(
            "Select feature to compare with Price:",
            ['Square_Feet', 'Num_Bedrooms', 'Num_Bathrooms',
             'Num_Floors', 'Year_Built', 'Garage_Size', 
             'Location_Score', 'Distance_to_Center']
        )
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(df_clean[feature_select], df_clean['Price'], 
                  alpha=0.5, color='#764ba2', edgecolor='white', s=50)
        ax.set_xlabel(feature_select, fontweight='bold')
        ax.set_ylabel('Price', fontweight='bold')
        ax.set_title(f'{feature_select} vs Price', fontweight='bold', fontsize=14)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # Feature importance (coefficients)
        st.subheader("⭐ Feature Importance (Coefficients)")
        coef_df = pd.DataFrame({
            'Feature': X_train.columns,
            'Coefficient': model.coef_
        }).sort_values('Coefficient', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#667eea' if x > 0 else '#f093fb' for x in coef_df['Coefficient']]
        ax.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors, alpha=0.8)
        ax.set_xlabel('Coefficient Value', fontweight='bold')
        ax.set_title('Feature Coefficients (Impact on Price)', fontweight='bold', fontsize=14)
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax.grid(True, alpha=0.3, axis='x')
        st.pyplot(fig)
    
    # PAGE 3: MODEL PERFORMANCE
    elif page == "🤖 Model Performance":
        st.header("🤖 Model Performance")
        
        # Metrics
        train_r2 = r2_score(y_train, y_train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)
        
        test_r2 = r2_score(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📚 Training Metrics")
            st.metric("R² Score", f"{train_r2:.4f}")
            st.metric("RMSE", f"${train_rmse:,.2f}")
            st.metric("MAE", f"${train_mae:,.2f}")
        
        with col2:
            st.subheader("🎯 Testing Metrics")
            st.metric("R² Score", f"{test_r2:.4f}")
            st.metric("RMSE", f"${test_rmse:,.2f}")
            st.metric("MAE", f"${test_mae:,.2f}")
        
        # Model interpretation
        st.markdown("---")
        st.subheader("💡 Model Interpretation")
        if test_r2 >= 0.9:
            st.success("🌟 Excellent! Model sangat baik dalam memprediksi harga rumah.")
        elif test_r2 >= 0.7:
            st.info("👍 Good! Model cukup baik dalam memprediksi harga rumah.")
        elif test_r2 >= 0.5:
            st.warning("⚠️ Fair! Model memiliki kemampuan prediksi yang cukup.")
        else:
            st.error("❌ Poor! Model kurang baik, perlu improvement.")
        
        # Actual vs Predicted
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Actual vs Predicted")
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(y_test, y_test_pred, alpha=0.6, color='#667eea', edgecolor='white', s=50)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                   color='#764ba2', linestyle='--', lw=3, label='Perfect Prediction')
            ax.set_xlabel('Actual Price', fontweight='bold')
            ax.set_ylabel('Predicted Price', fontweight='bold')
            ax.set_title('Actual vs Predicted Price (Test Data)', fontweight='bold', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        with col2:
            st.subheader("📉 Residual Plot")
            residuals = y_test - y_test_pred
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(y_test_pred, residuals, alpha=0.6, color='#f093fb', edgecolor='white', s=50)
            ax.axhline(y=0, color='#764ba2', linestyle='--', lw=3)
            ax.set_xlabel('Predicted Price', fontweight='bold')
            ax.set_ylabel('Residuals', fontweight='bold')
            ax.set_title('Residual Plot', fontweight='bold', fontsize=12)
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        # Residual distribution
        st.subheader("📊 Residual Distribution")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(residuals, bins=30, color='#667eea', alpha=0.7, edgecolor='white')
        ax.set_xlabel('Residuals', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title('Distribution of Residuals', fontweight='bold', fontsize=14)
        ax.axvline(x=0, color='#764ba2', linestyle='--', lw=3)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # Regression equation
        st.markdown("---")
        st.subheader("📐 Regression Equation")
        equation = f"**Price = {model.intercept_:,.2f}**"
        for feature, coef in zip(X_train.columns, model.coef_):
            sign = "+" if coef >= 0 else "-"
            equation += f" **{sign} ({abs(coef):,.2f} × {feature})**"
        st.markdown(equation)
    
    # PAGE 4: PRICE PREDICTION
    elif page == "🔮 Price Prediction":
        st.header("🔮 House Price Prediction")
        st.markdown("**Input the house features to get a price prediction**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            square_feet = st.number_input("🏘️ Square Feet", 
                                         min_value=50, max_value=5000, value=1500)
            num_bedrooms = st.number_input("🛏️ Number of Bedrooms", 
                                          min_value=1, max_value=10, value=3)
            num_bathrooms = st.number_input("🚿 Number of Bathrooms", 
                                           min_value=1, max_value=10, value=2)
            num_floors = st.number_input("🏢 Number of Floors", 
                                        min_value=1, max_value=5, value=2)
            year_built = st.number_input("📅 Year Built", 
                                        min_value=1900, max_value=2025, value=2000)
        
        with col2:
            has_garden = st.selectbox("🌳 Has Garden?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            has_pool = st.selectbox("🏊 Has Pool?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            garage_size = st.number_input("🚗 Garage Size (sq ft)", 
                                         min_value=0, max_value=1000, value=400)
            location_score = st.slider("📍 Location Score", 
                                      min_value=0.0, max_value=10.0, value=7.5, step=0.1)
            distance_to_center = st.slider("🗺️ Distance to Center (km)", 
                                          min_value=0.0, max_value=50.0, value=5.0, step=0.5)
        
        if st.button("🎯 Predict Price", type="primary"):
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
                    <h2 style='color: white !important; margin: 0;'>💰 Predicted Price</h2>
                    <h1 style='color: white !important; font-size: 3rem; margin: 0.5rem 0;'>${prediction:,.2f}</h1>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Show input summary
            st.subheader("📋 Input Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**🏘️ Square Feet:** {square_feet:,}")
                st.write(f"**🛏️ Bedrooms:** {num_bedrooms}")
                st.write(f"**🚿 Bathrooms:** {num_bathrooms}")
                st.write(f"**🏢 Floors:** {num_floors}")
            with col2:
                st.write(f"**📅 Year Built:** {year_built}")
                st.write(f"**🌳 Has Garden:** {'Yes' if has_garden else 'No'}")
                st.write(f"**🏊 Has Pool:** {'Yes' if has_pool else 'No'}")
                st.write(f"**🚗 Garage Size:** {garage_size} sq ft")
            with col3:
                st.write(f"**📍 Location Score:** {location_score}/10")
                st.write(f"**🗺️ Distance to Center:** {distance_to_center} km")
            
            # Price range
            st.markdown("---")
            st.subheader("📊 Price Range Estimation")
            margin = prediction * 0.1
            st.write(f"**Estimated Range:** ${prediction - margin:,.2f} - ${prediction + margin:,.2f}")
            st.caption("± 10% margin based on model uncertainty")

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.markdown("### 📖 About")
st.sidebar.info(
    "This dashboard provides comprehensive analysis and prediction "
    "of house prices using Linear Regression model."
)
st.sidebar.markdown("### 👨‍💻 Developer")
st.sidebar.text("Created with ❤️ using Streamlit")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #667eea; font-weight: 500;'>"
    "House Price Prediction Dashboard © 2024 | Powered by Machine Learning"
    "</div>",
    unsafe_allow_html=True
)