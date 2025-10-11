# -*- coding: utf-8 -*-
"""
House Price Prediction Dashboard
Streamlit Interactive Dashboard for Real Estate Analysis
"""

import streamlit as st
import numpy as np
import pandas as pd

# Set matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Agg')
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

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.title("🏠 House Price Prediction Dashboard")
st.markdown("---")

# Sidebar
st.sidebar.header("⚙️ Navigation")
page = st.sidebar.radio("Select Page:", 
                        ["📊 Data Overview", 
                         "📈 Exploratory Analysis", 
                         "🤖 Model Performance",
                         "🔮 Price Prediction"])

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
            st.metric("Total Records", len(df))
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
            buffer.append(f"Total Rows: {len(df)}")
            buffer.append(f"Total Columns: {len(df.columns)}")
            buffer.append("\nColumn Details:")
            for col in df.columns:
                non_null = df[col].count()
                dtype = df[col].dtype
                buffer.append(f"  • {col}: {dtype} ({non_null} non-null)")
            st.text("\n".join(buffer))
        
        # Missing values
        st.subheader("❓ Missing Values Analysis")
        missing = df.isnull().sum()
        if missing.sum() > 0:
            fig, ax = plt.subplots(figsize=(10, 4))
            missing[missing > 0].plot(kind='bar', ax=ax, color='coral')
            ax.set_title('Missing Values per Column')
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
        ax.hist(df_clean['Price'], bins=30, color='skyblue', edgecolor='black')
        ax.set_xlabel('Price')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of House Prices')
        st.pyplot(fig)
        
        # Correlation Matrix
        st.subheader("🔗 Correlation Matrix")
        numeric_cols = ['Square_Feet', 'Num_Bedrooms', 'Num_Bathrooms',
                       'Num_Floors', 'Year_Built', 'Garage_Size', 
                       'Location_Score', 'Distance_to_Center', 'Price']
        
        fig, ax = plt.subplots(figsize=(12, 8))
        correlation = df_clean[numeric_cols].corr()
        sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=1, fmt='.2f', ax=ax)
        ax.set_title('Correlation Matrix - Housing Variables')
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
                  alpha=0.6, color='coral', edgecolor='black')
        ax.set_xlabel(feature_select)
        ax.set_ylabel('Price')
        ax.set_title(f'{feature_select} vs Price')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # Feature importance (coefficients)
        st.subheader("⭐ Feature Importance (Coefficients)")
        coef_df = pd.DataFrame({
            'Feature': X_train.columns,
            'Coefficient': model.coef_
        }).sort_values('Coefficient', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['green' if x > 0 else 'red' for x in coef_df['Coefficient']]
        ax.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors)
        ax.set_xlabel('Coefficient Value')
        ax.set_title('Feature Coefficients (Impact on Price)')
        ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
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
            ax.scatter(y_test, y_test_pred, alpha=0.6, color='blue', edgecolor='black')
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                   'r--', lw=2, label='Perfect Prediction')
            ax.set_xlabel('Actual Price')
            ax.set_ylabel('Predicted Price')
            ax.set_title('Actual vs Predicted Price (Test Data)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        with col2:
            st.subheader("📉 Residual Plot")
            residuals = y_test - y_test_pred
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(y_test_pred, residuals, alpha=0.6, color='green', edgecolor='black')
            ax.axhline(y=0, color='r', linestyle='--', lw=2)
            ax.set_xlabel('Predicted Price')
            ax.set_ylabel('Residuals')
            ax.set_title('Residual Plot')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        # Residual distribution
        st.subheader("📊 Residual Distribution")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(residuals, bins=30, color='purple', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Residuals')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Residuals')
        ax.axvline(x=0, color='r', linestyle='--', lw=2)
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
        st.markdown("Input the house features to get a price prediction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            square_feet = st.number_input("Square Feet", 
                                         min_value=50, max_value=500, value=150)
            num_bedrooms = st.number_input("Number of Bedrooms", 
                                          min_value=1, max_value=10, value=3)
            num_bathrooms = st.number_input("Number of Bathrooms", 
                                           min_value=1, max_value=10, value=2)
            num_floors = st.number_input("Number of Floors", 
                                        min_value=1, max_value=5, value=2)
            year_built = st.number_input("Year Built", 
                                        min_value=1900, max_value=2025, value=2000)
        
        with col2:
            has_garden = st.selectbox("Has Garden?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            has_pool = st.selectbox("Has Pool?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            garage_size = st.number_input("Garage Size (sq ft)", 
                                         min_value=0, max_value=200, value=40)
            location_score = st.slider("Location Score", 
                                      min_value=0.0, max_value=10.0, value=7.5, step=0.1)
            distance_to_center = st.slider("Distance to Center (km)", 
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
            st.success(f"### 💰 Predicted Price: ${prediction:,.2f}")
            
            # Show input summary
            st.subheader("📋 Input Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Square Feet:** {square_feet}")
                st.write(f"**Bedrooms:** {num_bedrooms}")
                st.write(f"**Bathrooms:** {num_bathrooms}")
                st.write(f"**Floors:** {num_floors}")
            with col2:
                st.write(f"**Year Built:** {year_built}")
                st.write(f"**Has Garden:** {'Yes' if has_garden else 'No'}")
                st.write(f"**Has Pool:** {'Yes' if has_pool else 'No'}")
                st.write(f"**Garage Size:** {garage_size} sq ft")
            with col3:
                st.write(f"**Location Score:** {location_score}/10")
                st.write(f"**Distance to Center:** {distance_to_center} km")
            
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
    "<div style='text-align: center; color: gray;'>"
    "House Price Prediction Dashboard © 2024"
    "</div>",
    unsafe_allow_html=True
)