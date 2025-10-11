# 🏠 House Price Prediction Dashboard

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.43.1-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

Dashboard interaktif untuk analisis dan prediksi harga rumah menggunakan Machine Learning dengan algoritma Linear Regression.

## 🚀 Live Demo

🔗 **[Lihat Dashboard Live](https://your-app-url.streamlit.app)** *(Ganti dengan URL Streamlit Cloud Anda setelah deploy)*

## 📸 Screenshots

### Data Overview
![Data Overview](https://via.placeholder.com/800x400/4A90E2/ffffff?text=Data+Overview+Page)

### Model Performance
![Model Performance](https://via.placeholder.com/800x400/50C878/ffffff?text=Model+Performance+Page)

### Price Prediction
![Price Prediction](https://via.placeholder.com/800x400/FF6B6B/ffffff?text=Price+Prediction+Page)

## 📊 Features

### 1. 📋 Data Overview
- Statistik dataset lengkap (total records, features, average price)
- Preview dataset dengan pagination
- Analisis missing values
- Statistical summary dan data types info

### 2. 📈 Exploratory Data Analysis
- Visualisasi distribusi harga rumah
- Correlation matrix dengan heatmap interaktif
- Scatter plots untuk setiap feature vs Price
- Feature importance analysis berdasarkan koefisien regresi

### 3. 🤖 Model Performance
- Metrics evaluasi model (R², RMSE, MAE)
- Perbandingan performa Training vs Testing data
- Visualisasi Actual vs Predicted prices
- Residual plot untuk analisis error
- Distribusi residual
- Persamaan regresi lengkap

### 4. 🔮 Price Prediction
- Input interaktif untuk semua features rumah
- Prediksi harga real-time
- Summary input lengkap
- Estimasi range harga dengan confidence interval

## 🛠️ Tech Stack

- **Python 3.12** - Programming Language
- **Streamlit** - Web Dashboard Framework
- **Scikit-learn** - Machine Learning Library
- **Pandas** - Data Manipulation
- **NumPy** - Numerical Computing
- **Matplotlib & Seaborn** - Data Visualization

## 📦 Dataset

Dataset berisi informasi rumah dengan fitur-fitur berikut:

| Feature | Description | Type |
|---------|-------------|------|
| `Square_Feet` | Luas bangunan (sqft) | Numeric |
| `Num_Bedrooms` | Jumlah kamar tidur | Integer |
| `Num_Bathrooms` | Jumlah kamar mandi | Integer |
| `Num_Floors` | Jumlah lantai | Integer |
| `Year_Built` | Tahun dibangun | Integer |
| `Has_Garden` | Memiliki taman (0/1) | Binary |
| `Has_Pool` | Memiliki kolam renang (0/1) | Binary |
| `Garage_Size` | Ukuran garasi (sqft) | Numeric |
| `Location_Score` | Skor lokasi (0-10) | Numeric |
| `Distance_to_Center` | Jarak ke pusat kota (km) | Numeric |
| `Price` | Harga rumah (target) | Numeric |

## 🚀 Installation & Usage

### Prerequisites

- Python 3.12 atau lebih tinggi
- pip atau conda package manager

### Local Development

1. **Clone repository**
```bash
git clone https://github.com/username/uas-datamin.git
cd uas-datamin
```

2. **Create virtual environment (optional but recommended)**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the dashboard**
```bash
python -m streamlit run dashboard.py
```

5. **Open browser**
```
http://localhost:8501
```

## 📁 Project Structure

```
uas-datamin/
│
├── dashboard.py                    # Main Streamlit dashboard
├── uas_data_min.py                 # Original analysis script
├── real_estate_dataset2.csv        # Dataset
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── .gitignore                      # Git ignore file
│
└── assets/                         # (Optional) Screenshots folder
    ├── screenshot1.png
    └── screenshot2.png
```

## 📈 Model Performance

Model Linear Regression yang digunakan memiliki performa:

- **R² Score (Test):** ~0.XX (akan bervariasi tergantung data)
- **RMSE (Test):** $XX,XXX
- **MAE (Test):** $XX,XXX

> *Note: Nilai aktual akan ditampilkan di dashboard setelah model dilatih dengan dataset*

## 🎯 Model Features

Model menggunakan 10 features untuk memprediksi harga rumah:

1. Square Feet (Luas bangunan)
2. Number of Bedrooms
3. Number of Bathrooms
4. Number of Floors
5. Year Built
6. Has Garden
7. Has Pool
8. Garage Size
9. Location Score
10. Distance to Center

## 🌐 Deploy to Streamlit Cloud

1. **Push ke GitHub**
```bash
git add .
git commit -m "Initial commit"
git push origin main
```

2. **Deploy**
   - Buka [share.streamlit.io](https://share.streamlit.io/)
   - Sign in dengan GitHub
   - Pilih repository dan branch
   - Klik "Deploy!"

3. **Share**
   - Dapatkan public URL
   - Share ke teman atau dosen! 🎉

## 🔄 Update Dashboard

Untuk update dashboard setelah deploy:

```bash
# Make changes to your code
git add .
git commit -m "Update dashboard features"
git push origin main

# Streamlit Cloud akan otomatis re-deploy!
```

## 🤝 Contributing

Contributions are welcome! Silakan:

1. Fork repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Nama Anda**
- GitHub: [@username](https://github.com/username)
- Email: your.email@example.com
- LinkedIn: [Your Name](https://linkedin.com/in/yourprofile)

## 🙏 Acknowledgments

- Dataset dari [sumber dataset]
- Dibuat untuk UAS Data Mining - Semester 7
- Dosen Pengampu: [Nama Dosen]
- Universitas: [Nama Universitas]

## 📚 References

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Linear Regression Theory](https://en.wikipedia.org/wiki/Linear_regression)

## 🐛 Known Issues

- [ ] None yet! Report issues di [GitHub Issues](https://github.com/username/uas-datamin/issues)

## 🔜 Future Improvements

- [ ] Tambah algoritma ML lain (Random Forest, XGBoost)
- [ ] Model comparison feature
- [ ] Export prediction results ke CSV/Excel
- [ ] Add data upload feature
- [ ] Implementasi feature engineering
- [ ] Add model interpretability (SHAP values)

---

⭐ **Jangan lupa kasih star jika project ini membantu!** ⭐

Made with ❤️ by kelompok 3