# 🚀 Predictive Analytics for E-commerce Sales

A comprehensive machine learning project that predicts product sales using historical data and customer behavior patterns. Built with Python, Pandas, Scikit-learn, and XGBoost, featuring a beautiful web interface for real-time predictions.

![Project Banner](https://img.shields.io/badge/ML-Predictive%20Analytics-blue) ![Python](https://img.shields.io/badge/Python-3.8+-green) ![Accuracy](https://img.shields.io/badge/Accuracy-97.82%25-brightgreen) ![License](https://img.shields.io/badge/License-MIT-yellow)

## 🎯 Project Overview

This project implements advanced machine learning algorithms to predict e-commerce sales with **97.82% accuracy**. It processes a comprehensive dataset of 50,000+ records, performs sophisticated feature engineering, and provides real-time predictions through an intuitive web interface.

### Key Features
- **🤖 Dual Algorithm Approach**: Random Forest + XGBoost for maximum accuracy
- **📊 Advanced Analytics**: 52+ engineered features from customer behavior data
- **🌐 Web Interface**: Beautiful, responsive UI for real-time predictions
- **📈 High Performance**: 97.82% accuracy with XGBoost model
- **🔄 Comprehensive Pipeline**: End-to-end ML workflow from data generation to deployment

## 🏆 Model Performance

| Model | Accuracy | R² Score | RMSE | MAE |
|-------|----------|----------|------|-----|
| **XGBoost** | **97.82%** | **0.969** | 4.42 | 3.19 |
| Random Forest | 95.75% | 0.878 | 8.70 | 6.23 |

## 📁 Project Structure

```
ecommerce-prediction/
├── 📁 data/                    # Dataset storage
│   └── ecommerce_sales_data.csv
├── 📁 src/                     # Source code
│   ├── data_generator.py       # Synthetic data generation
│   ├── data_preprocessing.py   # Data cleaning & feature engineering
│   ├── models.py              # ML models implementation
│   └── generate_plots.py      # Visualization generation
├── 📁 models/                  # Trained models & plots
│   ├── random_forest_model.joblib
│   ├── xgboost_model.joblib
│   └── 📁 plots/              # Model visualizations
├── 📁 web/                     # Web application
│   ├── app.py                 # Flask backend
│   └── 📁 templates/
│       └── index.html         # Web interface
├── 📁 notebooks/              # Jupyter notebooks (for exploration)
├── 📁 tests/                  # Unit tests
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ecommerce-prediction.git
   cd ecommerce-prediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Generate dataset and train models**
   ```bash
   cd src
   python data_generator.py      # Generate synthetic dataset
   python generate_plots.py     # Train models and create visualizations
   ```

4. **Launch web application**
   ```bash
   cd ../web
   python app.py
   ```

5. **Open your browser and navigate to** `http://localhost:5000`

## 💡 Usage Examples

### Command Line Usage

```python
from src.models import SalesPredictionModels
from src.data_preprocessing import DataPreprocessor

# Load trained models
models = SalesPredictionModels()
models.load_models('models/')

# Make predictions
prediction = models.predict_sales('xgboost', input_features)
print(f"Predicted sales: {prediction[0]:.0f} units")
```

### Web Interface Usage

1. **Open the web application** at `http://localhost:5000`
2. **Fill in product details**:
   - Category (Electronics, Clothing, etc.)
   - Brand, Price, Discount
   - Ratings, Reviews, Promotion status
   - Stock level, Season
3. **Click "Predict Sales"** to get instant predictions from both models
4. **View results** with confidence metrics and model comparison

## 📊 Dataset Features

Our comprehensive dataset includes 23+ core features:

### Product Features
- Category, Brand, Price, Discount
- Average Rating, Number of Reviews
- Stock Level, Days Since Launch

### Customer Behavior
- Page Views, Time on Page, Bounce Rate
- Social Media Mentions
- Seasonal Patterns

### Market Intelligence
- Competitor Pricing
- Price Advantage Analysis
- Promotion Effectiveness

### Engineered Features (29 additional)
- Date-based features (seasonality, trends)
- Price-related ratios and interactions
- Customer engagement scores
- Inventory turnover metrics
- Competition analysis features

## 🧠 Machine Learning Pipeline

### 1. Data Generation
- **Synthetic Dataset Creation**: 50,000+ realistic e-commerce records
- **Multi-factor Sales Modeling**: Price, ratings, seasonality, promotions
- **Realistic Distributions**: Log-normal prices, Poisson reviews, Beta bounce rates

### 2. Data Preprocessing
- **Outlier Detection**: IQR-based outlier removal
- **Missing Value Handling**: Intelligent imputation strategies
- **Feature Scaling**: StandardScaler normalization
- **Categorical Encoding**: One-hot encoding for categories

### 3. Feature Engineering
- **Temporal Features**: Year, quarter, month, day patterns
- **Interaction Features**: Price × rating, promotion × weekend
- **Derived Metrics**: Engagement scores, turnover rates
- **Competitive Analysis**: Price advantage calculations

### 4. Model Training
- **Random Forest**: Ensemble of 100 decision trees
- **XGBoost**: Gradient boosting with optimized hyperparameters
- **Cross-validation**: Robust performance evaluation
- **Feature Importance**: Analysis of key predictive factors

### 5. Model Evaluation
- **Multiple Metrics**: Accuracy, R², RMSE, MAE
- **Visualization**: Actual vs predicted plots, residual analysis
- **Feature Importance**: Top predictive features identification

## 🛠️ Technical Implementation

### Technologies Used
- **Python 3.8+**: Core programming language
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning algorithms and preprocessing
- **XGBoost**: Gradient boosting framework
- **Flask**: Web framework for the prediction API
- **HTML/CSS/JavaScript**: Frontend web interface
- **Matplotlib/Seaborn**: Data visualization
- **Joblib**: Model serialization

### Key Algorithms
- **Random Forest Regressor**: Ensemble method with bagging
- **XGBoost Regressor**: Gradient boosting with regularization
- **StandardScaler**: Feature normalization
- **Train-test split**: 80/20 split with stratification

## 📈 Model Insights

### Top Predictive Features
1. **Page Views** - Strong indicator of customer interest
2. **Final Price** - Primary purchasing decision factor
3. **Average Rating** - Quality perception impact
4. **Stock Level** - Availability influence on sales
5. **Engagement Score** - Customer interaction depth

### Business Insights
- **Promotions increase sales by 15-20%** on average
- **Weekend sales patterns** differ by product category
- **Seasonal variations** are most pronounced in clothing and sports
- **Higher ratings** exponentially increase sales potential
- **Stock availability** is crucial for maintaining sales momentum

## 🌐 Web Application Features

### User Interface
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Real-time Predictions**: Instant results with dual model comparison
- **Interactive Forms**: Dynamic price calculation and validation
- **Visual Feedback**: Loading states, error handling, success animations

### API Endpoints
- `GET /`: Main prediction interface
- `POST /predict`: Sales prediction API
- `GET /model-info`: Model metrics and information

## 🔮 Future Enhancements

### Planned Features
- **📊 Advanced Analytics Dashboard**: Comprehensive business intelligence
- **🔄 Model Retraining Pipeline**: Automated model updates with new data
- **📱 Mobile App**: Native mobile application for on-the-go predictions
- **🔗 API Integration**: REST API for third-party integrations
- **📈 A/B Testing Framework**: Compare different model versions
- **🛒 Real-time Data Integration**: Connect with actual e-commerce platforms

### Technical Improvements
- **Docker Containerization**: Easy deployment and scaling
- **Cloud Deployment**: AWS/GCP deployment with auto-scaling
- **Model Monitoring**: Performance tracking and drift detection
- **Advanced Models**: Deep learning approaches (LSTM, Transformers)

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit your changes**: `git commit -m 'Add amazing feature'`
4. **Push to the branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/

# Run linting
flake8 src/

# Run type checking
mypy src/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)
- Email: your.email@example.com

## 🙏 Acknowledgments

- **Scikit-learn Community** for excellent ML tools
- **XGBoost Team** for the powerful gradient boosting library
- **Flask Community** for the lightweight web framework
- **Open Source Contributors** who made this project possible

## 📚 References

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Pandas Documentation](https://pandas.pydata.org/)

---

⭐ **Star this repository if you found it helpful!**

📊 **Check out our live demo**: [Demo Link](https://your-demo-link.com)

🐛 **Report bugs**: [Issues](https://github.com/yourusername/ecommerce-prediction/issues)

💬 **Join our community**: [Discussions](https://github.com/yourusername/ecommerce-prediction/discussions)
