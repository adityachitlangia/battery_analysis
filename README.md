# 🔋 Battery Lifecycle Analysis Dashboard

A comprehensive Streamlit dashboard for analyzing battery performance data and predicting battery lifecycle using machine learning regression models.

## 📊 Features

### Data Exploration & Visualization
- **Interactive Data Overview**: Real-time visualization of voltage and current data across all batteries
- **Detailed Analysis**: Scatter plots, correlation matrices, and statistical summaries
- **Cycle Phase Analysis**: Automatic identification of charge, discharge, and rest phases
- **Battery Comparison**: Side-by-side comparison of battery performance metrics

### Machine Learning Predictions
- **Multiple Regression Models**: Linear Regression, Ridge, Random Forest, and Gradient Boosting
- **Model Performance Comparison**: R² score, MSE, and MAE metrics for model selection
- **Interactive Prediction**: Manual input of battery characteristics for lifecycle prediction
- **Visual Model Validation**: Actual vs Predicted scatter plots with perfect prediction line

### Performance Metrics
- **Battery Health Assessment**: Categorization into Poor, Good, and Excellent health levels
- **Voltage Degradation Tracking**: Monitoring voltage decline over cycles
- **Cycle Duration Analysis**: Statistical analysis of cycle performance
- **Comprehensive Statistics**: Mean, standard deviation, and range calculations

### Key Insights & Recommendations
- **Automated Insights**: Data-driven observations about battery performance
- **Health Distribution**: Visual representation of battery health across the dataset
- **Actionable Recommendations**: Suggestions for battery monitoring and maintenance

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Excel file: `Battery Data Final (3).xlsx` (should be in the same directory)

### Installation & Running

#### Option 1: Using the Runner Script (Recommended)
```bash
python run_dashboard.py
```

#### Option 2: Manual Installation
```bash
# Install requirements
pip install -r requirements.txt

# Run the dashboard
streamlit run battery_dashboard.py
```

The dashboard will be available at `http://localhost:8501`

## 📁 File Structure

```
Bhavya project/
├── Battery Data Final (3).xlsx          # Input data file
├── battery_dashboard.py                 # Main Streamlit dashboard
├── battery_analysis.py                  # Standalone analysis script
├── run_dashboard.py                     # Dashboard runner script
├── requirements.txt                     # Python dependencies
└── README.md                           # This file
```

## 🔧 Data Format

The dashboard expects an Excel file with the following structure:
- **Multiple sheets**: One sheet per battery (Battery 1, Battery 2, etc.)
- **Data columns**: Time (s), Voltage (V), Current (µA)
- **Header row**: Identified by 's' in the first column
- **Data types**: Numeric values for time, voltage, and current

## 📈 Dashboard Tabs

### 1. 📈 Data Overview
- Key performance metrics
- Voltage vs Time plots
- Current vs Time plots
- Battery selection controls

### 2. 🔍 Detailed Analysis
- Voltage vs Current scatter plots
- Cycle phase distribution
- Battery statistics table
- Correlation matrix heatmap

### 3. 🤖 ML Predictions
- Model training and comparison
- Performance metrics visualization
- Interactive prediction interface
- Actual vs Predicted validation plots

### 4. 📊 Performance Metrics
- Battery performance summary
- Voltage degradation over cycles
- Cycle duration analysis
- Statistical summaries

### 5. 💡 Insights
- Battery health distribution
- Key data insights
- Actionable recommendations
- Performance observations

## 🧠 Machine Learning Models

The dashboard includes four regression models for battery lifecycle prediction:

1. **Linear Regression**: Simple linear relationship modeling
2. **Ridge Regression**: L2 regularization to prevent overfitting
3. **Random Forest**: Ensemble method with multiple decision trees
4. **Gradient Boosting**: Sequential ensemble learning

### Features Used for Prediction
- Average voltage and current
- Voltage and current standard deviations
- Maximum and minimum voltages
- Voltage range
- Total operational time
- Charge and discharge ratios

## 📊 Key Metrics

### Battery Performance Metrics
- **Max/Min Voltage**: Peak and minimum voltage values per cycle
- **Average Voltage**: Mean voltage across all measurements
- **Voltage Range**: Difference between max and min voltage
- **Cycle Duration**: Time taken for each complete cycle
- **Current Range**: Difference between max and min current

### Model Performance Metrics
- **R² Score**: Coefficient of determination (0-1, higher is better)
- **MSE**: Mean Squared Error (lower is better)
- **MAE**: Mean Absolute Error (lower is better)

## 🎯 Use Cases

### Battery Manufacturers
- Quality control and performance validation
- Batch analysis and comparison
- Lifecycle prediction for warranty estimation

### Research & Development
- Battery chemistry optimization
- Performance degradation analysis
- Comparative studies across different battery types

### Energy Storage Systems
- Predictive maintenance scheduling
- Battery replacement planning
- System performance optimization

### Data Scientists & Engineers
- Exploratory data analysis
- Machine learning model development
- Statistical analysis and visualization

## 🔍 Data Insights

The dashboard automatically generates insights such as:
- Total dataset size and scope
- Voltage and current ranges
- Statistical distributions
- Battery health categorization
- Performance trends and patterns

## 🛠️ Customization

### Adding New Visualizations
1. Modify the `BatteryDashboard` class in `battery_dashboard.py`
2. Add new methods for data processing
3. Create new Plotly/Matplotlib visualizations
4. Add new tabs or sections in the main dashboard

### Modifying ML Models
1. Update the `train_models()` method
2. Add new scikit-learn models to the models dictionary
3. Adjust feature engineering in the `calculate_metrics()` method
4. Update the prediction interface accordingly

### Styling Changes
1. Modify the CSS in the `st.markdown()` sections
2. Update color schemes and layouts
3. Add custom HTML components
4. Adjust the page configuration

## 🐛 Troubleshooting

### Common Issues

1. **Excel file not found**
   - Ensure `Battery Data Final (3).xlsx` is in the same directory
   - Check file permissions

2. **Import errors**
   - Run `pip install -r requirements.txt`
   - Check Python version compatibility

3. **Data loading issues**
   - Verify Excel file format matches expected structure
   - Check for non-numeric values in data columns

4. **Dashboard not loading**
   - Check if port 8501 is available
   - Try running with `--server.port 8502`

### Performance Optimization

1. **Large datasets**: Consider data sampling for initial exploration
2. **Memory usage**: Monitor RAM usage with very large Excel files
3. **Model training**: Use smaller datasets for faster model training

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:
- New visualization types
- Additional ML models
- Performance improvements
- Bug fixes
- Documentation updates

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Review the code comments
3. Open an issue with detailed error information
4. Include sample data if possible

---

**Happy Analyzing! 🔋📊**
