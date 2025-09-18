import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Battery Lifecycle Analysis Dashboard",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .insight-box {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .prediction-box {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class BatteryDashboard:
    def __init__(self):
        self.battery_data = {}
        self.processed_data = None
        self.metrics_df = None
        self.models = {}
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load and process battery data"""
        try:
            # Load all sheets
            all_sheets = pd.read_excel('Battery Data Final (3).xlsx', sheet_name=None)
            
            for sheet_name, df in all_sheets.items():
                if 'Battery' in sheet_name:
                    # Find the actual data rows (look for 's' in first column)
                    data_start = None
                    for i, row in df.iterrows():
                        if pd.notna(row.iloc[0]) and str(row.iloc[0]).strip() == 's':
                            data_start = i
                            break
                    
                    if data_start is not None:
                        # Extract data starting from the header row
                        data_df = df.iloc[data_start:].copy()
                        
                        # Handle different column structures
                        if data_df.shape[1] == 3:
                            # Standard 3-column format
                            data_df.columns = ['Time_s', 'Voltage_V', 'Current_uA']
                        elif data_df.shape[1] > 3:
                            # Multi-column format - use first 3 columns
                            data_df = data_df.iloc[:, :3].copy()
                            data_df.columns = ['Time_s', 'Voltage_V', 'Current_uA']
                        else:
                            st.warning(f"Unexpected column structure for {sheet_name}: {data_df.shape[1]} columns")
                            continue
                        
                        data_df = data_df.reset_index(drop=True)
                        
                        # Convert to numeric, handling any non-numeric values
                        for col in data_df.columns:
                            data_df[col] = pd.to_numeric(data_df[col], errors='coerce')
                        
                        # Remove rows with NaN values
                        data_df = data_df.dropna()
                        
                        # Only proceed if we have data
                        if len(data_df) > 0:
                            battery_id = sheet_name.replace('Battery ', '')
                            data_df['Battery_ID'] = battery_id
                            
                            # Identify cycles
                            data_df = self._identify_cycles(data_df)
                            self.battery_data[sheet_name] = data_df
                            
                            st.write(f"✅ Loaded {sheet_name}: {len(data_df)} data points")
            
            # Combine all data
            if self.battery_data:
                self.processed_data = pd.concat(self.battery_data.values(), ignore_index=True)
                self.metrics_df = self.calculate_metrics()
                st.success(f"Successfully loaded {len(self.battery_data)} batteries with {len(self.processed_data)} total data points")
                return True
            else:
                st.error("No valid battery data found in the Excel file")
                return False
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            st.write("Please check that the Excel file format matches the expected structure")
            return False
    
    def _identify_cycles(self, df):
        """Identify charge/discharge/rest cycles"""
        df = df.copy()
        df['Cycle_Phase'] = 'Unknown'
        df['Cycle_Number'] = 0
        
        current_threshold = 10
        df.loc[df['Current_uA'] > current_threshold, 'Cycle_Phase'] = 'Charge'
        df.loc[df['Current_uA'] < -current_threshold, 'Cycle_Phase'] = 'Discharge'
        df.loc[abs(df['Current_uA']) <= current_threshold, 'Cycle_Phase'] = 'Rest'
        
        phase_changes = df['Cycle_Phase'] != df['Cycle_Phase'].shift(1)
        df['Cycle_Number'] = phase_changes.cumsum() // 3
        
        return df
    
    def calculate_metrics(self):
        """Calculate battery performance metrics"""
        if self.processed_data is None or len(self.processed_data) == 0:
            return None
        
        metrics = []
        for battery_id in self.processed_data['Battery_ID'].unique():
            battery_df = self.processed_data[self.processed_data['Battery_ID'] == battery_id]
            
            # Calculate overall battery metrics
            battery_metrics = {
                'Battery_ID': battery_id,
                'Total_Data_Points': len(battery_df),
                'Max_Voltage': battery_df['Voltage_V'].max(),
                'Min_Voltage': battery_df['Voltage_V'].min(),
                'Avg_Voltage': battery_df['Voltage_V'].mean(),
                'Voltage_Std': battery_df['Voltage_V'].std(),
                'Max_Current': battery_df['Current_uA'].max(),
                'Min_Current': battery_df['Current_uA'].min(),
                'Avg_Current': battery_df['Current_uA'].mean(),
                'Current_Std': battery_df['Current_uA'].std(),
                'Total_Time': battery_df['Time_s'].max() - battery_df['Time_s'].min(),
                'Voltage_Range': battery_df['Voltage_V'].max() - battery_df['Voltage_V'].min(),
                'Current_Range': battery_df['Current_uA'].max() - battery_df['Current_uA'].min(),
                'Max_Cycle': battery_df['Cycle_Number'].max() if 'Cycle_Number' in battery_df.columns else 0,
            }
            metrics.append(battery_metrics)
        
        return pd.DataFrame(metrics)
    
    def train_models(self):
        """Train regression models for lifecycle prediction"""
        if self.processed_data is None or len(self.processed_data) == 0:
            return None
        
        # Calculate features for each battery
        battery_features = []
        for battery_id in self.processed_data['Battery_ID'].unique():
            battery_df = self.processed_data[self.processed_data['Battery_ID'] == battery_id]
            
            # Calculate features
            features = {
                'Battery_ID': battery_id,
                'Total_Cycles': battery_df['Cycle_Number'].max() if 'Cycle_Number' in battery_df.columns else 0,
                'Avg_Voltage': battery_df['Voltage_V'].mean(),
                'Voltage_Std': battery_df['Voltage_V'].std(),
                'Avg_Current': battery_df['Current_uA'].mean(),
                'Current_Std': battery_df['Current_uA'].std(),
                'Max_Voltage': battery_df['Voltage_V'].max(),
                'Min_Voltage': battery_df['Voltage_V'].min(),
                'Voltage_Range': battery_df['Voltage_V'].max() - battery_df['Voltage_V'].min(),
                'Total_Time': battery_df['Time_s'].max(),
                'Data_Points': len(battery_df),
                'Charge_Ratio': len(battery_df[battery_df['Cycle_Phase'] == 'Charge']) / len(battery_df) if 'Cycle_Phase' in battery_df.columns else 0,
                'Discharge_Ratio': len(battery_df[battery_df['Cycle_Phase'] == 'Discharge']) / len(battery_df) if 'Cycle_Phase' in battery_df.columns else 0,
            }
            battery_features.append(features)
        
        features_df = pd.DataFrame(battery_features)
        
        # Use data points as a proxy for cycles if cycle data is not available
        if features_df['Total_Cycles'].sum() == 0:
            features_df['Total_Cycles'] = features_df['Data_Points'] // 100  # Rough estimate
        
        feature_cols = [col for col in features_df.columns if col not in ['Battery_ID', 'Total_Cycles']]
        X = features_df[feature_cols].fillna(0)
        y = features_df['Total_Cycles']
        
        # Check if we have enough data for training
        if len(features_df) < 2:
            st.warning("Not enough battery data for model training. Need at least 2 batteries.")
            return None
        
        # Split and scale
        if len(features_df) >= 4:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        else:
            # Use all data for training if we have very few batteries
            X_train, X_test, y_train, y_test = X, X, y, y
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train models
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        model_results = {}
        for name, model in models.items():
            try:
                if 'Random Forest' in name or 'Gradient Boosting' in name:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                else:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                
                r2 = r2_score(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                
                model_results[name] = {
                    'model': model,
                    'r2': r2,
                    'mse': mse,
                    'mae': mae,
                    'predictions': y_pred,
                    'actual': y_test
                }
            except Exception as e:
                st.warning(f"Error training {name}: {str(e)}")
                continue
        
        self.models = model_results
        return model_results

def main():
    # Initialize dashboard
    dashboard = BatteryDashboard()
    
    # Header
    st.markdown('<h1 class="main-header">🔋 Battery Lifecycle Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    if not dashboard.load_data():
        st.error("Failed to load battery data. Please check the Excel file.")
        return
    
    # Sidebar
    st.sidebar.title("📊 Dashboard Controls")
    
    # Data overview
    st.sidebar.markdown("### Data Overview")
    st.sidebar.metric("Total Batteries", len(dashboard.battery_data))
    st.sidebar.metric("Total Data Points", len(dashboard.processed_data))
    st.sidebar.metric("Total Cycles", dashboard.processed_data['Cycle_Number'].max())
    
    # Battery selection
    selected_batteries = st.sidebar.multiselect(
        "Select Batteries to Analyze",
        options=sorted(dashboard.processed_data['Battery_ID'].unique()),
        default=sorted(dashboard.processed_data['Battery_ID'].unique())[:5]
    )
    
    # Filter data
    filtered_data = dashboard.processed_data[
        dashboard.processed_data['Battery_ID'].isin(selected_batteries)
    ]
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Data Overview", 
        "🔍 Detailed Analysis", 
        "🤖 ML Predictions", 
        "📊 Performance Metrics", 
        "💡 Insights"
    ])
    
    with tab1:
        st.header("📈 Data Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Avg Voltage", f"{filtered_data['Voltage_V'].mean():.3f} V")
        with col2:
            st.metric("Avg Current", f"{filtered_data['Current_uA'].mean():.1f} µA")
        with col3:
            st.metric("Max Cycles", filtered_data['Cycle_Number'].max())
        with col4:
            st.metric("Data Points", len(filtered_data))
        
        # Voltage vs Time
        fig1 = px.line(
            filtered_data, 
            x='Time_s', 
            y='Voltage_V', 
            color='Battery_ID',
            title='Voltage vs Time',
            labels={'Time_s': 'Time (seconds)', 'Voltage_V': 'Voltage (V)'}
        )
        fig1.update_layout(height=500)
        st.plotly_chart(fig1, use_container_width=True)
        
        # Current vs Time
        fig2 = px.line(
            filtered_data, 
            x='Time_s', 
            y='Current_uA', 
            color='Battery_ID',
            title='Current vs Time',
            labels={'Time_s': 'Time (seconds)', 'Current_uA': 'Current (µA)'}
        )
        fig2.update_layout(height=500)
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab2:
        st.header("🔍 Detailed Analysis")
        
        # Voltage vs Current scatter
        fig3 = px.scatter(
            filtered_data,
            x='Current_uA',
            y='Voltage_V',
            color='Battery_ID',
            size='Time_s',
            title='Voltage vs Current Scatter Plot',
            labels={'Current_uA': 'Current (µA)', 'Voltage_V': 'Voltage (V)'}
        )
        fig3.update_layout(height=500)
        st.plotly_chart(fig3, use_container_width=True)
        
        # Cycle phase distribution
        col1, col2 = st.columns(2)
        
        with col1:
            phase_counts = filtered_data['Cycle_Phase'].value_counts()
            fig4 = px.pie(
                values=phase_counts.values,
                names=phase_counts.index,
                title='Cycle Phase Distribution'
            )
            st.plotly_chart(fig4, use_container_width=True)
        
        with col2:
            # Battery comparison
            battery_stats = filtered_data.groupby('Battery_ID').agg({
                'Voltage_V': ['mean', 'std'],
                'Current_uA': ['mean', 'std'],
                'Cycle_Number': 'max'
            }).round(3)
            
            st.subheader("Battery Statistics")
            st.dataframe(battery_stats)
        
        # Correlation heatmap
        numeric_cols = ['Time_s', 'Voltage_V', 'Current_uA']
        corr_matrix = filtered_data[numeric_cols].corr()
        
        fig5 = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="Correlation Matrix"
        )
        st.plotly_chart(fig5, use_container_width=True)
    
    with tab3:
        st.header("🤖 Machine Learning Predictions")
        
        # Train models
        if st.button("Train Models", type="primary"):
            with st.spinner("Training models..."):
                model_results = dashboard.train_models()
                
                if model_results:
                    st.success("Models trained successfully!")
                    
                    # Model comparison
                    model_comparison = []
                    for name, results in model_results.items():
                        model_comparison.append({
                            'Model': name,
                            'R² Score': results['r2'],
                            'MSE': results['mse'],
                            'MAE': results['mae']
                        })
                    
                    comparison_df = pd.DataFrame(model_comparison)
                    st.subheader("Model Performance Comparison")
                    st.dataframe(comparison_df, use_container_width=True)
                    
                    # Best model
                    best_model = max(model_comparison, key=lambda x: x['R² Score'])
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h3>🏆 Best Model: {best_model['Model']}</h3>
                        <p>R² Score: {best_model['R² Score']:.3f}</p>
                        <p>Mean Absolute Error: {best_model['MAE']:.3f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Prediction visualization
                    best_model_name = best_model['Model']
                    best_results = model_results[best_model_name]
                    
                    fig6 = px.scatter(
                        x=best_results['actual'],
                        y=best_results['predictions'],
                        title=f'Actual vs Predicted Cycles - {best_model_name}',
                        labels={'x': 'Actual Cycles', 'y': 'Predicted Cycles'}
                    )
                    fig6.add_shape(
                        type="line",
                        x0=best_results['actual'].min(),
                        y0=best_results['actual'].min(),
                        x1=best_results['actual'].max(),
                        y1=best_results['actual'].max(),
                        line=dict(dash="dash", color="red")
                    )
                    st.plotly_chart(fig6, use_container_width=True)
        
        # Manual prediction
        st.subheader("🔮 Manual Prediction")
        st.write("Enter battery characteristics to predict lifecycle:")
        
        col1, col2 = st.columns(2)
        with col1:
            avg_voltage = st.number_input("Average Voltage (V)", value=1.2, min_value=0.0, max_value=5.0)
            voltage_std = st.number_input("Voltage Std Dev", value=0.1, min_value=0.0, max_value=1.0)
            max_voltage = st.number_input("Max Voltage (V)", value=1.5, min_value=0.0, max_value=5.0)
        
        with col2:
            avg_current = st.number_input("Average Current (µA)", value=100.0, min_value=-1000.0, max_value=1000.0)
            current_std = st.number_input("Current Std Dev", value=50.0, min_value=0.0, max_value=500.0)
            total_time = st.number_input("Total Time (s)", value=10000, min_value=0, max_value=1000000)
        
        if st.button("Predict Lifecycle"):
            if dashboard.models:
                # Prepare features (matching the training feature order)
                features = [
                    avg_voltage, voltage_std, avg_current, current_std,
                    max_voltage, 0, total_time, 1000, 0.3, 0.3  # Added data_points estimate
                ]
                
                # Get best model
                best_model_name = max(dashboard.models.keys(), key=lambda x: dashboard.models[x]['r2'])
                best_model = dashboard.models[best_model_name]['model']
                
                # Make prediction
                if 'Random Forest' in best_model_name or 'Gradient Boosting' in best_model_name:
                    prediction = best_model.predict([features])[0]
                else:
                    features_scaled = dashboard.scaler.transform([features])
                    prediction = best_model.predict(features_scaled)[0]
                
                st.markdown(f"""
                <div class="prediction-box">
                    <h3>🔮 Predicted Lifecycle</h3>
                    <h2>{prediction:.1f} cycles</h2>
                    <p>Using {best_model_name}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("Please train models first!")
    
    with tab4:
        st.header("📊 Performance Metrics")
        
        if dashboard.metrics_df is not None:
            # Cycle analysis
            available_cols = dashboard.metrics_df.columns.tolist()
            agg_dict = {
                'Max_Voltage': 'mean',
                'Min_Voltage': 'mean',
                'Voltage_Range': 'mean'
            }
            
            # Add cycle-related columns if they exist
            if 'Max_Cycle' in available_cols:
                agg_dict['Max_Cycle'] = 'max'
            if 'Total_Time' in available_cols:
                agg_dict['Total_Time'] = 'mean'
            
            cycle_analysis = dashboard.metrics_df.groupby('Battery_ID').agg(agg_dict).round(3)
            
            st.subheader("Battery Performance Summary")
            st.dataframe(cycle_analysis, use_container_width=True)
            
            # Voltage analysis
            if 'Max_Cycle' in available_cols:
                fig7 = px.scatter(
                    dashboard.metrics_df,
                    x='Max_Cycle',
                    y='Max_Voltage',
                    color='Battery_ID',
                    title='Max Voltage vs Max Cycles',
                    labels={'Max_Cycle': 'Max Cycles', 'Max_Voltage': 'Max Voltage (V)'}
                )
                st.plotly_chart(fig7, use_container_width=True)
            
            # Voltage range analysis
            fig8 = px.box(
                dashboard.metrics_df,
                x='Battery_ID',
                y='Voltage_Range',
                title='Voltage Range Distribution by Battery'
            )
            st.plotly_chart(fig8, use_container_width=True)
    
    with tab5:
        st.header("💡 Key Insights")
        
        # Calculate insights
        total_batteries = len(dashboard.battery_data)
        total_cycles = dashboard.processed_data['Cycle_Number'].max() if 'Cycle_Number' in dashboard.processed_data.columns else 0
        avg_voltage = dashboard.processed_data['Voltage_V'].mean()
        voltage_std = dashboard.processed_data['Voltage_V'].std()
        
        # Battery health distribution
        if dashboard.metrics_df is not None and 'Max_Cycle' in dashboard.metrics_df.columns:
            max_cycles = dashboard.metrics_df.groupby('Battery_ID')['Max_Cycle'].max()
            health_categories = pd.cut(max_cycles, bins=3, labels=['Poor', 'Good', 'Excellent'])
            health_dist = health_categories.value_counts()
            
            st.markdown("### 🔋 Battery Health Distribution")
            fig9 = px.pie(
                values=health_dist.values,
                names=health_dist.index,
                title='Battery Health Categories'
            )
            st.plotly_chart(fig9, use_container_width=True)
        
        # Key insights
        st.markdown("### 📊 Key Insights")
        
        insights = [
            f"**Total Dataset**: {total_batteries} batteries with {len(dashboard.processed_data):,} data points",
            f"**Voltage Range**: {dashboard.processed_data['Voltage_V'].min():.3f}V - {dashboard.processed_data['Voltage_V'].max():.3f}V",
            f"**Current Range**: {dashboard.processed_data['Current_uA'].min():.1f}µA - {dashboard.processed_data['Current_uA'].max():.1f}µA",
            f"**Average Voltage**: {avg_voltage:.3f}V ± {voltage_std:.3f}V",
        ]
        
        if total_cycles > 0:
            insights.append(f"**Maximum Cycles Observed**: {total_cycles}")
        else:
            insights.append("**Cycle Analysis**: Cycle detection in progress")
        
        for insight in insights:
            st.markdown(f"<div class='insight-box'>{insight}</div>", unsafe_allow_html=True)
        
        # Recommendations
        st.markdown("### 🎯 Recommendations")
        recommendations = [
            "Monitor voltage degradation patterns to predict battery failure",
            "Implement early warning systems based on cycle performance metrics",
            "Use machine learning models for proactive battery replacement",
            "Focus on batteries with high voltage variance for closer monitoring",
            "Consider temperature and usage pattern data for more accurate predictions"
        ]
        
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"**{i}.** {rec}")

if __name__ == "__main__":
    main()
