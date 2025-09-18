#!/usr/bin/env python3
"""
Demo script showing key battery analysis features
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def load_battery_data():
    """Load and process battery data"""
    print("🔋 Loading Battery Data...")
    print("=" * 50)
    
    all_sheets = pd.read_excel('Battery Data Final (3).xlsx', sheet_name=None)
    battery_data = {}
    
    for sheet_name, df in all_sheets.items():
        if 'Battery' in sheet_name:
            # Find data start
            data_start = None
            for i, row in df.iterrows():
                if pd.notna(row.iloc[0]) and str(row.iloc[0]).strip() == 's':
                    data_start = i
                    break
            
            if data_start is not None:
                data_df = df.iloc[data_start:].copy()
                
                # Handle different column structures
                if data_df.shape[1] == 3:
                    data_df.columns = ['Time_s', 'Voltage_V', 'Current_uA']
                elif data_df.shape[1] > 3:
                    data_df = data_df.iloc[:, :3].copy()
                    data_df.columns = ['Time_s', 'Voltage_V', 'Current_uA']
                
                data_df = data_df.reset_index(drop=True)
                
                # Convert to numeric
                for col in data_df.columns:
                    data_df[col] = pd.to_numeric(data_df[col], errors='coerce')
                
                data_df = data_df.dropna()
                
                if len(data_df) > 0:
                    battery_id = sheet_name.replace('Battery ', '')
                    data_df['Battery_ID'] = battery_id
                    
                    # Identify cycles
                    data_df['Cycle_Phase'] = 'Unknown'
                    data_df['Cycle_Number'] = 0
                    
                    current_threshold = 10
                    data_df.loc[data_df['Current_uA'] > current_threshold, 'Cycle_Phase'] = 'Charge'
                    data_df.loc[data_df['Current_uA'] < -current_threshold, 'Cycle_Phase'] = 'Discharge'
                    data_df.loc[abs(data_df['Current_uA']) <= current_threshold, 'Cycle_Phase'] = 'Rest'
                    
                    phase_changes = data_df['Cycle_Phase'] != data_df['Cycle_Phase'].shift(1)
                    data_df['Cycle_Number'] = phase_changes.cumsum() // 3
                    
                    battery_data[sheet_name] = data_df
                    print(f"✅ {sheet_name}: {len(data_df):,} data points, {data_df['Cycle_Number'].max()} cycles")
    
    # Combine all data
    combined_data = pd.concat(battery_data.values(), ignore_index=True)
    print(f"\n📊 Total Dataset: {len(battery_data)} batteries, {len(combined_data):,} data points")
    
    return combined_data, battery_data

def analyze_battery_performance(data):
    """Analyze battery performance metrics"""
    print("\n📈 Battery Performance Analysis")
    print("=" * 50)
    
    # Calculate battery metrics
    battery_metrics = []
    for battery_id in data['Battery_ID'].unique():
        battery_df = data[data['Battery_ID'] == battery_id]
        
        metrics = {
            'Battery_ID': battery_id,
            'Total_Data_Points': len(battery_df),
            'Max_Voltage': battery_df['Voltage_V'].max(),
            'Min_Voltage': battery_df['Voltage_V'].min(),
            'Avg_Voltage': battery_df['Voltage_V'].mean(),
            'Voltage_Std': battery_df['Voltage_V'].std(),
            'Max_Current': battery_df['Current_uA'].max(),
            'Min_Current': battery_df['Current_uA'].min(),
            'Avg_Current': battery_df['Current_uA'].mean(),
            'Total_Time': battery_df['Time_s'].max() - battery_df['Time_s'].min(),
            'Max_Cycles': battery_df['Cycle_Number'].max(),
            'Voltage_Range': battery_df['Voltage_V'].max() - battery_df['Voltage_V'].min(),
        }
        battery_metrics.append(metrics)
    
    metrics_df = pd.DataFrame(battery_metrics)
    
    print("🔋 Battery Performance Summary:")
    print(metrics_df[['Battery_ID', 'Max_Cycles', 'Avg_Voltage', 'Voltage_Range', 'Total_Time']].round(3))
    
    # Key insights
    print(f"\n💡 Key Insights:")
    print(f"   • Best performing battery: Battery {metrics_df.loc[metrics_df['Max_Cycles'].idxmax(), 'Battery_ID']} ({metrics_df['Max_Cycles'].max()} cycles)")
    print(f"   • Highest voltage: {metrics_df['Max_Voltage'].max():.3f}V (Battery {metrics_df.loc[metrics_df['Max_Voltage'].idxmax(), 'Battery_ID']})")
    print(f"   • Most stable voltage: {metrics_df['Voltage_Std'].min():.3f}V std dev (Battery {metrics_df.loc[metrics_df['Voltage_Std'].idxmin(), 'Battery_ID']})")
    print(f"   • Longest test duration: {metrics_df['Total_Time'].max()/3600:.1f} hours (Battery {metrics_df.loc[metrics_df['Total_Time'].idxmax(), 'Battery_ID']})")
    
    return metrics_df

def train_lifecycle_models(data):
    """Train machine learning models for lifecycle prediction"""
    print("\n🤖 Machine Learning Model Training")
    print("=" * 50)
    
    # Calculate features
    battery_features = []
    for battery_id in data['Battery_ID'].unique():
        battery_df = data[data['Battery_ID'] == battery_id]
        
        features = {
            'Battery_ID': battery_id,
            'Total_Cycles': battery_df['Cycle_Number'].max(),
            'Avg_Voltage': battery_df['Voltage_V'].mean(),
            'Voltage_Std': battery_df['Voltage_V'].std(),
            'Avg_Current': battery_df['Current_uA'].mean(),
            'Current_Std': battery_df['Current_uA'].std(),
            'Max_Voltage': battery_df['Voltage_V'].max(),
            'Min_Voltage': battery_df['Voltage_V'].min(),
            'Voltage_Range': battery_df['Voltage_V'].max() - battery_df['Voltage_V'].min(),
            'Total_Time': battery_df['Time_s'].max(),
            'Data_Points': len(battery_df),
        }
        battery_features.append(features)
    
    features_df = pd.DataFrame(battery_features)
    feature_cols = [col for col in features_df.columns if col not in ['Battery_ID', 'Total_Cycles']]
    X = features_df[feature_cols].fillna(0)
    y = features_df['Total_Cycles']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    print("📊 Model Performance:")
    for name, model in models.items():
        if 'Random Forest' in name:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        else:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        
        print(f"   {name}:")
        print(f"     R² Score: {r2:.3f}")
        print(f"     MSE: {mse:.3f}")
    
    return models, scaler, feature_cols

def create_visualizations(data):
    """Create key visualizations"""
    print("\n📊 Creating Visualizations...")
    
    # Set up the plot
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Battery Analysis Dashboard - Key Visualizations', fontsize=16, fontweight='bold')
    
    # 1. Voltage vs Time for first 5 batteries
    ax1 = axes[0, 0]
    for battery_id in data['Battery_ID'].unique()[:5]:
        battery_df = data[data['Battery_ID'] == battery_id]
        ax1.plot(battery_df['Time_s'], battery_df['Voltage_V'], 
                label=f'Battery {battery_id}', alpha=0.7, linewidth=1)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Voltage (V)')
    ax1.set_title('Voltage vs Time (First 5 Batteries)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Current vs Time for first 5 batteries
    ax2 = axes[0, 1]
    for battery_id in data['Battery_ID'].unique()[:5]:
        battery_df = data[data['Battery_ID'] == battery_id]
        ax2.plot(battery_df['Time_s'], battery_df['Current_uA'], 
                label=f'Battery {battery_id}', alpha=0.7, linewidth=1)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Current (µA)')
    ax2.set_title('Current vs Time (First 5 Batteries)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Voltage vs Current scatter
    ax3 = axes[1, 0]
    scatter = ax3.scatter(data['Current_uA'], data['Voltage_V'], 
                         c=data['Battery_ID'].astype(int), alpha=0.6, cmap='tab10')
    ax3.set_xlabel('Current (µA)')
    ax3.set_ylabel('Voltage (V)')
    ax3.set_title('Voltage vs Current Scatter Plot')
    plt.colorbar(scatter, ax=ax3, label='Battery ID')
    ax3.grid(True, alpha=0.3)
    
    # 4. Cycle phase distribution
    ax4 = axes[1, 1]
    phase_counts = data['Cycle_Phase'].value_counts()
    ax4.pie(phase_counts.values, labels=phase_counts.index, autopct='%1.1f%%')
    ax4.set_title('Cycle Phase Distribution')
    
    plt.tight_layout()
    plt.savefig('battery_analysis_demo.png', dpi=300, bbox_inches='tight')
    print("✅ Visualizations saved as 'battery_analysis_demo.png'")
    
    return fig

def main():
    """Main demo function"""
    print("🔋 Battery Lifecycle Analysis Demo")
    print("=" * 60)
    
    # Load data
    data, battery_data = load_battery_data()
    
    # Analyze performance
    metrics_df = analyze_battery_performance(data)
    
    # Train models
    models, scaler, feature_cols = train_lifecycle_models(data)
    
    # Create visualizations
    fig = create_visualizations(data)
    
    print("\n🎉 Demo Complete!")
    print("=" * 60)
    print("Key Features Demonstrated:")
    print("✅ Data loading and preprocessing")
    print("✅ Battery performance analysis")
    print("✅ Machine learning model training")
    print("✅ Data visualization")
    print("\n🚀 Run 'python run_dashboard.py' for the full interactive dashboard!")

if __name__ == "__main__":
    main()
