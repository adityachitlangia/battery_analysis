import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class BatteryAnalyzer:
    def __init__(self, excel_file):
        self.excel_file = excel_file
        self.battery_data = {}
        self.processed_data = None
        self.models = {}
        self.scaler = StandardScaler()
        
    def load_and_clean_data(self):
        """Load and clean battery data from Excel file"""
        print("Loading battery data...")
        
        # Load all sheets
        all_sheets = pd.read_excel(self.excel_file, sheet_name=None)
        
        for sheet_name, df in all_sheets.items():
            if 'Battery' in sheet_name:
                # Find the actual data rows (skip header rows)
                data_start = None
                for i, row in df.iterrows():
                    if pd.notna(row.iloc[0]) and str(row.iloc[0]).strip() == 's':
                        data_start = i
                        break
                
                if data_start is not None:
                    # Extract data starting from the header row
                    data_df = df.iloc[data_start:].copy()
                    data_df.columns = ['Time_s', 'Voltage_V', 'Current_uA']
                    data_df = data_df.reset_index(drop=True)
                    
                    # Convert to numeric, handling any non-numeric values
                    for col in data_df.columns:
                        data_df[col] = pd.to_numeric(data_df[col], errors='coerce')
                    
                    # Remove rows with NaN values
                    data_df = data_df.dropna()
                    
                    # Add battery ID
                    battery_id = sheet_name.replace('Battery ', '')
                    data_df['Battery_ID'] = battery_id
                    
                    # Identify cycles based on current changes
                    data_df = self._identify_cycles(data_df)
                    
                    self.battery_data[sheet_name] = data_df
                    print(f"Loaded {sheet_name}: {len(data_df)} data points")
        
        # Combine all battery data
        if self.battery_data:
            self.processed_data = pd.concat(self.battery_data.values(), ignore_index=True)
            print(f"Total data points: {len(self.processed_data)}")
            return True
        return False
    
    def _identify_cycles(self, df):
        """Identify charge/discharge/rest cycles based on current patterns"""
        df = df.copy()
        df['Cycle_Phase'] = 'Unknown'
        df['Cycle_Number'] = 0
        
        # Simple heuristic: positive current = charge, negative = discharge, near zero = rest
        current_threshold = 10  # µA threshold for rest
        
        df.loc[df['Current_uA'] > current_threshold, 'Cycle_Phase'] = 'Charge'
        df.loc[df['Current_uA'] < -current_threshold, 'Cycle_Phase'] = 'Discharge'
        df.loc[abs(df['Current_uA']) <= current_threshold, 'Cycle_Phase'] = 'Rest'
        
        # Identify cycle numbers based on phase changes
        phase_changes = df['Cycle_Phase'] != df['Cycle_Phase'].shift(1)
        df['Cycle_Number'] = phase_changes.cumsum() // 3  # Assuming 3 phases per cycle
        
        return df
    
    def calculate_battery_metrics(self):
        """Calculate key battery performance metrics"""
        if self.processed_data is None:
            return None
        
        metrics = []
        
        for battery_id in self.processed_data['Battery_ID'].unique():
            battery_df = self.processed_data[self.processed_data['Battery_ID'] == battery_id]
            
            for cycle in battery_df['Cycle_Number'].unique():
                cycle_df = battery_df[battery_df['Cycle_Number'] == cycle]
                
                if len(cycle_df) == 0:
                    continue
                
                # Calculate metrics for this cycle
                cycle_metrics = {
                    'Battery_ID': battery_id,
                    'Cycle_Number': cycle,
                    'Max_Voltage': cycle_df['Voltage_V'].max(),
                    'Min_Voltage': cycle_df['Voltage_V'].min(),
                    'Avg_Voltage': cycle_df['Voltage_V'].mean(),
                    'Max_Current': cycle_df['Current_uA'].max(),
                    'Min_Current': cycle_df['Current_uA'].min(),
                    'Avg_Current': cycle_df['Current_uA'].mean(),
                    'Cycle_Duration': cycle_df['Time_s'].max() - cycle_df['Time_s'].min(),
                    'Energy_Input': self._calculate_energy(cycle_df, 'Charge'),
                    'Energy_Output': self._calculate_energy(cycle_df, 'Discharge'),
                    'Efficiency': 0
                }
                
                # Calculate efficiency
                if cycle_metrics['Energy_Input'] > 0:
                    cycle_metrics['Efficiency'] = (cycle_metrics['Energy_Output'] / 
                                                 cycle_metrics['Energy_Input']) * 100
                
                metrics.append(cycle_metrics)
        
        return pd.DataFrame(metrics)
    
    def _calculate_energy(self, df, phase):
        """Calculate energy for a specific phase"""
        phase_df = df[df['Cycle_Phase'] == phase]
        if len(phase_df) == 0:
            return 0
        
        # Energy = Power * Time = Voltage * Current * Time
        # Convert current from µA to A
        power = phase_df['Voltage_V'] * (phase_df['Current_uA'] / 1e6)
        time_diff = phase_df['Time_s'].diff().fillna(0)
        energy = (power * time_diff).sum()
        
        return energy
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        if self.processed_data is None:
            return None
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Voltage vs Time for all batteries
        plt.subplot(3, 3, 1)
        for battery_id in self.processed_data['Battery_ID'].unique()[:5]:  # Show first 5 batteries
            battery_df = self.processed_data[self.processed_data['Battery_ID'] == battery_id]
            plt.plot(battery_df['Time_s'], battery_df['Voltage_V'], 
                    label=f'Battery {battery_id}', alpha=0.7)
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (V)')
        plt.title('Voltage vs Time (First 5 Batteries)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Current vs Time for all batteries
        plt.subplot(3, 3, 2)
        for battery_id in self.processed_data['Battery_ID'].unique()[:5]:
            battery_df = self.processed_data[self.processed_data['Battery_ID'] == battery_id]
            plt.plot(battery_df['Time_s'], battery_df['Current_uA'], 
                    label=f'Battery {battery_id}', alpha=0.7)
        plt.xlabel('Time (s)')
        plt.ylabel('Current (µA)')
        plt.title('Current vs Time (First 5 Batteries)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. Voltage vs Current scatter
        plt.subplot(3, 3, 3)
        scatter = plt.scatter(self.processed_data['Current_uA'], 
                            self.processed_data['Voltage_V'], 
                            c=self.processed_data['Battery_ID'].astype(int), 
                            alpha=0.6, cmap='tab10')
        plt.xlabel('Current (µA)')
        plt.ylabel('Voltage (V)')
        plt.title('Voltage vs Current Scatter')
        plt.colorbar(scatter, label='Battery ID')
        plt.grid(True, alpha=0.3)
        
        # 4. Cycle phase distribution
        plt.subplot(3, 3, 4)
        phase_counts = self.processed_data['Cycle_Phase'].value_counts()
        plt.pie(phase_counts.values, labels=phase_counts.index, autopct='%1.1f%%')
        plt.title('Cycle Phase Distribution')
        
        # 5. Battery capacity over cycles (if we have cycle data)
        plt.subplot(3, 3, 5)
        metrics_df = self.calculate_battery_metrics()
        if metrics_df is not None and len(metrics_df) > 0:
            for battery_id in metrics_df['Battery_ID'].unique()[:5]:
                battery_metrics = metrics_df[metrics_df['Battery_ID'] == battery_id]
                plt.plot(battery_metrics['Cycle_Number'], 
                        battery_metrics['Max_Voltage'], 
                        'o-', label=f'Battery {battery_id}', alpha=0.7)
        plt.xlabel('Cycle Number')
        plt.ylabel('Max Voltage (V)')
        plt.title('Max Voltage vs Cycle Number')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 6. Current distribution
        plt.subplot(3, 3, 6)
        plt.hist(self.processed_data['Current_uA'], bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Current (µA)')
        plt.ylabel('Frequency')
        plt.title('Current Distribution')
        plt.grid(True, alpha=0.3)
        
        # 7. Voltage distribution
        plt.subplot(3, 3, 7)
        plt.hist(self.processed_data['Voltage_V'], bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Voltage (V)')
        plt.ylabel('Frequency')
        plt.title('Voltage Distribution')
        plt.grid(True, alpha=0.3)
        
        # 8. Battery comparison - average voltage
        plt.subplot(3, 3, 8)
        battery_avg_voltage = self.processed_data.groupby('Battery_ID')['Voltage_V'].mean()
        battery_avg_voltage.plot(kind='bar', alpha=0.7)
        plt.xlabel('Battery ID')
        plt.ylabel('Average Voltage (V)')
        plt.title('Average Voltage by Battery')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 9. Correlation heatmap
        plt.subplot(3, 3, 9)
        numeric_cols = ['Time_s', 'Voltage_V', 'Current_uA']
        corr_matrix = self.processed_data[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix')
        
        plt.tight_layout()
        return fig
    
    def train_lifecycle_models(self):
        """Train regression models for battery lifecycle prediction"""
        if self.processed_data is None:
            return None
        
        # Calculate features for each battery
        battery_features = []
        
        for battery_id in self.processed_data['Battery_ID'].unique():
            battery_df = self.processed_data[self.processed_data['Battery_ID'] == battery_id]
            
            # Calculate features
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
                'Charge_Time': len(battery_df[battery_df['Cycle_Phase'] == 'Charge']),
                'Discharge_Time': len(battery_df[battery_df['Cycle_Phase'] == 'Discharge']),
                'Rest_Time': len(battery_df[battery_df['Cycle_Phase'] == 'Rest']),
            }
            
            battery_features.append(features)
        
        features_df = pd.DataFrame(battery_features)
        
        # Prepare features and target
        feature_cols = [col for col in features_df.columns if col != 'Battery_ID']
        X = features_df[feature_cols].fillna(0)
        y = features_df['Total_Cycles']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train multiple models
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        model_results = {}
        
        for name, model in models.items():
            # Train model
            if 'Random Forest' in name or 'Gradient Boosting' in name:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            else:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            model_results[name] = {
                'model': model,
                'mse': mse,
                'r2': r2,
                'mae': mae,
                'predictions': y_pred,
                'actual': y_test
            }
            
            print(f"{name} - R²: {r2:.3f}, MSE: {mse:.3f}, MAE: {mae:.3f}")
        
        self.models = model_results
        return model_results
    
    def predict_battery_lifecycle(self, battery_features):
        """Predict battery lifecycle for given features"""
        if not self.models:
            return None
        
        # Use the best model (highest R²)
        best_model_name = max(self.models.keys(), key=lambda x: self.models[x]['r2'])
        best_model = self.models[best_model_name]['model']
        
        # Scale features if needed
        if 'Random Forest' not in best_model_name and 'Gradient Boosting' not in best_model_name:
            battery_features_scaled = self.scaler.transform([battery_features])
            prediction = best_model.predict(battery_features_scaled)[0]
        else:
            prediction = best_model.predict([battery_features])[0]
        
        return prediction, best_model_name

def main():
    # Initialize analyzer
    analyzer = BatteryAnalyzer('Battery Data Final (3).xlsx')
    
    # Load and clean data
    if analyzer.load_and_clean_data():
        print("Data loaded successfully!")
        
        # Create visualizations
        print("Creating visualizations...")
        fig = analyzer.create_visualizations()
        if fig:
            plt.savefig('battery_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # Train models
        print("Training lifecycle prediction models...")
        model_results = analyzer.train_lifecycle_models()
        
        # Calculate battery metrics
        print("Calculating battery metrics...")
        metrics_df = analyzer.calculate_battery_metrics()
        if metrics_df is not None:
            print(f"Calculated metrics for {len(metrics_df)} cycles")
            print(metrics_df.head())
        
    else:
        print("Failed to load data!")

if __name__ == "__main__":
    main()
