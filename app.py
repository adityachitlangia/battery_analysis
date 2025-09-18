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
    page_title="Battery Lifecycle Analysis",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
            # For Streamlit Cloud, we'll use a sample data URL or create sample data
            # You can replace this with your actual data loading logic
            
            # Create sample data for demonstration
            st.info("📊 Using sample battery data for demonstration. Upload your Excel file to use real data.")
            
            # Generate sample data
            np.random.seed(42)
            sample_data = []
            
            for battery_id in range(1, 15):
                # Generate time series data
                time_points = np.linspace(0, 1000, 100)
                voltage = 12.0 + 0.5 * np.sin(time_points/100) + np.random.normal(0, 0.1, 100)
                current = 2.0 + 1.0 * np.sin(time_points/50) + np.random.normal(0, 0.2, 100)
                
                battery_df = pd.DataFrame({
                    'Time_s': time_points,
                    'Voltage_V': voltage,
                    'Current_uA': current,
                    'Battery_ID': f'Battery {battery_id}'
                })
                
                # Add cycle information
                battery_df['Cycle_Number'] = (time_points // 100).astype(int)
                battery_df['Cycle_Phase'] = 'Charge'
                battery_df.loc[battery_df['Current_uA'] < 0, 'Cycle_Phase'] = 'Discharge'
                battery_df.loc[abs(battery_df['Current_uA']) < 0.5, 'Cycle_Phase'] = 'Rest'
                
                self.battery_data[f'Battery {battery_id}'] = battery_df
                sample_data.append(battery_df)
            
            self.processed_data = pd.concat(sample_data, ignore_index=True)
            self.metrics_df = self.calculate_metrics()
            
            st.success(f"✅ Loaded {len(self.battery_data)} batteries with {len(self.processed_data)} total data points")
            return True
            
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            return False
    
    def _identify_cycles(self, df):
        """Identify charge/discharge cycles"""
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
    
    def create_visualizations(self, selected_batteries):
        """Create battery visualizations"""
        filtered_data = self.processed_data[self.processed_data['Battery_ID'].isin(selected_batteries)]
        
        # Voltage vs Time
        fig1 = px.line(
            filtered_data,
            x='Time_s',
            y='Voltage_V',
            color='Battery_ID',
            title='🔋 Battery Voltage Over Time',
            labels={'Time_s': 'Time (seconds)', 'Voltage_V': 'Voltage (V)'}
        )
        fig1.update_layout(height=500)
        
        # Current vs Time
        fig2 = px.line(
            filtered_data,
            x='Time_s',
            y='Current_uA',
            color='Battery_ID',
            title='⚡ Battery Current Over Time',
            labels={'Time_s': 'Time (seconds)', 'Current_uA': 'Current (µA)'}
        )
        fig2.update_layout(height=500)
        
        # Voltage vs Current scatter
        fig3 = px.scatter(
            filtered_data,
            x='Current_uA',
            y='Voltage_V',
            color='Battery_ID',
            title='🔍 Voltage vs Current Relationship',
            labels={'Current_uA': 'Current (µA)', 'Voltage_V': 'Voltage (V)'}
        )
        fig3.update_layout(height=500)
        
        return fig1, fig2, fig3

def main():
    st.title("🔋 Battery Lifecycle Analysis Dashboard")
    st.markdown("---")
    
    # Initialize dashboard
    dashboard = BatteryDashboard()
    
    # Load data
    if not dashboard.load_data():
        st.error("Failed to load battery data.")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Controls")
        
        # Battery selection
        selected_batteries = st.multiselect(
            "Select batteries to analyze",
            options=sorted(dashboard.processed_data['Battery_ID'].unique()),
            default=sorted(dashboard.processed_data['Battery_ID'].unique())[:5]
        )
        
        # Analysis type
        analysis_type = st.selectbox(
            "Analysis Type",
            ["Comprehensive", "Performance", "Lifecycle", "Quality Control"]
        )
    
    # Main content
    if not selected_batteries:
        st.warning("Please select at least one battery to analyze.")
        return
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Batteries", len(dashboard.battery_data))
    
    with col2:
        avg_voltage = dashboard.processed_data['Voltage_V'].mean()
        st.metric("Average Voltage", f"{avg_voltage:.3f}V")
    
    with col3:
        max_cycles = dashboard.processed_data['Cycle_Number'].max()
        st.metric("Max Cycles", max_cycles)
    
    with col4:
        st.metric("Data Points", f"{len(dashboard.processed_data):,}")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Real-Time Monitor", 
        "🔍 Deep Analysis", 
        "📊 Performance Metrics", 
        "💡 Insights"
    ])
    
    with tab1:
        st.header("📈 Real-Time Battery Monitoring")
        
        fig1, fig2, fig3 = dashboard.create_visualizations(selected_batteries)
        
        st.plotly_chart(fig1, use_container_width=True)
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab2:
        st.header("🔍 Deep Analysis")
        
        fig1, fig2, fig3 = dashboard.create_visualizations(selected_batteries)
        st.plotly_chart(fig3, use_container_width=True)
        
        # Additional analysis
        st.subheader("📊 Statistical Summary")
        if dashboard.metrics_df is not None:
            st.dataframe(dashboard.metrics_df, use_container_width=True)
    
    with tab3:
        st.header("📊 Performance Metrics")
        
        if dashboard.metrics_df is not None:
            st.subheader("🔋 Battery Performance Dashboard")
            st.dataframe(dashboard.metrics_df, use_container_width=True)
            
            # Performance charts
            col1, col2 = st.columns(2)
            
            with col1:
                fig_voltage = px.box(
                    dashboard.metrics_df,
                    x='Battery_ID',
                    y='Avg_Voltage',
                    title='Average Voltage by Battery'
                )
                st.plotly_chart(fig_voltage, use_container_width=True)
            
            with col2:
                fig_current = px.box(
                    dashboard.metrics_df,
                    x='Battery_ID',
                    y='Avg_Current',
                    title='Average Current by Battery'
                )
                st.plotly_chart(fig_current, use_container_width=True)
    
    with tab4:
        st.header("💡 Key Insights")
        
        # Calculate insights
        total_batteries = len(dashboard.battery_data)
        total_data_points = len(dashboard.processed_data)
        avg_voltage = dashboard.processed_data['Voltage_V'].mean()
        voltage_std = dashboard.processed_data['Voltage_V'].std()
        
        insights = [
            f"**Total Dataset**: {total_batteries} batteries with {total_data_points:,} data points",
            f"**Voltage Range**: {dashboard.processed_data['Voltage_V'].min():.3f}V - {dashboard.processed_data['Voltage_V'].max():.3f}V",
            f"**Current Range**: {dashboard.processed_data['Current_uA'].min():.1f}µA - {dashboard.processed_data['Current_uA'].max():.1f}µA",
            f"**Average Voltage**: {avg_voltage:.3f}V ± {voltage_std:.3f}V",
        ]
        
        for insight in insights:
            st.markdown(f"<div style='background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin: 10px 0;'>{insight}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
