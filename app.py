"""
Data Analysis AI Agent - Streamlit Web Interface
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json
import os
from typing import Optional

# Add src directory to Python path
import sys
sys.path.append('./src')

from data_analysis_agent import DataAnalysisAgent
from data_processor import DataProcessor
from analysis_modules import (
    StatisticalAnalyzer, 
    VisualizationGenerator, 
    MLAnalyzer, 
    TrendAnalyzer,
    ComparativeAnalyzer
)

# Page configuration
st.set_page_config(
    page_title="Data Analysis AI Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def init_session_state():
    """Initialize session state variables"""
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []

def setup_agent():
    """Setup the Data Analysis Agent with API key"""
    if st.session_state.agent is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("Please set your OpenAI API key in the environment variables or use the sidebar.")
            api_key = st.sidebar.text_input("OpenAI API Key", type="password")
            if api_key:
                st.session_state.agent = DataAnalysisAgent(api_key=api_key)
        else:
            st.session_state.agent = DataAnalysisAgent(api_key=api_key)

def main():
    """Main application function"""
    init_session_state()
    
    st.title("🤖 Data Analysis AI Agent")
    st.markdown("### AI-powered data analysis with GPT-4 and interactive visualizations")
    
    # Sidebar for configuration and data upload
    with st.sidebar:
        st.header("Configuration")
        
        # API Key setup
        if st.session_state.agent is None:
            api_key = st.text_input("OpenAI API Key", type="password", 
                                   help="Enter your OpenAI API key to enable GPT-4 analysis")
            if api_key:
                try:
                    st.session_state.agent = DataAnalysisAgent(api_key=api_key)
                    st.success("API key configured successfully!")
                except Exception as e:
                    st.error(f"Error setting up agent: {e}")
        else:
            st.success("✅ AI Agent ready!")
        
        st.markdown("---")
        
        # File upload
        st.header("Data Upload")
        uploaded_file = st.file_uploader(
            "Upload your dataset",
            type=['csv', 'xlsx', 'xls', 'json'],
            help="Supported formats: CSV, Excel, JSON"
        )
        
        if uploaded_file is not None:
            try:
                # Save uploaded file temporarily
                temp_path = f"./data/{uploaded_file.name}"
                os.makedirs("./data", exist_ok=True)
                
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Load data using the agent
                if st.session_state.agent:
                    df = st.session_state.agent.load_data(temp_path)
                    st.session_state.data_loaded = True
                    st.success(f"✅ Data loaded: {df.shape[0]} rows × {df.shape[1]} columns")
                    
                    # Show basic info
                    st.subheader("Dataset Overview")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Rows", df.shape[0])
                    with col2:
                        st.metric("Columns", df.shape[1])
                    with col3:
                        st.metric("Missing Values", df.isnull().sum().sum())
                
            except Exception as e:
                st.error(f"Error loading data: {e}")
    
    # Main content area
    if st.session_state.agent is None:
        st.warning("Please configure your OpenAI API key in the sidebar to get started.")
        return
    
    if not st.session_state.data_loaded:
        st.info("Please upload a dataset to begin analysis.")
        return
    
    # Tabs for different analysis types
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🤖 AI Analysis", 
        "📊 Quick Stats", 
        "📈 Visualizations", 
        "🧮 Advanced Analysis",
        "📋 Data Quality"
    ])
    
    with tab1:
        st.header("AI-Powered Analysis")
        
        # Natural language query interface
        query = st.text_area(
            "Ask me anything about your data:",
            placeholder="e.g., What are the main patterns in this data? Which variables are most correlated? Are there any outliers?",
            height=100
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("🔍 Analyze", type="primary"):
                if query:
                    with st.spinner("Analyzing data with GPT-4..."):
                        try:
                            result = st.session_state.agent.analyze_with_gpt4(query)
                            
                            if 'error' not in result:
                                st.session_state.analysis_history.append(result)
                                
                                st.subheader("Analysis Results")
                                st.markdown(result['analysis'])
                                
                                # Show metadata
                                with st.expander("Analysis Details"):
                                    st.json({
                                        "Query": result['query'],
                                        "Data Shape": result['data_shape'],
                                        "Timestamp": result['timestamp']
                                    })
                            else:
                                st.error(result['error'])
                        except Exception as e:
                            st.error(f"Analysis failed: {e}")
                else:
                    st.warning("Please enter a query to analyze.")
        
        with col2:
            if st.button("💡 Generate Code", type="secondary"):
                if query:
                    with st.spinner("Generating code suggestions..."):
                        try:
                            code = st.session_state.agent.generate_code_suggestion(query)
                            st.subheader("Generated Code")
                            st.code(code, language="python")
                        except Exception as e:
                            st.error(f"Code generation failed: {e}")
        
        # Analysis history
        if st.session_state.analysis_history:
            st.subheader("Analysis History")
            for i, analysis in enumerate(reversed(st.session_state.analysis_history)):
                with st.expander(f"Analysis {len(st.session_state.analysis_history) - i}: {analysis['query'][:50]}..."):
                    st.markdown(analysis['analysis'])
    
    with tab2:
        st.header("Quick Statistical Summary")
        
        if st.session_state.agent.data is not None:
            df = st.session_state.agent.data
            
            # Basic info
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Rows", df.shape[0])
            with col2:
                st.metric("Total Columns", df.shape[1])
            with col3:
                st.metric("Missing Values", df.isnull().sum().sum())
            with col4:
                st.metric("Duplicates", df.duplicated().sum())
            
            # Data preview
            st.subheader("Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Column information
            st.subheader("Column Information")
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes,
                'Non-Null Count': df.count(),
                'Null Count': df.isnull().sum(),
                'Unique Values': df.nunique()
            })
            st.dataframe(col_info, use_container_width=True)
            
            # Basic statistics for numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                st.subheader("Numeric Column Statistics")
                st.dataframe(df[numeric_cols].describe(), use_container_width=True)
    
    with tab3:
        st.header("Data Visualizations")
        
        if st.session_state.agent.data is not None:
            df = st.session_state.agent.data
            
            # Column selection for plotting
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Distribution Plots")
                selected_col = st.selectbox("Select column for distribution", df.columns)
                
                if st.button("Generate Distribution Plot"):
                    fig = VisualizationGenerator.create_distribution_plot(df, selected_col)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Correlation Analysis")
                numeric_cols = df.select_dtypes(include=['number']).columns
                
                if len(numeric_cols) > 1:
                    if st.button("Generate Correlation Heatmap"):
                        fig = VisualizationGenerator.create_correlation_heatmap(df)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Need at least 2 numeric columns for correlation analysis")
            
            # Scatter plots
            st.subheader("Scatter Plot Analysis")
            if len(numeric_cols) >= 2:
                col1, col2, col3 = st.columns(3)
                with col1:
                    x_col = st.selectbox("X-axis", numeric_cols, key="scatter_x")
                with col2:
                    y_col = st.selectbox("Y-axis", numeric_cols, key="scatter_y")
                with col3:
                    color_col = st.selectbox("Color by (optional)", 
                                           ["None"] + list(df.columns), key="scatter_color")
                
                if st.button("Generate Scatter Plot"):
                    color_column = None if color_col == "None" else color_col
                    fig = VisualizationGenerator.create_scatter_plot(df, x_col, y_col, color_column)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("Advanced Analysis")
        
        if st.session_state.agent.data is not None:
            df = st.session_state.agent.data
            
            # Analysis options
            analysis_type = st.selectbox(
                "Select Analysis Type",
                ["Clustering Analysis", "Principal Component Analysis", "Outlier Detection", "Group Comparison"]
            )
            
            if analysis_type == "Clustering Analysis":
                st.subheader("K-Means Clustering")
                n_clusters = st.slider("Number of clusters", 2, 10, 3)
                
                if st.button("Perform Clustering"):
                    with st.spinner("Performing clustering analysis..."):
                        result = MLAnalyzer.perform_clustering(df, n_clusters)
                        
                        if 'error' not in result:
                            st.success("Clustering completed!")
                            st.json(result['cluster_stats'])
                            
                            # Add cluster labels to dataframe for visualization
                            df_with_clusters = df.copy()
                            df_with_clusters['Cluster'] = result['cluster_labels']
                            
                            # Create cluster visualization if we have at least 2 numeric columns
                            numeric_cols = df.select_dtypes(include=['number']).columns
                            if len(numeric_cols) >= 2:
                                fig = VisualizationGenerator.create_scatter_plot(
                                    df_with_clusters, 
                                    numeric_cols[0], 
                                    numeric_cols[1], 
                                    'Cluster'
                                )
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.error(result['error'])
            
            elif analysis_type == "Principal Component Analysis":
                st.subheader("PCA Analysis")
                n_components = st.slider("Number of components", 2, min(5, len(df.select_dtypes(include=['number']).columns)), 2)
                
                if st.button("Perform PCA"):
                    with st.spinner("Performing PCA..."):
                        result = MLAnalyzer.perform_pca(df, n_components)
                        
                        if 'error' not in result:
                            st.success("PCA completed!")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.subheader("Explained Variance")
                                variance_df = pd.DataFrame({
                                    'Component': [f'PC{i+1}' for i in range(n_components)],
                                    'Explained Variance': result['explained_variance_ratio'],
                                    'Cumulative Variance': result['cumulative_variance']
                                })
                                st.dataframe(variance_df)
                            
                            with col2:
                                st.subheader("Feature Loadings")
                                loadings_df = pd.DataFrame(result['feature_loadings'])
                                st.dataframe(loadings_df)
                        else:
                            st.error(result['error'])
            
            elif analysis_type == "Outlier Detection":
                st.subheader("Outlier Detection")
                
                if st.button("Detect Outliers"):
                    with st.spinner("Detecting outliers..."):
                        result = StatisticalAnalyzer.outlier_detection(df)
                        
                        if 'error' not in result:
                            st.success("Outlier detection completed!")
                            
                            for col, outlier_info in result.items():
                                st.subheader(f"Outliers in {col}")
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("IQR Outliers", outlier_info['iqr_outliers']['count'])
                                    st.metric("Percentage", f"{outlier_info['iqr_outliers']['percentage']:.2f}%")
                                
                                with col2:
                                    st.metric("Z-Score Outliers", outlier_info['z_score_outliers']['count'])
                                    st.metric("Percentage", f"{outlier_info['z_score_outliers']['percentage']:.2f}%")
                        else:
                            st.error(result['error'])
            
            elif analysis_type == "Group Comparison":
                st.subheader("Group Comparison Analysis")
                
                categorical_cols = df.select_dtypes(include=['object', 'category']).columns
                numeric_cols = df.select_dtypes(include=['number']).columns
                
                if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                    col1, col2 = st.columns(2)
                    with col1:
                        group_col = st.selectbox("Group by", categorical_cols)
                    with col2:
                        value_col = st.selectbox("Compare values", numeric_cols)
                    
                    test_type = st.selectbox("Statistical Test", 
                                           ["auto", "ttest", "anova", "kruskal"])
                    
                    if st.button("Perform Group Comparison"):
                        with st.spinner("Performing group comparison..."):
                            result = ComparativeAnalyzer.group_comparison(df, group_col, value_col, test_type)
                            
                            if 'error' not in result:
                                st.success("Group comparison completed!")
                                
                                # Show group statistics
                                st.subheader("Group Statistics")
                                stats_df = pd.DataFrame(result['group_statistics'])
                                st.dataframe(stats_df.T)
                                
                                # Show statistical test results
                                if 'error' not in result['statistical_test']:
                                    st.subheader("Statistical Test Results")
                                    st.json(result['statistical_test'])
                                else:
                                    st.error(result['statistical_test']['error'])
                            else:
                                st.error(result['error'])
                else:
                    st.info("Need both categorical and numeric columns for group comparison")
    
    with tab5:
        st.header("Data Quality Report")
        
        if st.session_state.agent.data is not None:
            df = st.session_state.agent.data
            
            if st.button("Generate Quality Report"):
                with st.spinner("Generating data quality report..."):
                    report = DataProcessor.get_data_quality_report(df)
                    
                    # Overview metrics
                    st.subheader("Overview")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Rows", report['overview']['total_rows'])
                    with col2:
                        st.metric("Total Columns", report['overview']['total_columns'])
                    with col3:
                        st.metric("Memory Usage (MB)", f"{report['overview']['memory_usage_mb']:.2f}")
                    with col4:
                        st.metric("Duplicate Rows", report['overview']['duplicate_rows'])
                    
                    # Missing data analysis
                    st.subheader("Missing Data Analysis")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Missing Values", report['missing_data']['total_missing'])
                    with col2:
                        st.metric("Rows with Missing Data", report['missing_data']['rows_with_missing'])
                    
                    # Column analysis
                    st.subheader("Column Analysis")
                    col_analysis_df = pd.DataFrame(report['column_analysis']).T
                    st.dataframe(col_analysis_df, use_container_width=True)
            
            # Data cleaning options
            st.subheader("Data Cleaning Options")
            
            col1, col2 = st.columns(2)
            with col1:
                remove_duplicates = st.checkbox("Remove duplicates", value=True)
                convert_types = st.checkbox("Auto-convert data types", value=True)
            
            with col2:
                handle_missing = st.selectbox("Handle missing values", 
                                            ["auto", "drop", "fill", "none"])
            
            if st.button("Clean Data"):
                with st.spinner("Cleaning data..."):
                    cleaned_df = DataProcessor.clean_data(
                        df, 
                        remove_duplicates=remove_duplicates,
                        handle_missing=handle_missing,
                        convert_types=convert_types
                    )
                    
                    # Update the agent's data
                    st.session_state.agent.data = cleaned_df
                    
                    st.success(f"Data cleaned! New shape: {cleaned_df.shape}")
                    st.dataframe(cleaned_df.head(), use_container_width=True)

if __name__ == "__main__":
    main()
