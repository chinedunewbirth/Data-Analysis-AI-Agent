"""
Analysis Modules for the Data Analysis AI Agent
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy import stats
from typing import Dict, List, Optional, Any, Tuple

class StatisticalAnalyzer:
    """Statistical analysis utilities"""
    
    @staticmethod
    def descriptive_statistics(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive descriptive statistics
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary containing descriptive statistics
        """
        results = {
            "numeric_summary": {},
            "categorical_summary": {},
            "overall_stats": {}
        }
        
        # Numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            results["numeric_summary"] = {
                "basic_stats": df[numeric_cols].describe().to_dict(),
                "correlation_matrix": df[numeric_cols].corr().to_dict(),
                "skewness": df[numeric_cols].skew().to_dict(),
                "kurtosis": df[numeric_cols].kurtosis().to_dict()
            }
        
        # Categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            cat_summary = {}
            for col in categorical_cols:
                cat_summary[col] = {
                    "unique_count": df[col].nunique(),
                    "top_values": df[col].value_counts().head(10).to_dict(),
                    "mode": df[col].mode().iloc[0] if not df[col].mode().empty else None
                }
            results["categorical_summary"] = cat_summary
        
        # Overall statistics
        results["overall_stats"] = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "missing_values": df.isnull().sum().sum(),
            "duplicate_rows": df.duplicated().sum(),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024)
        }
        
        return results
    
    @staticmethod
    def correlation_analysis(df: pd.DataFrame, method: str = 'pearson') -> Dict[str, Any]:
        """
        Perform correlation analysis on numeric columns
        
        Args:
            df: Input DataFrame
            method: Correlation method ('pearson', 'spearman', 'kendall')
            
        Returns:
            Correlation analysis results
        """
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            return {"error": "No numeric columns found for correlation analysis"}
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr(method=method)
        
        # Find highly correlated pairs
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:  # High correlation threshold
                    corr_pairs.append({
                        "column1": col1,
                        "column2": col2,
                        "correlation": corr_value
                    })
        
        return {
            "correlation_matrix": corr_matrix.to_dict(),
            "high_correlations": corr_pairs,
            "method": method
        }
    
    @staticmethod
    def outlier_detection(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect outliers in numeric columns using multiple methods
        
        Args:
            df: Input DataFrame
            
        Returns:
            Outlier detection results
        """
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            return {"error": "No numeric columns found for outlier detection"}
        
        outlier_results = {}
        
        for col in numeric_df.columns:
            series = numeric_df[col].dropna()
            
            # IQR method
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            iqr_outliers = series[(series < lower_bound) | (series > upper_bound)]
            
            # Z-score method
            z_scores = np.abs(stats.zscore(series))
            z_outliers = series[z_scores > 3]
            
            outlier_results[col] = {
                "iqr_outliers": {
                    "count": len(iqr_outliers),
                    "percentage": (len(iqr_outliers) / len(series)) * 100,
                    "values": iqr_outliers.tolist()
                },
                "z_score_outliers": {
                    "count": len(z_outliers),
                    "percentage": (len(z_outliers) / len(series)) * 100,
                    "values": z_outliers.tolist()
                }
            }
        
        return outlier_results

class VisualizationGenerator:
    """Generate various data visualizations"""
    
    @staticmethod
    def create_distribution_plot(df: pd.DataFrame, column: str) -> go.Figure:
        """
        Create distribution plot for a given column
        
        Args:
            df: Input DataFrame
            column: Column name to plot
            
        Returns:
            Plotly figure
        """
        if column not in df.columns:
            return None
        
        if df[column].dtype in ['int64', 'float64']:
            # Histogram for numeric data
            fig = px.histogram(df, x=column, title=f'Distribution of {column}',
                             marginal="box", nbins=50)
        else:
            # Bar chart for categorical data
            value_counts = df[column].value_counts().head(20)
            fig = px.bar(x=value_counts.index, y=value_counts.values,
                        title=f'Distribution of {column}',
                        labels={'x': column, 'y': 'Count'})
        
        return fig
    
    @staticmethod
    def create_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
        """
        Create correlation heatmap for numeric columns
        
        Args:
            df: Input DataFrame
            
        Returns:
            Plotly figure
        """
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            return None
        
        corr_matrix = numeric_df.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.round(2).values,
            texttemplate="%{text}",
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title="Correlation Heatmap",
            width=600,
            height=600
        )
        
        return fig
    
    @staticmethod
    def create_scatter_plot(df: pd.DataFrame, x_col: str, y_col: str, 
                          color_col: Optional[str] = None) -> go.Figure:
        """
        Create scatter plot between two numeric columns
        
        Args:
            df: Input DataFrame
            x_col: X-axis column
            y_col: Y-axis column
            color_col: Optional column for color coding
            
        Returns:
            Plotly figure
        """
        if x_col not in df.columns or y_col not in df.columns:
            return None
        
        fig = px.scatter(df, x=x_col, y=y_col, color=color_col,
                        title=f'{y_col} vs {x_col}',
                        trendline="ols" if color_col is None else None)
        
        return fig
    
    @staticmethod
    def create_time_series_plot(df: pd.DataFrame, date_col: str, 
                              value_cols: List[str]) -> go.Figure:
        """
        Create time series plot
        
        Args:
            df: Input DataFrame
            date_col: Date column name
            value_cols: List of value columns to plot
            
        Returns:
            Plotly figure
        """
        if date_col not in df.columns:
            return None
        
        fig = go.Figure()
        
        for col in value_cols:
            if col in df.columns:
                fig.add_trace(go.Scatter(
                    x=df[date_col],
                    y=df[col],
                    mode='lines+markers',
                    name=col
                ))
        
        fig.update_layout(
            title="Time Series Analysis",
            xaxis_title=date_col,
            yaxis_title="Values",
            hovermode='x unified'
        )
        
        return fig

class MLAnalyzer:
    """Machine learning analysis utilities"""
    
    @staticmethod
    def perform_clustering(df: pd.DataFrame, n_clusters: int = 3, 
                          features: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform K-means clustering on the dataset
        
        Args:
            df: Input DataFrame
            n_clusters: Number of clusters
            features: List of feature columns (uses all numeric if None)
            
        Returns:
            Clustering results
        """
        # Select features
        if features is None:
            numeric_df = df.select_dtypes(include=[np.number]).dropna()
        else:
            numeric_df = df[features].dropna()
        
        if numeric_df.empty:
            return {"error": "No suitable features found for clustering"}
        
        # Standardize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(numeric_df)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(scaled_features)
        
        # Calculate cluster statistics
        cluster_stats = {}
        for i in range(n_clusters):
            cluster_mask = cluster_labels == i
            cluster_data = numeric_df[cluster_mask]
            
            cluster_stats[f"cluster_{i}"] = {
                "size": cluster_mask.sum(),
                "percentage": (cluster_mask.sum() / len(cluster_labels)) * 100,
                "mean_values": cluster_data.mean().to_dict(),
                "std_values": cluster_data.std().to_dict()
            }
        
        return {
            "cluster_labels": cluster_labels.tolist(),
            "cluster_centers": kmeans.cluster_centers_.tolist(),
            "cluster_stats": cluster_stats,
            "inertia": kmeans.inertia_,
            "features_used": list(numeric_df.columns)
        }
    
    @staticmethod
    def perform_pca(df: pd.DataFrame, n_components: int = 2,
                   features: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform Principal Component Analysis
        
        Args:
            df: Input DataFrame
            n_components: Number of principal components
            features: List of feature columns (uses all numeric if None)
            
        Returns:
            PCA results
        """
        # Select features
        if features is None:
            numeric_df = df.select_dtypes(include=[np.number]).dropna()
        else:
            numeric_df = df[features].dropna()
        
        if numeric_df.empty:
            return {"error": "No suitable features found for PCA"}
        
        # Standardize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(numeric_df)
        
        # Perform PCA
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(scaled_features)
        
        # Create component DataFrame
        component_columns = [f"PC{i+1}" for i in range(n_components)]
        pca_df = pd.DataFrame(pca_result, columns=component_columns)
        
        return {
            "pca_components": pca_df.to_dict(),
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "cumulative_variance": np.cumsum(pca.explained_variance_ratio_).tolist(),
            "feature_loadings": pd.DataFrame(
                pca.components_.T,
                columns=component_columns,
                index=numeric_df.columns
            ).to_dict(),
            "features_used": list(numeric_df.columns)
        }

class TrendAnalyzer:
    """Time series and trend analysis utilities"""
    
    @staticmethod
    def detect_trends(df: pd.DataFrame, date_col: str, 
                     value_col: str) -> Dict[str, Any]:
        """
        Detect trends in time series data
        
        Args:
            df: Input DataFrame
            date_col: Date column name
            value_col: Value column name
            
        Returns:
            Trend analysis results
        """
        if date_col not in df.columns or value_col not in df.columns:
            return {"error": "Specified columns not found in DataFrame"}
        
        # Sort by date
        df_sorted = df.sort_values(date_col)
        
        # Calculate basic trend metrics
        values = df_sorted[value_col].dropna()
        
        if len(values) < 2:
            return {"error": "Insufficient data for trend analysis"}
        
        # Linear trend
        x = np.arange(len(values))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
        
        # Calculate moving averages
        ma_7 = values.rolling(window=min(7, len(values))).mean()
        ma_30 = values.rolling(window=min(30, len(values))).mean()
        
        # Detect seasonal patterns (if enough data)
        seasonal_info = None
        if len(values) > 24:  # Need at least 24 points for basic seasonal detection
            seasonal_info = TrendAnalyzer._detect_seasonality(values)
        
        return {
            "trend_direction": "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable",
            "trend_strength": abs(r_value),
            "trend_significance": p_value,
            "slope": slope,
            "r_squared": r_value ** 2,
            "moving_averages": {
                "ma_7": ma_7.tolist(),
                "ma_30": ma_30.tolist()
            },
            "seasonal_info": seasonal_info
        }
    
    @staticmethod
    def _detect_seasonality(values: pd.Series) -> Dict[str, Any]:
        """
        Basic seasonality detection using autocorrelation
        
        Args:
            values: Time series values
            
        Returns:
            Seasonality information
        """
        # Calculate autocorrelation for different lags
        lags_to_test = [7, 30, 365] if len(values) > 365 else [7, 30] if len(values) > 30 else [7]
        
        autocorr_results = {}
        for lag in lags_to_test:
            if lag < len(values):
                autocorr = values.autocorr(lag=lag)
                autocorr_results[f"lag_{lag}"] = autocorr
        
        # Find the lag with highest autocorrelation
        if autocorr_results:
            best_lag = max(autocorr_results.keys(), key=lambda k: abs(autocorr_results[k]))
            seasonal_period = int(best_lag.split('_')[1])
            seasonal_strength = abs(autocorr_results[best_lag])
            
            return {
                "seasonal_period": seasonal_period,
                "seasonal_strength": seasonal_strength,
                "autocorrelations": autocorr_results
            }
        
        return None

class ComparativeAnalyzer:
    """Comparative analysis utilities"""
    
    @staticmethod
    def group_comparison(df: pd.DataFrame, group_col: str, 
                        value_col: str, test_type: str = 'auto') -> Dict[str, Any]:
        """
        Compare groups using statistical tests
        
        Args:
            df: Input DataFrame
            group_col: Grouping column
            value_col: Value column to compare
            test_type: Type of test ('auto', 'ttest', 'anova', 'kruskal')
            
        Returns:
            Group comparison results
        """
        if group_col not in df.columns or value_col not in df.columns:
            return {"error": "Specified columns not found in DataFrame"}
        
        # Group statistics
        group_stats = df.groupby(group_col)[value_col].agg([
            'count', 'mean', 'median', 'std', 'min', 'max'
        ]).to_dict()
        
        # Get groups for statistical testing
        groups = [group[value_col].dropna() for name, group in df.groupby(group_col)]
        groups = [g for g in groups if len(g) > 0]  # Remove empty groups
        
        if len(groups) < 2:
            return {
                "group_statistics": group_stats,
                "statistical_test": {"error": "Need at least 2 groups for comparison"}
            }
        
        # Perform statistical test
        test_result = None
        if test_type == 'auto':
            if len(groups) == 2:
                test_result = ComparativeAnalyzer._perform_ttest(groups[0], groups[1])
            else:
                test_result = ComparativeAnalyzer._perform_anova(groups)
        elif test_type == 'ttest' and len(groups) == 2:
            test_result = ComparativeAnalyzer._perform_ttest(groups[0], groups[1])
        elif test_type == 'anova':
            test_result = ComparativeAnalyzer._perform_anova(groups)
        elif test_type == 'kruskal':
            test_result = ComparativeAnalyzer._perform_kruskal(groups)
        
        return {
            "group_statistics": group_stats,
            "statistical_test": test_result
        }
    
    @staticmethod
    def _perform_ttest(group1: pd.Series, group2: pd.Series) -> Dict[str, Any]:
        """Perform independent t-test"""
        try:
            t_stat, p_value = stats.ttest_ind(group1, group2)
            return {
                "test_type": "Independent t-test",
                "t_statistic": t_stat,
                "p_value": p_value,
                "significant": p_value < 0.05,
                "interpretation": "Groups are significantly different" if p_value < 0.05 else "No significant difference"
            }
        except Exception as e:
            return {"error": f"T-test failed: {str(e)}"}
    
    @staticmethod
    def _perform_anova(groups: List[pd.Series]) -> Dict[str, Any]:
        """Perform one-way ANOVA"""
        try:
            f_stat, p_value = stats.f_oneway(*groups)
            return {
                "test_type": "One-way ANOVA",
                "f_statistic": f_stat,
                "p_value": p_value,
                "significant": p_value < 0.05,
                "interpretation": "Groups are significantly different" if p_value < 0.05 else "No significant difference"
            }
        except Exception as e:
            return {"error": f"ANOVA failed: {str(e)}"}
    
    @staticmethod
    def _perform_kruskal(groups: List[pd.Series]) -> Dict[str, Any]:
        """Perform Kruskal-Wallis test (non-parametric alternative to ANOVA)"""
        try:
            h_stat, p_value = stats.kruskal(*groups)
            return {
                "test_type": "Kruskal-Wallis test",
                "h_statistic": h_stat,
                "p_value": p_value,
                "significant": p_value < 0.05,
                "interpretation": "Groups are significantly different" if p_value < 0.05 else "No significant difference"
            }
        except Exception as e:
            return {"error": f"Kruskal-Wallis test failed: {str(e)}"}
