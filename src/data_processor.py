"""
Data Processing Utilities for the Data Analysis AI Agent
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import re
from datetime import datetime

class DataProcessor:
    """Utility class for data processing and cleaning operations"""
    
    @staticmethod
    def detect_data_types(df: pd.DataFrame) -> Dict[str, str]:
        """
        Automatically detect and suggest appropriate data types for columns
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary mapping column names to suggested data types
        """
        suggestions = {}
        
        for col in df.columns:
            series = df[col].dropna()
            
            if series.empty:
                suggestions[col] = 'object'
                continue
                
            # Try to detect datetime
            if DataProcessor._is_datetime_column(series):
                suggestions[col] = 'datetime64'
            # Try to detect numeric
            elif DataProcessor._is_numeric_column(series):
                if series.dtype == 'int64' or all(float(x).is_integer() for x in series if pd.notna(x)):
                    suggestions[col] = 'int64'
                else:
                    suggestions[col] = 'float64'
            # Try to detect boolean
            elif DataProcessor._is_boolean_column(series):
                suggestions[col] = 'bool'
            # Try to detect categorical
            elif DataProcessor._is_categorical_column(series, df.shape[0]):
                suggestions[col] = 'category'
            else:
                suggestions[col] = 'object'
                
        return suggestions
    
    @staticmethod
    def _is_datetime_column(series: pd.Series) -> bool:
        """Check if a series contains datetime values"""
        try:
            pd.to_datetime(series.head(100), errors='raise')
            return True
        except:
            return False
    
    @staticmethod
    def _is_numeric_column(series: pd.Series) -> bool:
        """Check if a series contains numeric values"""
        try:
            pd.to_numeric(series.head(100), errors='raise')
            return True
        except:
            return False
    
    @staticmethod
    def _is_boolean_column(series: pd.Series) -> bool:
        """Check if a series contains boolean values"""
        unique_vals = set(str(x).lower() for x in series.unique() if pd.notna(x))
        bool_vals = {'true', 'false', '1', '0', 'yes', 'no', 'y', 'n'}
        return len(unique_vals) <= 2 and unique_vals.issubset(bool_vals)
    
    @staticmethod
    def _is_categorical_column(series: pd.Series, total_rows: int) -> bool:
        """Check if a series should be categorical"""
        unique_ratio = len(series.unique()) / total_rows
        return unique_ratio < 0.05 and len(series.unique()) < 50
    
    @staticmethod
    def clean_data(df: pd.DataFrame, 
                   remove_duplicates: bool = True,
                   handle_missing: str = 'auto',
                   convert_types: bool = True) -> pd.DataFrame:
        """
        Comprehensive data cleaning function
        
        Args:
            df: Input DataFrame
            remove_duplicates: Whether to remove duplicate rows
            handle_missing: Strategy for missing values ('auto', 'drop', 'fill', 'none')
            convert_types: Whether to automatically convert data types
            
        Returns:
            Cleaned DataFrame
        """
        cleaned_df = df.copy()
        
        # Remove duplicates
        if remove_duplicates:
            cleaned_df = cleaned_df.drop_duplicates()
        
        # Handle missing values
        if handle_missing == 'auto':
            cleaned_df = DataProcessor._handle_missing_auto(cleaned_df)
        elif handle_missing == 'drop':
            cleaned_df = cleaned_df.dropna()
        elif handle_missing == 'fill':
            cleaned_df = DataProcessor._fill_missing_values(cleaned_df)
        
        # Convert data types
        if convert_types:
            type_suggestions = DataProcessor.detect_data_types(cleaned_df)
            cleaned_df = DataProcessor.convert_data_types(cleaned_df, type_suggestions)
        
        return cleaned_df
    
    @staticmethod
    def _handle_missing_auto(df: pd.DataFrame) -> pd.DataFrame:
        """Automatically handle missing values based on data type and percentage"""
        for col in df.columns:
            missing_pct = df[col].isnull().sum() / len(df)
            
            # Drop columns with >50% missing values
            if missing_pct > 0.5:
                df = df.drop(columns=[col])
                continue
            
            # Fill missing values based on data type
            if df[col].dtype in ['int64', 'float64']:
                df[col] = df[col].fillna(df[col].median())
            elif df[col].dtype == 'object':
                df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown')
            elif df[col].dtype == 'datetime64[ns]':
                df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else pd.Timestamp.now())
        
        return df
    
    @staticmethod
    def _fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values with appropriate defaults"""
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                df[col] = df[col].fillna(df[col].mean())
            else:
                df[col] = df[col].fillna('Missing')
        return df
    
    @staticmethod
    def convert_data_types(df: pd.DataFrame, type_mapping: Dict[str, str]) -> pd.DataFrame:
        """
        Convert DataFrame columns to specified data types
        
        Args:
            df: Input DataFrame
            type_mapping: Dictionary mapping column names to target data types
            
        Returns:
            DataFrame with converted types
        """
        converted_df = df.copy()
        
        for col, dtype in type_mapping.items():
            if col not in converted_df.columns:
                continue
                
            try:
                if dtype == 'datetime64':
                    converted_df[col] = pd.to_datetime(converted_df[col], errors='coerce')
                elif dtype == 'category':
                    converted_df[col] = converted_df[col].astype('category')
                elif dtype in ['int64', 'float64', 'bool']:
                    converted_df[col] = converted_df[col].astype(dtype)
            except Exception as e:
                print(f"Warning: Could not convert {col} to {dtype}: {e}")
                
        return converted_df
    
    @staticmethod
    def get_data_quality_report(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a comprehensive data quality report
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary containing data quality metrics
        """
        report = {
            "overview": {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
                "duplicate_rows": df.duplicated().sum()
            },
            "column_analysis": {},
            "missing_data": {
                "total_missing": df.isnull().sum().sum(),
                "missing_by_column": df.isnull().sum().to_dict(),
                "rows_with_missing": df.isnull().any(axis=1).sum()
            }
        }
        
        # Analyze each column
        for col in df.columns:
            col_data = df[col]
            unique_vals = col_data.nunique()
            
            report["column_analysis"][col] = {
                "data_type": str(col_data.dtype),
                "unique_values": unique_vals,
                "unique_ratio": unique_vals / len(df) if len(df) > 0 else 0,
                "missing_count": col_data.isnull().sum(),
                "missing_percentage": (col_data.isnull().sum() / len(df)) * 100 if len(df) > 0 else 0,
                "most_common": col_data.mode().iloc[0] if not col_data.mode().empty else None
            }
            
            # Add statistics for numeric columns
            if col_data.dtype in ['int64', 'float64']:
                report["column_analysis"][col].update({
                    "min": col_data.min(),
                    "max": col_data.max(),
                    "mean": col_data.mean(),
                    "std": col_data.std(),
                    "outliers": DataProcessor._detect_outliers(col_data)
                })
        
        return report
    
    @staticmethod
    def _detect_outliers(series: pd.Series) -> int:
        """Detect outliers using IQR method"""
        if series.dtype not in ['int64', 'float64']:
            return 0
            
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        return len(outliers)
    
    @staticmethod
    def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names to snake_case and remove special characters
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with standardized column names
        """
        standardized_df = df.copy()
        
        new_columns = []
        for col in df.columns:
            # Convert to lowercase
            new_col = str(col).lower()
            
            # Replace spaces and special characters with underscores
            new_col = re.sub(r'[^a-z0-9_]', '_', new_col)
            
            # Remove multiple consecutive underscores
            new_col = re.sub(r'_+', '_', new_col)
            
            # Remove leading/trailing underscores
            new_col = new_col.strip('_')
            
            new_columns.append(new_col)
        
        standardized_df.columns = new_columns
        return standardized_df
