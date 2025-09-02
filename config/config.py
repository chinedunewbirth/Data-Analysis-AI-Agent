"""
Configuration management for Data Analysis AI Agent
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv

class Config:
    """Configuration manager for the application"""
    
    def __init__(self):
        """Initialize configuration from environment variables"""
        load_dotenv()
        
        # OpenAI Configuration
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")
        self.MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2000"))
        
        # Streamlit Configuration
        self.STREAMLIT_SERVER_PORT = int(os.getenv("STREAMLIT_SERVER_PORT", "8501"))
        self.STREAMLIT_SERVER_ADDRESS = os.getenv("STREAMLIT_SERVER_ADDRESS", "localhost")
        
        # Data Configuration
        self.MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "100"))
        self.SUPPORTED_FORMATS = os.getenv("SUPPORTED_FORMATS", "csv,xlsx,xls,json,parquet").split(",")
        self.DATA_DIR = os.getenv("DATA_DIR", "./data")
        
        # Analysis Configuration
        self.DEFAULT_CORRELATION_THRESHOLD = float(os.getenv("DEFAULT_CORRELATION_THRESHOLD", "0.7"))
        self.DEFAULT_OUTLIER_METHOD = os.getenv("DEFAULT_OUTLIER_METHOD", "iqr")
        self.MAX_CLUSTERS = int(os.getenv("MAX_CLUSTERS", "10"))
        
        # Visualization Configuration
        self.PLOT_WIDTH = int(os.getenv("PLOT_WIDTH", "800"))
        self.PLOT_HEIGHT = int(os.getenv("PLOT_HEIGHT", "600"))
        
    def validate_config(self) -> Dict[str, bool]:
        """
        Validate configuration settings
        
        Returns:
            Dictionary indicating validation status for each setting
        """
        validation = {
            "openai_api_key": bool(self.OPENAI_API_KEY),
            "data_directory": os.path.exists(self.DATA_DIR),
            "supported_formats": len(self.SUPPORTED_FORMATS) > 0,
            "valid_ports": 1000 <= self.STREAMLIT_SERVER_PORT <= 65535
        }
        
        return validation
    
    def get_streamlit_config(self) -> Dict[str, Any]:
        """
        Get Streamlit-specific configuration
        
        Returns:
            Dictionary with Streamlit configuration
        """
        return {
            "server.port": self.STREAMLIT_SERVER_PORT,
            "server.address": self.STREAMLIT_SERVER_ADDRESS,
            "server.maxUploadSize": self.MAX_FILE_SIZE_MB,
            "theme.primaryColor": "#1f77b4",
            "theme.backgroundColor": "#ffffff",
            "theme.secondaryBackgroundColor": "#f0f2f6"
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary
        
        Returns:
            Configuration as dictionary
        """
        return {
            "openai": {
                "api_key": "***" if self.OPENAI_API_KEY else None,
                "model": self.OPENAI_MODEL,
                "max_tokens": self.MAX_TOKENS
            },
            "streamlit": {
                "port": self.STREAMLIT_SERVER_PORT,
                "address": self.STREAMLIT_SERVER_ADDRESS
            },
            "data": {
                "max_file_size_mb": self.MAX_FILE_SIZE_MB,
                "supported_formats": self.SUPPORTED_FORMATS,
                "data_dir": self.DATA_DIR
            },
            "analysis": {
                "correlation_threshold": self.DEFAULT_CORRELATION_THRESHOLD,
                "outlier_method": self.DEFAULT_OUTLIER_METHOD,
                "max_clusters": self.MAX_CLUSTERS
            },
            "visualization": {
                "plot_width": self.PLOT_WIDTH,
                "plot_height": self.PLOT_HEIGHT
            }
        }

# Global configuration instance
config = Config()
