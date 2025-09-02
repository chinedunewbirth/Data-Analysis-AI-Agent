"""
Data Analysis AI Agent - Core module for GPT-4 powered data analysis
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from openai import OpenAI
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class DataAnalysisAgent:
    """AI-powered data analysis agent using GPT-4"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Data Analysis Agent
        
        Args:
            api_key: OpenAI API key. If not provided, will look for OPENAI_API_KEY env var
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.data = None
        self.data_info = {}
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from various file formats
        
        Args:
            file_path: Path to the data file
            
        Returns:
            Loaded DataFrame
        """
        try:
            file_extension = os.path.splitext(file_path)[1].lower()
            
            if file_extension == '.csv':
                self.data = pd.read_csv(file_path)
            elif file_extension in ['.xlsx', '.xls']:
                self.data = pd.read_excel(file_path)
            elif file_extension == '.json':
                self.data = pd.read_json(file_path)
            elif file_extension == '.parquet':
                self.data = pd.read_parquet(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            # Store basic info about the dataset
            self.data_info = {
                "shape": self.data.shape,
                "columns": list(self.data.columns),
                "dtypes": self.data.dtypes.to_dict(),
                "memory_usage": self.data.memory_usage(deep=True).sum(),
                "null_counts": self.data.isnull().sum().to_dict()
            }
            
            return self.data
            
        except Exception as e:
            raise Exception(f"Error loading data from {file_path}: {str(e)}")
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of the loaded dataset
        
        Returns:
            Dictionary containing dataset summary information
        """
        if self.data is None:
            return {"error": "No data loaded"}
        
        summary = {
            "basic_info": self.data_info,
            "statistical_summary": self.data.describe().to_dict(),
            "sample_data": self.data.head().to_dict()
        }
        
        return summary
    
    def analyze_with_gpt4(self, query: str, include_data_context: bool = True) -> Dict[str, Any]:
        """
        Analyze data using GPT-4 with natural language queries
        
        Args:
            query: Natural language query about the data
            include_data_context: Whether to include data summary in the prompt
            
        Returns:
            Analysis results from GPT-4
        """
        if self.data is None:
            return {"error": "No data loaded. Please load data first."}
        
        # Prepare data context
        data_context = ""
        if include_data_context:
            summary = self.get_data_summary()
            data_context = f"""
            Dataset Information:
            - Shape: {summary['basic_info']['shape']}
            - Columns: {summary['basic_info']['columns']}
            - Data types: {summary['basic_info']['dtypes']}
            - Missing values: {summary['basic_info']['null_counts']}
            
            Statistical Summary:
            {json.dumps(summary['statistical_summary'], indent=2)}
            
            Sample Data (first 5 rows):
            {json.dumps(summary['sample_data'], indent=2)}
            """
        
        # Create the prompt
        system_prompt = """You are an expert data analyst. Analyze the provided dataset and answer the user's query.
        
        Provide insights in the following format:
        1. Key findings
        2. Specific analysis results
        3. Recommendations for further analysis
        4. Suggested Python code for deeper analysis (if applicable)
        
        Be specific and actionable in your responses."""
        
        user_prompt = f"""
        {data_context}
        
        User Query: {query}
        
        Please provide a comprehensive analysis addressing the user's query.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=2000,
                temperature=0.3
            )
            
            analysis_result = {
                "query": query,
                "analysis": response.choices[0].message.content,
                "data_shape": self.data.shape,
                "timestamp": pd.Timestamp.now().isoformat()
            }
            
            return analysis_result
            
        except Exception as e:
            return {"error": f"Error during GPT-4 analysis: {str(e)}"}
    
    def generate_code_suggestion(self, analysis_goal: str) -> str:
        """
        Generate Python code suggestions for specific analysis tasks
        
        Args:
            analysis_goal: Description of the analysis to perform
            
        Returns:
            Generated Python code as a string
        """
        if self.data is None:
            return "# Error: No data loaded"
        
        prompt = f"""
        Given this dataset with columns {list(self.data.columns)} and shape {self.data.shape},
        generate Python code using pandas, numpy, matplotlib/plotly for: {analysis_goal}
        
        Assume the data is already loaded in a variable called 'df'.
        Provide clean, executable Python code with comments.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.2
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"# Error generating code: {str(e)}"
    
    def execute_analysis_code(self, code: str) -> Dict[str, Any]:
        """
        Safely execute analysis code on the loaded dataset
        
        Args:
            code: Python code to execute
            
        Returns:
            Execution results
        """
        if self.data is None:
            return {"error": "No data loaded"}
        
        # Create a safe execution environment
        safe_globals = {
            'pd': pd,
            'np': np,
            'px': px,
            'go': go,
            'df': self.data.copy(),
            '__builtins__': {}
        }
        
        try:
            # Execute the code
            exec(code, safe_globals)
            
            # Extract any new variables created
            result_vars = {k: v for k, v in safe_globals.items() 
                          if k not in ['pd', 'np', 'px', 'go', 'df', '__builtins__']}
            
            return {
                "success": True,
                "variables": result_vars,
                "execution_info": "Code executed successfully"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "execution_info": "Code execution failed"
            }
