# 🤖 Data Analysis AI Agent

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/Streamlit-1.28+-red.svg" alt="Streamlit">
  <img src="https://img.shields.io/badge/OpenAI-GPT--4-green.svg" alt="OpenAI GPT-4">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</div>

<div align="center">
  <h3>🚀 An intelligent data analysis tool powered by GPT-4</h3>
  <p>Transform your data exploration with natural language queries, automated insights, and interactive visualizations</p>
</div>

---

## 📋 Table of Contents

- [✨ Features](#-features)
- [🚀 Quick Start](#-quick-start)
- [📖 Usage Guide](#-usage-guide)
- [🏗️ Project Structure](#️-project-structure)
- [🔧 API Reference](#-api-reference)
- [💡 Example Queries](#-example-queries)
- [📊 Supported Data Formats](#-supported-data-formats)
- [⚙️ Configuration](#️-configuration)
- [❓ Troubleshooting](#-troubleshooting)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)
- [🆘 Support](#-support)

## ✨ Features

<div align="center">
  <table>
    <tr>
      <td align="center"><img src="https://img.shields.io/badge/🧠-AI--Powered-blue.svg" alt="AI-Powered"></td>
      <td align="center"><img src="https://img.shields.io/badge/📊-Analytics-green.svg" alt="Analytics"></td>
      <td align="center"><img src="https://img.shields.io/badge/🎨-Visualizations-purple.svg" alt="Visualizations"></td>
      <td align="center"><img src="https://img.shields.io/badge/🧹-Data_Processing-orange.svg" alt="Data Processing"></td>
    </tr>
  </table>
</div>

### 🧠 AI-Powered Analysis

| Feature | Description |
|---------|-------------|
| 💬 **Natural Language Queries** | Ask questions about your data in plain English |
| 🚀 **GPT-4 Integration** | Leverage advanced AI for intelligent data insights |
| 🔧 **Code Generation** | Get Python code suggestions for custom analysis |
| 📚 **Analysis History** | Track your analysis queries and results |

### 📊 Comprehensive Analytics

| Analysis Type | Capabilities |
|---------------|-------------|
| 📈 **Descriptive Statistics** | Automatic statistical summaries and data profiling |
| 🔗 **Correlation Analysis** | Identify relationships between variables with heatmaps |
| 🎯 **Outlier Detection** | Spot anomalies using IQR and Z-score methods |
| 🎲 **Clustering** | K-means clustering with interactive visualizations |
| 📐 **PCA** | Principal Component Analysis for dimensionality reduction |
| ⚖️ **Group Comparisons** | Statistical testing (t-test, ANOVA, Kruskal-Wallis) |

### 🎨 Interactive Visualizations

| Visualization | Description |
|---------------|-------------|
| 📊 **Distribution Plots** | Histograms, bar charts, and density plots |
| 🌡️ **Correlation Heatmaps** | Visual correlation matrices with color coding |
| ⚫ **Scatter Plots** | Relationship analysis with trend lines and grouping |
| 📉 **Time Series** | Temporal trend analysis and seasonal patterns |
| 🎯 **Custom Charts** | Plotly-powered interactive and responsive visualizations |

### 🧹 Data Processing

| Feature | Supported Formats |
|---------|------------------|
| 📂 **Multi-format Support** | CSV, Excel (`.xlsx`, `.xls`), JSON, Parquet |
| 🧽 **Data Cleaning** | Automated cleaning, missing value handling, duplicate removal |
| 🏷️ **Type Detection** | Smart data type inference and conversion |
| 📋 **Quality Reports** | Comprehensive data quality assessment and profiling |

## 🚀 Quick Start

### 📝 Prerequisites

Before you begin, ensure you have:

- **Python 3.8+** installed on your system
- An **OpenAI API key** ([Get one here](https://platform.openai.com/api-keys))
- **Internet connection** for AI analysis features

### 1. 📎 Installation

#### Option A: Using Git (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd Data-Analysis-AI-Agent

# Create a virtual environment (recommended)
python -m venv data-analysis-env

# Activate virtual environment
# On Windows:
data-analysis-env\Scripts\activate
# On macOS/Linux:
source data-analysis-env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### Option B: Direct Download

```bash
# Download and extract the project
cd Data-Analysis-AI-Agent

# Install dependencies
pip install -r requirements.txt
```

### 2. ⚙️ Configuration

#### Step 1: Create Environment File

```bash
# Copy the environment template
cp .env.template .env
```

#### Step 2: Configure API Key

Edit the `.env` file and add your OpenAI API key:

```env
# Required: Your OpenAI API key
OPENAI_API_KEY=sk-your_actual_openai_api_key_here

# Optional: Customize other settings
STREAMLIT_SERVER_PORT=8501
MAX_FILE_SIZE_MB=100
```

> ⚠️ **Important**: Never commit your `.env` file to version control. Keep your API key secure!

#### Step 3: Test Configuration (Optional)

```bash
# Run the setup script to verify everything is working
python run.py
```

### 3. 🎆 Launch the Application

#### Method 1: Using Streamlit directly

```bash
streamlit run app.py
```

#### Method 2: Using the startup script (Recommended)

```bash
python run.py
```

### 4. 🌍 Access the Application

The web interface will automatically open at:

🔗 **http://localhost:8501**

If it doesn't open automatically, copy and paste the URL into your browser.

### 5. 📡 First Time Setup

1. **API Key Verification**: The sidebar will show "✅ AI Agent ready!" if your API key is configured correctly
2. **Upload Sample Data**: Try uploading the included `data/sample_sales_data.csv` file
3. **Test Analysis**: Ask a simple question like "What are the main patterns in this data?"

### 🎉 You're Ready!

Congratulations! Your Data Analysis AI Agent is now running. Start exploring your data with natural language queries!

## 📖 Usage Guide

### 🏁 Getting Started

| Step | Action | Description |
|------|--------|--------------|
| 1️⃣ | **Configure API Key** | Enter your OpenAI API key in the sidebar |
| 2️⃣ | **Upload Data** | Use the file uploader to load your dataset |
| 3️⃣ | **Start Analyzing** | Use any of the analysis tabs to explore your data |

---

### 🤖 AI Analysis Tab

**Purpose**: Natural language querying and AI-powered insights

#### Features:
- 💬 **Natural Language Interface**: Type questions in plain English
- 🧠 **GPT-4 Analysis**: Get intelligent insights and recommendations
- 💻 **Code Generation**: Receive Python code suggestions
- 📈 **Analysis History**: Track your previous queries and results

#### Example Queries:
```
📊 "What are the main trends in this dataset?"
🔗 "Which variables are most correlated?"
🎯 "Are there any unusual patterns or outliers?"
🏆 "Compare sales performance across regions"
📅 "Show me seasonal patterns in the data"
🔍 "What insights can you provide about customer behavior?"
```

---

### 📊 Quick Stats Tab

**Purpose**: Instant dataset overview and basic statistics

#### What You'll See:
- 📄 **Dataset Overview**: Rows, columns, missing values, duplicates
- 🔍 **Data Preview**: First 10 rows of your dataset
- 📋 **Column Information**: Data types, null counts, unique values
- 📈 **Statistical Summary**: Mean, median, std dev for numeric columns

#### Key Metrics:
| Metric | Description |
|--------|-------------|
| **Total Rows** | Number of records in your dataset |
| **Total Columns** | Number of variables/features |
| **Missing Values** | Count of null/empty values |
| **Duplicates** | Number of duplicate rows |

---

### 📈 Visualizations Tab

**Purpose**: Create interactive charts and visual analysis

#### Available Visualizations:

| Visualization | Use Case | Features |
|---------------|----------|----------|
| 📊 **Distribution Plots** | Understand data distribution | Histograms, bar charts, automatic binning |
| 🌡️ **Correlation Heatmaps** | Find variable relationships | Color-coded correlation matrix |
| ⚫ **Scatter Plots** | Explore relationships | Trend lines, color grouping, interactive zoom |

#### How to Use:
1. Select your desired visualization type
2. Choose columns from dropdown menus
3. Click generate to create interactive plots
4. Hover over plots for detailed information

---

### 🧮 Advanced Analysis Tab

**Purpose**: Sophisticated statistical analysis and machine learning

#### Analysis Options:

| Analysis | Description | When to Use |
|----------|-------------|-------------|
| 🎲 **Clustering** | Group similar data points using K-means | Finding customer segments, data patterns |
| 📐 **PCA** | Reduce dimensionality, find principal components | High-dimensional data, feature reduction |
| 🎯 **Outlier Detection** | Identify anomalous data points | Data quality, fraud detection |
| ⚖️ **Group Comparison** | Statistical testing between categories | A/B testing, group differences |

#### Configuration Options:
- **Clustering**: Choose number of clusters (2-10)
- **PCA**: Select number of components to extract
- **Outlier Detection**: Automatic using IQR and Z-score methods
- **Group Comparison**: Automatic test selection (t-test, ANOVA, Kruskal-Wallis)

---

### 📋 Data Quality Tab

**Purpose**: Assess and improve data quality

#### Quality Report Features:
- 📈 **Overview Metrics**: Basic dataset information
- ❌ **Missing Data Analysis**: Identify and quantify missing values
- 📋 **Column Analysis**: Detailed per-column quality assessment

#### Data Cleaning Options:

| Option | Description | Effect |
|--------|-------------|--------|
| 🧽 **Remove Duplicates** | Eliminate duplicate rows | Reduces dataset size, improves quality |
| 🏷️ **Auto-convert Types** | Smart data type conversion | Better analysis performance |
| ❓ **Handle Missing Values** | Various strategies for null values | Choose: auto, drop, fill, or none |

#### Missing Value Strategies:
- **Auto**: Intelligent handling based on data type
- **Drop**: Remove rows/columns with missing values
- **Fill**: Replace with mean/median/mode
- **None**: Keep data as-is

## 🏗️ Project Structure

```
Data-Analysis-AI-Agent/
├── 🖥️ app.py                      # Main Streamlit web application
├── 🚀 run.py                      # Startup script with environment checks
├── 📋 requirements.txt            # Python dependencies and versions
├── ⚙️ .env.template              # Environment variables template
├── 📄 README.md                  # Project documentation (this file)
│
├── 📁 src/                        # Core application modules
│   ├── 🤖 data_analysis_agent.py # Main AI agent with GPT-4 integration
│   ├── 🧹 data_processor.py      # Data loading, cleaning, and quality assessment
│   └── 📈 analysis_modules.py    # Statistical analysis and ML algorithms
│
├── 📁 config/                    # Configuration management
│   └── ⚙️ config.py             # Environment and application settings
│
├── 📁 data/                      # Sample datasets and user uploads
│   └── 📈 sample_sales_data.csv # Example dataset for testing
│
└── 📁 tests/                     # Unit tests and test utilities
    └── 🧪 (Coming soon)          # Test files for quality assurance
```

### 📂 Key Files Description

| File/Directory | Purpose | Key Features |
|---------------|---------|-------------|
| `app.py` | Main web interface | Streamlit UI, tabs, file upload, visualization |
| `run.py` | Application launcher | Environment validation, dependency checks |
| `data_analysis_agent.py` | Core AI logic | GPT-4 integration, natural language processing |
| `data_processor.py` | Data handling | File loading, cleaning, quality reports |
| `analysis_modules.py` | Analytics engine | Statistics, ML algorithms, visualizations |
| `config.py` | Configuration | Environment variables, application settings |

## API Reference

### DataAnalysisAgent Class

Main class for AI-powered data analysis:

```python
from src.data_analysis_agent import DataAnalysisAgent

# Initialize agent
agent = DataAnalysisAgent(api_key="your_openai_key")

# Load data
df = agent.load_data("path/to/your/data.csv")

# Analyze with natural language
result = agent.analyze_with_gpt4("What patterns do you see in this data?")

# Generate code suggestions
code = agent.generate_code_suggestion("Create a correlation matrix")
```

### Analysis Modules

Statistical analysis utilities:

```python
from src.analysis_modules import StatisticalAnalyzer, VisualizationGenerator

# Descriptive statistics
stats = StatisticalAnalyzer.descriptive_statistics(df)

# Create visualizations
fig = VisualizationGenerator.create_correlation_heatmap(df)
```

### Data Processing

Data cleaning and preprocessing:

```python
from src.data_processor import DataProcessor

# Clean data automatically
cleaned_df = DataProcessor.clean_data(df)

# Generate quality report
report = DataProcessor.get_data_quality_report(df)
```

## 💡 Example Queries

Here are categorized examples of natural language queries you can use with the AI agent:

### 🔍 General Analysis

| Query | Expected Insight |
|-------|------------------|
| "📊 Give me an overview of this dataset" | Basic statistics, data types, structure summary |
| "🔍 What are the key insights from this data?" | Main patterns, trends, notable findings |
| "📈 Summarize the main trends and patterns" | Statistical trends, correlations, distributions |
| "❓ What story does this data tell?" | High-level narrative and business insights |
| "🏆 What are the most important variables?" | Feature importance and impact analysis |

### 🎯 Specific Analysis

| Query | Analysis Type |
|-------|---------------|
| "💰 Which product category has the highest average sales?" | Categorical comparison |
| "🔗 Is there a correlation between marketing spend and sales?" | Correlation analysis |
| "📅 Show me sales trends over time" | Time series analysis |
| "🎯 Are there any outliers in the sales data?" | Anomaly detection |
| "📉 What factors predict customer churn?" | Predictive analysis |
| "📊 How seasonal is this business?" | Seasonal pattern analysis |

### ⚖️ Comparative Analysis

| Query | Comparison Type |
|-------|----------------|
| "🌍 Compare sales performance across different regions" | Geographic analysis |
| "🏂 How do enterprise and consumer segments differ?" | Segment comparison |
| "📅 Which month had the best sales performance?" | Temporal comparison |
| "👥 Compare customer behavior between age groups" | Demographic analysis |
| "🏆 What's the difference between top and bottom performers?" | Performance analysis |

### 🔮 Advanced Queries

| Query | Advanced Feature |
|-------|------------------|
| "🤖 Generate code to create a predictive model" | Code generation |
| "📈 Create a visualization showing the relationship between X and Y" | Custom visualization |
| "🔍 Find clusters in customer data and describe them" | Machine learning analysis |
| "📊 Perform statistical significance testing on these groups" | Statistical testing |
| "🎯 Identify the most important features for predicting sales" | Feature analysis |

---

## 📊 Supported Data Formats

<div align="center">
  <table>
    <tr>
      <td align="center">
        <img src="https://img.shields.io/badge/CSV-Supported-green.svg" alt="CSV">
        <br><strong>CSV Files</strong>
        <br><code>.csv</code>
        <br><em>Comma-separated values</em>
      </td>
      <td align="center">
        <img src="https://img.shields.io/badge/Excel-Supported-blue.svg" alt="Excel">
        <br><strong>Excel Files</strong>
        <br><code>.xlsx, .xls</code>
        <br><em>Microsoft Excel formats</em>
      </td>
      <td align="center">
        <img src="https://img.shields.io/badge/JSON-Supported-orange.svg" alt="JSON">
        <br><strong>JSON Files</strong>
        <br><code>.json</code>
        <br><em>JavaScript Object Notation</em>
      </td>
      <td align="center">
        <img src="https://img.shields.io/badge/Parquet-Supported-purple.svg" alt="Parquet">
        <br><strong>Parquet Files</strong>
        <br><code>.parquet</code>
        <br><em>Apache Parquet format</em>
      </td>
    </tr>
  </table>
</div>

### 📋 Format Details

| Format | Max File Size | Best Use Case | Loading Speed |
|--------|---------------|---------------|---------------|
| **CSV** | 100MB | Simple datasets, universal compatibility | Fast |
| **Excel** | 100MB | Business reports, formatted data | Medium |
| **JSON** | 100MB | Nested/hierarchical data, web APIs | Medium |
| **Parquet** | 100MB | Large datasets, analytics workloads | Very Fast |

---

## ⚙️ Requirements

### 🔧 System Requirements

| Requirement | Version | Purpose |
|-------------|---------|----------|
| **Python** | 3.8+ | Core runtime environment |
| **Memory** | 4GB RAM+ | Data processing and AI analysis |
| **Storage** | 1GB+ | Application and dependencies |
| **Internet** | Stable connection | OpenAI API calls |

### 🔑 API Requirements

- **OpenAI API Key**: Required for GPT-4 analysis features
- **API Credits**: Pay-per-use pricing for analysis queries
- **Rate Limits**: Standard OpenAI API rate limits apply

---

## ⚙️ Configuration

The following environment variables can be configured in your `.env` file:

```bash
# OpenAI Settings
OPENAI_API_KEY=your_api_key
OPENAI_MODEL=gpt-4
MAX_TOKENS=2000

# Streamlit Settings
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=localhost

# Data Settings
MAX_FILE_SIZE_MB=100
SUPPORTED_FORMATS=csv,xlsx,xls,json,parquet

# Analysis Settings
DEFAULT_CORRELATION_THRESHOLD=0.7
DEFAULT_OUTLIER_METHOD=iqr
MAX_CLUSTERS=10
```

## ❓ Troubleshooting

### 🔴 Common Issues

<details>
<summary><b>🔑 API Key Error</b></summary>

**Symptoms:**
- "Please set your OpenAI API key" error message
- "Error setting up agent" in sidebar
- Analysis queries failing

**Solutions:**
- ✅ Ensure your OpenAI API key is correctly set in the `.env` file
- ✅ Verify the API key starts with `sk-` and is complete
- ✅ Check that you have sufficient API credits in your OpenAI account
- ✅ Test your API key using OpenAI's API documentation

</details>

<details>
<summary><b>📁 File Upload Error</b></summary>

**Symptoms:**
- "Error loading data" message
- File upload fails silently
- "Unsupported file format" error

**Solutions:**
- ✅ Verify your file format is supported (CSV, Excel, JSON, Parquet)
- ✅ Check file size doesn't exceed limit (default: 100MB)
- ✅ Ensure the file is not corrupted or password-protected
- ✅ Try uploading a different file to isolate the issue
- ✅ Check file encoding (UTF-8 recommended for CSV files)

</details>

<details>
<summary><b>📈 Analysis Errors</b></summary>

**Symptoms:**
- "Analysis failed" error message
- Blank or incomplete analysis results
- Timeout errors during analysis

**Solutions:**
- ✅ Make sure your data has appropriate column types
- ✅ Check for sufficient data points for the requested analysis
- ✅ Verify stable internet connection for API calls
- ✅ Try simpler queries first to test the system
- ✅ Clean your data using the Data Quality tab before analysis

</details>

<details>
<summary><b>🐎 Installation Issues</b></summary>

**Symptoms:**
- Package installation failures
- Import errors when running the application
- Version conflicts

**Solutions:**
- ✅ Use a virtual environment to avoid conflicts
- ✅ Ensure Python 3.8+ is installed
- ✅ Update pip: `pip install --upgrade pip`
- ✅ Try installing packages individually if batch install fails
- ✅ Check for system-specific requirements (e.g., Visual C++ on Windows)

</details>

### 🚀 Performance Tips

| Tip | Benefit | Implementation |
|-----|---------|---------------|
| 📉 **Use smaller datasets** | Faster processing | Sample large datasets before upload |
| 🧽 **Clean data first** | Better analysis quality | Use Data Quality tab before analysis |
| 🎲 **Limit clusters** | Avoid timeouts | Use 2-5 clusters for large datasets |
| 🎯 **Sample large files** | Reduce memory usage | Use representative subsets of data |
| 🔄 **Cache results** | Faster re-analysis | Save analysis history for reference |

### 🆘 Need More Help?

If you're still experiencing issues:

1. 🔍 Check the [example queries](#-example-queries) section
2. 📚 Review the [usage guide](#-usage-guide) for detailed instructions
3. ⚙️ Verify your [configuration](#️-configuration) settings
4. 📄 Ensure all [requirements](#️-requirements) are met

---

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help make the Data Analysis AI Agent even better.

### 🏁 Getting Started

1. 🍴 **Fork the repository** on your platform
2. 🗺 **Clone your fork** locally
3. 🌱 **Create a feature branch**: `git checkout -b feature/amazing-feature`
4. 🛠️ **Make your changes** with proper testing
5. 📝 **Commit your changes**: `git commit -m 'Add amazing feature'`
6. 🚀 **Push to the branch**: `git push origin feature/amazing-feature`
7. 📨 **Submit a pull request** with a clear description

### 🎨 Types of Contributions

| Type | Examples | Impact |
|------|----------|--------|
| 🐛 **Bug Fixes** | Fix calculation errors, UI issues | High |
| ✨ **New Features** | Additional analysis methods, visualizations | High |
| 📄 **Documentation** | README improvements, code comments | Medium |
| 🗺 **UI/UX** | Better interface design, usability | Medium |
| 🎨 **Code Quality** | Refactoring, optimization | Medium |
| 🧪 **Testing** | Unit tests, integration tests | Medium |

### 📝 Development Guidelines

- 🔍 **Code Style**: Follow PEP 8 for Python code
- 🧪 **Testing**: Add tests for new functionality
- 📝 **Documentation**: Update relevant documentation
- 📋 **Commits**: Use clear, descriptive commit messages
- 🎨 **Features**: Ensure new features are user-friendly

### 🐛 Reporting Issues

Found a bug? Please include:

- 📄 **Description**: Clear description of the issue
- 🔄 **Steps to reproduce**: Detailed reproduction steps
- 💻 **Environment**: OS, Python version, dependencies
- 📈 **Sample data**: If possible, provide sample data (anonymized)
- 📷 **Screenshots**: Visual evidence of the issue

---

## 📄 License

<div align="center">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License">
</div>

This project is licensed under the **MIT License** - see the details below:

```
MIT License

Copyright (c) 2024 Data Analysis AI Agent

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### ⚖️ What This Means

- ✅ **Use**: Free to use for personal and commercial projects
- ✅ **Modify**: Free to modify and adapt the code
- ✅ **Distribute**: Free to distribute original or modified versions
- ✅ **Private Use**: Free to use privately without restrictions
- ⚠️ **Attribution**: Must include original license in distributions
- ⚠️ **No Warranty**: Software provided "as-is" without warranties

---

## 🆘 Support

<div align="center">
  <h3>🚀 Need Help? We're Here to Support You!</h3>
</div>

### 🎯 Quick Help

For immediate assistance, try these resources in order:

1. 🔍 **[Troubleshooting](#-troubleshooting)** - Common issues and solutions
2. 📖 **[Usage Guide](#-usage-guide)** - Detailed feature explanations
3. 💡 **[Example Queries](#-example-queries)** - Sample questions to try
4. ⚙️ **[Configuration](#️-configuration)** - Setup and customization

### 📚 Knowledge Base

| Resource | What You'll Find | Best For |
|----------|------------------|----------|
| **README.md** | Complete project documentation | General understanding |
| **Code Comments** | Detailed implementation notes | Development questions |
| **Example Data** | Sample dataset for testing | Learning the interface |
| **.env.template** | Configuration options | Setup assistance |

### ❓ Still Need Help?

If you can't find what you need:

- 📝 **Create an Issue**: Report bugs or request features
- 💬 **Ask Questions**: Get help from the community
- 📚 **Check Documentation**: Review inline code documentation
- 🤝 **Contribute**: Help improve the project for everyone

### ⚖️ Important Notes

- 🔑 **API Keys**: We cannot provide OpenAI API keys - get yours from OpenAI
- 💸 **API Costs**: You are responsible for OpenAI API usage costs
- 🔒 **Data Privacy**: Your data is processed locally and sent only to OpenAI's API
- 🚀 **Updates**: Check back regularly for new features and improvements

---

<div align="center">
  <h2>🎆 Ready to Explore Your Data?</h2>
  <p><strong>Happy analyzing! 🚀</strong></p>
  <p><em>Transform your data into insights with the power of AI</em></p>
</div>

---

<div align="center">
  <sub>Made with ❤️ by the Data Analysis AI Agent team</sub>
</div>
