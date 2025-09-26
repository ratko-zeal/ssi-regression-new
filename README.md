# SSI Regression - Ecosystem Maturity Index

A comprehensive system for building and visualizing ecosystem maturity indices using regression models and domain-specific indicators. This project provides both a data processing pipeline and an interactive Streamlit dashboard for analyzing startup ecosystem maturity across different countries and regions.

## 🚀 Overview

The SSI Regression system creates composite indices to measure ecosystem maturity by:

- **Data Processing**: Building regression models using domain-specific indicators
- **Index Calculation**: Computing weighted scores across multiple domains
- **Interactive Visualization**: Providing an intuitive dashboard for exploring results
- **Comparative Analysis**: Enabling country-by-country and regional comparisons

## 📁 Project Structure

```
ssi-regression-new/
├── build_maturity_index.py      # Main data processing and model building script
├── requirements.txt             # Python dependencies for core processing
├── streamlit_app/              # Interactive dashboard application
│   ├── Home.py                 # Main Streamlit application
│   ├── requirements.txt        # Streamlit-specific dependencies
│   └── data/                   # Data files for the dashboard
│       ├── Input_scores.csv    # Raw indicator scores by country
│       ├── domains.csv         # Mapping of indicators to domains
│       ├── country_regions.csv # Country-to-region mappings
│       ├── final_scores.csv    # Computed final scores and rankings
│       └── indicator_scores.csv # Individual indicator scores (0-100)
└── README.md                   # This file
```

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Core Dependencies Installation

```bash
# Install core processing dependencies
pip install -r requirements.txt

# Install Streamlit dashboard dependencies
pip install -r streamlit_app/requirements.txt
```

### Dependencies Overview

**Core Processing** (`requirements.txt`):
- `pandas>=2.0.0` - Data manipulation and analysis
- `numpy>=1.24.0` - Numerical computing
- `scikit-learn>=1.3.0` - Machine learning models (Ridge, ElasticNet)

**Dashboard** (`streamlit_app/requirements.txt`):
- `streamlit>=1.33` - Web application framework
- `plotly>=5.18` - Interactive visualizations
- Plus core dependencies (pandas, numpy, scikit-learn)

## 🔧 Usage

### 1. Data Processing & Model Building

The core processing script builds maturity indices using regression models:

```bash
python build_maturity_index.py
```

**Input Requirements:**
- `domains.csv` - Mapping of indicators to domains
- `Input_scores.csv` - Raw scores for each country and indicator

**Key Features:**
- **Multiple Models**: Ridge and ElasticNet regression with cross-validation
- **Robust Preprocessing**: Winsorization, imputation, and standardization
- **Multi-target Support**: Handles different target variables (normalized, per-capita, log-transformed)
- **Weighted Blending**: Combines multiple model outputs with configurable weights

**Output Structure:**
```
outputs/
├── _manifest_outputs.csv        # Summary of all generated files
├── _metrics.csv                 # Model performance metrics
├── ridge/                       # Ridge regression results
│   └── HG_log_raw/
│       ├── indicator_weights.csv
│       ├── domain_weights.csv
│       └── scores.csv
├── enet/                        # ElasticNet regression results
│   └── HG_log_raw/
│       ├── indicator_weights.csv
│       ├── domain_weights.csv
│       └── scores.csv
└── final/                       # Final composite scores
    ├── final_scores.csv         # Main output with all indices
    └── indicator_scores.csv     # Individual indicator scores
```

### 2. Interactive Dashboard

Launch the Streamlit dashboard for interactive exploration:

```bash
cd streamlit_app
streamlit run Home.py
```

**Dashboard Features:**

#### 🏆 Ecosystem Maturity Breakdown
- **Leaderboard**: Country rankings with maturity categorization
- **Maturity Distribution**: Pie chart showing ecosystem maturity breakdown
- **Categories**:
  - **Mature** (≥55): Well-developed ecosystems
  - **Advancing** (20-54): Growing ecosystems  
  - **Nascent** (≤20): Emerging ecosystems

#### 📊 High Growth Analysis
- **Bubble Chart**: Correlation between index scores and high-growth companies
- **Log-scale Visualization**: Handles wide data ranges effectively
- **Maturity Overlays**: Visual zones showing different maturity levels

#### 🔍 Country Comparison Deep-Dive
- **Domain Radar Charts**: Multi-dimensional comparison across domains
- **Indicator Drill-down**: Detailed analysis by domain and indicator
- **Interactive Filtering**: Regional and country-specific views

## ⚙️ Configuration

### Model Configuration (`build_maturity_index.py`)

Key configuration parameters:

```python
# Data paths
DOMAINS_PATH = "domains.csv"
SCORES_PATH = "Input_scores.csv"

# Target variables
TGT_NORM = "# of High Growth Company - Normalized"
TGT_RAW = "# of High Growth Companies"

# Blend weights for final score
BLEND_W_LOG = 0.60      # Log-transformed model weight
BLEND_W_PERCAP = 0.30   # Per-capita model weight  
BLEND_W_DOMAIN = 0.10   # Domain average weight

# Data preprocessing
ENABLE_WINSORIZE = True  # Outlier handling
WINSOR_LO = 0.01        # 1st percentile clipping
WINSOR_HI = 0.99        # 99th percentile clipping
```

### Dashboard Configuration (`streamlit_app/Home.py`)

Color scheme and display settings:

```python
# Maturity color mapping
COLOR_MATURE = '#ff7433'     # Orange for mature ecosystems
COLOR_ADVANCING = '#59b0F2'  # Blue for advancing ecosystems  
COLOR_NASCENT = '#0865AC'    # Dark blue for nascent ecosystems

# Chart styling
COMPARISON_COLORS = ["#054b81", "#FF7433", "#59b0F2", "#29A0B1", "#686868"]
```

## 📊 Data Requirements

### Input Data Format

#### `domains.csv`
Maps indicators to their respective domains:
```csv
INDICATOR,DOMAIN
Startup_Density,Infrastructure
Venture_Capital_Access,Funding
Research_Output,Innovation
...
```

#### `Input_scores.csv`
Raw indicator scores by country:
```csv
COUNTRY,Startup_Density,Venture_Capital_Access,Research_Output,...
United States,85.2,92.1,78.5,...
United Kingdom,72.8,84.3,71.2,...
...
```

#### Optional Files

- `country_regions.csv`: Regional groupings for analysis
- Population data: For per-capita calculations

## 🔬 Methodology

### Model Architecture

1. **Data Preprocessing**:
   - Winsorization for outlier handling
   - Median imputation for missing values
   - Z-score standardization
   - Zero-variance feature removal

2. **Model Selection**:
   - **Ridge Regression**: L2 regularization with cross-validation
   - **ElasticNet**: Combined L1/L2 regularization
   - Hyperparameter tuning via cross-validation

3. **Score Composition**:
   - Individual model predictions (0-100 scale)
   - Domain-averaged scores
   - Weighted ensemble of multiple approaches

4. **Validation**:
   - Cross-validation for model selection
   - R² and RMSE metrics
   - Feature importance analysis

### Index Calculation

The final maturity index combines multiple approaches:

- **Primary Model** (60%): Log-transformed high-growth companies
- **Per-Capita Model** (30%): Population-adjusted metrics
- **Domain Average** (10%): Unweighted domain means

## 🤝 Contributing

### Development Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Prepare input data in the expected format
4. Run processing: `python build_maturity_index.py`
5. Launch dashboard: `streamlit run streamlit_app/Home.py`

### Extending the System

- **New Indicators**: Add to `domains.csv` and `Input_scores.csv`
- **Custom Models**: Modify the model fitting section in `build_maturity_index.py`
- **Dashboard Features**: Extend `streamlit_app/Home.py` with new visualizations
- **Data Sources**: Update data loading functions for new input formats

## 📈 Use Cases

- **Policy Makers**: Assess regional ecosystem development
- **Investors**: Identify emerging markets and opportunities
- **Researchers**: Analyze factors driving ecosystem growth
- **Entrepreneurs**: Understand market conditions and potential

## 🔍 Troubleshooting

### Common Issues

1. **Missing Data Files**: Ensure `domains.csv` and `Input_scores.csv` are present
2. **Column Name Mismatches**: Check that indicator names match between files
3. **Memory Issues**: For large datasets, consider chunked processing
4. **Display Problems**: Clear Streamlit cache if visualizations don't update

### Performance Optimization

- Use winsorization to handle extreme outliers
- Consider feature selection for high-dimensional data
- Optimize cross-validation parameters for faster processing

## 📝 License

This project is available for educational and research purposes. Please refer to the license file for detailed terms and conditions.

## 📧 Support

For questions, issues, or contributions, please open an issue in the project repository or contact the development team.