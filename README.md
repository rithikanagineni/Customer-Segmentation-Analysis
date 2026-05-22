# 👥 Customer Segmentation Analysis

Machine Learning-powered customer segmentation dashboard using K-Means clustering for behavioral and demographic analysis.

## 🎯 Project Overview

This project segments customers into distinct groups based on their purchasing behavior, demographics, and engagement patterns using unsupervised machine learning techniques.

## 🚀 Features

### Segmentation Capabilities
- ✅ K-Means clustering algorithm
- ✅ Customizable number of segments
- ✅ Feature selection for clustering
- ✅ PCA visualization
- ✅ Automated segment profiling

### Analytics & Insights
- 📊 Segment distribution analysis
- 📈 Comparative segment metrics
- 🎯 Customer behavioral patterns
- 💡 Business recommendations
- 📋 Detailed segment characteristics

### Visualizations
- 2D PCA scatter plots
- Segment distribution pie charts
- Feature comparison bar charts
- Normalized heatmaps
- Distribution histograms

### Data Management
- 📁 Upload CSV/Excel files
- 🎲 Generate sample data
- 💾 Export segmented data
- 📊 Download segment summaries

## 💻 Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup

1. **Install dependencies**
```bash
pip install -r requirements.txt

2. How to run:
streamlit run app.py

📊 How It Works
1. Data Preparation
Load customer data (demographic & behavioral)
Select relevant features for segmentation
Standardize features using StandardScaler
2. Clustering
Apply K-Means algorithm
Determine optimal number of clusters
Assign customers to segments
3. Analysis
Profile each segment
Identify key characteristics
Generate business insights
4. Visualization
PCA for 2D visualization
Compare segments across metrics
Analyze distributions
📁 Data Format
Required columns:

CustomerID: Unique identifier
Age: Customer age
Gender: Male/Female
AnnualIncome: Yearly income
AnnualSpending: Total spending
PurchaseFrequency: Number of purchases
Recency: Days since last purchase
Tenure: Months as customer
Optional columns:

WebsiteVisits
EmailOpenRate
PreferredCategory
AvgTransactionValue
🎓 Machine Learning Techniques
K-Means Clustering
Unsupervised learning algorithm
Groups similar customers together
Optimizes within-cluster variance
Principal Component Analysis (PCA)
Dimensionality reduction
Visualizes high-dimensional data in 2D
Preserves variance in data
Standardization
Scales features to same range
Prevents feature dominance
Improves clustering performance
💡 Business Applications
Marketing Strategies
High-Value Segment: VIP programs, premium services
At-Risk Segment: Re-engagement campaigns
Budget Segment: Value offers, discounts
New Customers: Onboarding programs
Use Cases
🎯 Targeted email campaigns
💰 Pricing optimization
📦 Product recommendations
🎁 Personalized offers
📊 Resource allocation
📈 Key Metrics
Segment Size: Number of customers per segment
Revenue Contribution: % of total revenue
Average Spending: Mean spending per segment
Purchase Frequency: Average purchases
Customer Lifetime Value: Projected value
Churn Risk: Based on recency
🔧 Customization
Adjust Number of Clusters
Python

n_clusters = st.sidebar.slider("Number of Segments", 2, 10, 4)
Select Features
Choose which customer attributes to use for segmentation:

Demographic features
Behavioral metrics
Engagement data
Segment Naming
Customize segment labels based on characteristics

📊 Sample Insights
Example Segment Profiles:

VIP Customers

High spending, frequent purchases
Low recency, long tenure
Premium service candidates
Occasional Buyers

Moderate spending, low frequency
High recency
Re-engagement opportunity
Bargain Hunters

Low spending, high frequency
Price-sensitive
Volume discount targets
🛠️ Technologies Used
Python: Core programming
Streamlit: Web interface
Scikit-learn: ML algorithms
Pandas: Data manipulation
Plotly: Interactive visualizations
NumPy: Numerical operations
📧 Author
Data Analyst Internship Project
Task 2: Customer Segmentation Analysis

📄 License
Educational project for internship purposes.

Built with ❤️ using Python & Machine Learning
