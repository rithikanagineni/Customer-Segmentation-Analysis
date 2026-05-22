import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Customer Segmentation Analysis",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    h1 {
        color: #2e7d32;
        font-weight: 700;
    }
    .segment-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.title("👥 Customer Segmentation Analysis")
st.markdown("**Segment customers based on behavior and demographics using Machine Learning**")
st.markdown("---")

# Function to generate sample data
@st.cache_data
def generate_sample_data(n_customers=1000):
    np.random.seed(42)
    
    # Customer demographics
    customer_ids = [f'CUST{str(i).zfill(4)}' for i in range(1, n_customers + 1)]
    ages = np.random.normal(40, 15, n_customers).astype(int)
    ages = np.clip(ages, 18, 80)
    
    genders = np.random.choice(['Male', 'Female'], n_customers, p=[0.48, 0.52])
    
    # Income (correlated with age)
    incomes = ages * 1000 + np.random.normal(20000, 15000, n_customers)
    incomes = np.clip(incomes, 15000, 150000)
    
    # Purchase behavior
    annual_spending = incomes * np.random.uniform(0.1, 0.4, n_customers)
    purchase_frequency = np.random.poisson(15, n_customers) + 1
    avg_transaction_value = annual_spending / purchase_frequency
    
    # Recency (days since last purchase)
    recency = np.random.exponential(30, n_customers).astype(int)
    recency = np.clip(recency, 1, 365)
    
    # Customer tenure (months)
    tenure = np.random.poisson(24, n_customers) + 1
    tenure = np.clip(tenure, 1, 120)
    
    # Product categories purchased
    categories = ['Electronics', 'Clothing', 'Home', 'Beauty', 'Sports', 'Books']
    preferred_category = np.random.choice(categories, n_customers)
    
    # Website visits
    website_visits = np.random.poisson(20, n_customers)
    
    # Email engagement
    email_open_rate = np.random.beta(2, 5, n_customers) * 100
    
    # Create DataFrame
    df = pd.DataFrame({
        'CustomerID': customer_ids,
        'Age': ages,
        'Gender': genders,
        'AnnualIncome': incomes.round(2),
        'AnnualSpending': annual_spending.round(2),
        'PurchaseFrequency': purchase_frequency,
        'AvgTransactionValue': avg_transaction_value.round(2),
        'Recency': recency,
        'Tenure': tenure,
        'PreferredCategory': preferred_category,
        'WebsiteVisits': website_visits,
        'EmailOpenRate': email_open_rate.round(2)
    })
    
    return df

# Function to load data
@st.cache_data
def load_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format!")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

# Sidebar
st.sidebar.header("📁 Data Source")
data_source = st.sidebar.radio(
    "Choose data source:",
    ["Use Sample Data", "Upload Your Data"]
)

df = None
if data_source == "Use Sample Data":
    n_customers = st.sidebar.slider("Number of customers", 100, 5000, 1000, 100)
    df = generate_sample_data(n_customers)
    st.sidebar.success(f"✅ Generated {len(df)} sample customers")
else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload customer data (CSV/Excel)",
        type=['csv', 'xlsx', 'xls']
    )
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        if df is not None:
            st.sidebar.success(f"✅ Loaded {len(df)} customers")

# Main analysis
if df is not None:
    
    # Sidebar - Segmentation Settings
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Segmentation Settings")
    
    # Select features for clustering
    st.sidebar.subheader("Select Features")
    
    available_numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    default_features = ['AnnualSpending', 'PurchaseFrequency', 'Recency', 'Tenure']
    default_features = [f for f in default_features if f in available_numeric_cols]
    
    selected_features = st.sidebar.multiselect(
        "Features for clustering:",
        available_numeric_cols,
        default=default_features if default_features else available_numeric_cols[:4]
    )
    
    if len(selected_features) < 2:
        st.warning("⚠️ Please select at least 2 features for clustering")
        st.stop()
    
    # Number of clusters
    n_clusters = st.sidebar.slider("Number of Segments", 2, 10, 4)
    
    # Perform clustering
    if st.sidebar.button("🔄 Run Segmentation", type="primary"):
        st.session_state.run_clustering = True
    
    if 'run_clustering' not in st.session_state:
        st.session_state.run_clustering = True
    
    if st.session_state.run_clustering:
        
        # Prepare data for clustering
        X = df[selected_features].copy()
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform K-Means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df['Segment'] = kmeans.fit_predict(X_scaled)
        df['Segment'] = df['Segment'].apply(lambda x: f'Segment {x+1}')
        
        # PCA for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        df['PCA1'] = X_pca[:, 0]
        df['PCA2'] = X_pca[:, 1]
        
        # Overview metrics
        st.header("📊 Segmentation Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Customers", f"{len(df):,}")
        
        with col2:
            st.metric("Number of Segments", n_clusters)
        
        with col3:
            avg_segment_size = len(df) / n_clusters
            st.metric("Avg Segment Size", f"{int(avg_segment_size):,}")
        
        with col4:
            variance_explained = sum(pca.explained_variance_ratio_) * 100
            st.metric("Variance Explained", f"{variance_explained:.1f}%")
        
        st.markdown("---")
        
        # Segment distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Segment Distribution")
            
            segment_counts = df['Segment'].value_counts().reset_index()
            segment_counts.columns = ['Segment', 'Count']
            
            fig_dist = px.pie(
                segment_counts,
                values='Count',
                names='Segment',
                title='Customer Distribution Across Segments',
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig_dist, use_container_width=True)
        
        with col2:
            st.subheader("🎯 2D Segment Visualization (PCA)")
            
            fig_pca = px.scatter(
                df,
                x='PCA1',
                y='PCA2',
                color='Segment',
                title='Customer Segments (PCA Projection)',
                color_discrete_sequence=px.colors.qualitative.Set3,
                hover_data=['CustomerID']
            )
            fig_pca.update_traces(marker=dict(size=8, opacity=0.6))
            st.plotly_chart(fig_pca, use_container_width=True)
        
        st.markdown("---")
        
        # Segment characteristics
        st.header("📋 Segment Characteristics")
        
        # Calculate segment statistics
        segment_stats = df.groupby('Segment')[selected_features].mean().round(2)
        
        # Display segment profiles
        segments = sorted(df['Segment'].unique())
        
        for i, segment in enumerate(segments):
            with st.expander(f"**{segment}** - {len(df[df['Segment']==segment])} customers", expanded=(i==0)):
                
                segment_data = df[df['Segment'] == segment]
                
                col1, col2, col3 = st.columns(3)
                
                # Key metrics for this segment
                metrics_to_show = selected_features[:6]  # Show top 6 metrics
                
                for idx, metric in enumerate(metrics_to_show):
                    col = [col1, col2, col3][idx % 3]
                    
                    with col:
                        avg_value = segment_data[metric].mean()
                        overall_avg = df[metric].mean()
                        diff = ((avg_value - overall_avg) / overall_avg * 100)
                        
                        st.metric(
                            label=metric,
                            value=f"{avg_value:,.0f}",
                            delta=f"{diff:+.1f}% vs avg"
                        )
                
                # Demographic breakdown
                if 'Age' in df.columns and 'Gender' in df.columns:
                    st.markdown("**Demographics:**")
                    demo_col1, demo_col2 = st.columns(2)
                    
                    with demo_col1:
                        avg_age = segment_data['Age'].mean()
                        st.write(f"📅 Average Age: **{avg_age:.0f} years**")
                    
                    with demo_col2:
                        gender_dist = segment_data['Gender'].value_counts()
                        st.write(f"👤 Gender: **{gender_dist.index[0]}** ({gender_dist.iloc[0]/len(segment_data)*100:.0f}%)")
                
                # Preferred category
                if 'PreferredCategory' in df.columns:
                    top_category = segment_data['PreferredCategory'].mode()[0]
                    st.write(f"🛍️ Top Category: **{top_category}**")
        
        st.markdown("---")
        
        # Comparative Analysis
        st.header("📊 Comparative Analysis")
        
        # Feature comparison across segments
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Segment Comparison - Key Metrics")
            
            # Select metric to compare
            metric_to_compare = st.selectbox(
                "Select metric to compare:",
                selected_features
            )
            
            comparison_data = df.groupby('Segment')[metric_to_compare].mean().reset_index()
            comparison_data = comparison_data.sort_values(metric_to_compare, ascending=False)
            
            fig_comparison = px.bar(
                comparison_data,
                x='Segment',
                y=metric_to_compare,
                title=f'Average {metric_to_compare} by Segment',
                color=metric_to_compare,
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_comparison, use_container_width=True)
        
        with col2:
            st.subheader("Segment Heatmap")
            
            # Create heatmap data
            heatmap_data = df.groupby('Segment')[selected_features].mean()
            
            # Normalize for better visualization
            heatmap_normalized = (heatmap_data - heatmap_data.mean()) / heatmap_data.std()
            
            fig_heatmap = px.imshow(
                heatmap_normalized.T,
                labels=dict(x="Segment", y="Feature", color="Normalized Value"),
                x=heatmap_normalized.index,
                y=heatmap_normalized.columns,
                color_continuous_scale='RdYlGn',
                title='Segment Feature Heatmap (Normalized)'
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        st.markdown("---")
        
        # Segment Deep Dive
        st.header("🔍 Segment Deep Dive")
        
        selected_segment = st.selectbox(
            "Select a segment to analyze in detail:",
            segments
        )
        
        segment_detail = df[df['Segment'] == selected_segment]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Segment Size", f"{len(segment_detail):,} customers")
            st.metric("% of Total", f"{len(segment_detail)/len(df)*100:.1f}%")
        
        with col2:
            if 'AnnualSpending' in df.columns:
                total_revenue = segment_detail['AnnualSpending'].sum()
                st.metric("Total Revenue", f"${total_revenue:,.0f}")
                st.metric("Avg Revenue/Customer", f"${total_revenue/len(segment_detail):,.0f}")
        
        with col3:
            if 'PurchaseFrequency' in df.columns:
                avg_frequency = segment_detail['PurchaseFrequency'].mean()
                st.metric("Avg Purchase Frequency", f"{avg_frequency:.1f}")
            if 'Recency' in df.columns:
                avg_recency = segment_detail['Recency'].mean()
                st.metric("Avg Recency (days)", f"{avg_recency:.0f}")
        
        # Distribution charts for selected segment
        st.markdown("---")
        st.subheader(f"📊 {selected_segment} - Detailed Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Age' in df.columns:
                fig_age = px.histogram(
                    segment_detail,
                    x='Age',
                    nbins=20,
                    title=f'Age Distribution - {selected_segment}',
                    color_discrete_sequence=['#2e7d32']
                )
                st.plotly_chart(fig_age, use_container_width=True)
        
        with col2:
            if 'AnnualIncome' in df.columns:
                fig_income = px.histogram(
                    segment_detail,
                    x='AnnualIncome',
                    nbins=20,
                    title=f'Income Distribution - {selected_segment}',
                    color_discrete_sequence=['#1976d2']
                )
                st.plotly_chart(fig_income, use_container_width=True)
        
        st.markdown("---")
        
        # Business Insights
        st.header("💡 Business Insights & Recommendations")
        
        # Generate insights based on segments
        insights_col1, insights_col2 = st.columns(2)
        
        with insights_col1:
            st.subheader("🎯 Segment Profiles")
            
            # Identify segment characteristics
            for segment in segments:
                seg_data = df[df['Segment'] == segment]
                
                profile = ""
                
                if 'AnnualSpending' in df.columns:
                    avg_spending = seg_data['AnnualSpending'].mean()
                    if avg_spending > df['AnnualSpending'].quantile(0.75):
                        profile = "🌟 **High-Value Customers**"
                    elif avg_spending < df['AnnualSpending'].quantile(0.25):
                        profile = "💡 **Budget-Conscious Customers**"
                    else:
                        profile = "📊 **Mid-Tier Customers**"
                
                if 'Recency' in df.columns:
                    avg_recency = seg_data['Recency'].mean()
                    if avg_recency > 60:
                        profile += " - At Risk (High Recency)"
                    elif avg_recency < 15:
                        profile += " - Highly Active"
                
                st.markdown(f"**{segment}:** {profile}")
                st.write(f"- Size: {len(seg_data)} customers ({len(seg_data)/len(df)*100:.1f}%)")
                
                if 'AnnualSpending' in df.columns:
                    revenue_contribution = seg_data['AnnualSpending'].sum() / df['AnnualSpending'].sum() * 100
                    st.write(f"- Revenue contribution: {revenue_contribution:.1f}%")
                
                st.markdown("---")
        
        with insights_col2:
            st.subheader("📈 Recommended Actions")
            
            st.markdown("""
            **Segment-Specific Strategies:**
            
            1. **High-Value Segments:**
               - VIP loyalty programs
               - Exclusive early access to new products
               - Premium customer service
            
            2. **At-Risk Segments:**
               - Re-engagement campaigns
               - Special win-back offers
               - Personalized email outreach
            
            3. **Growth Opportunities:**
               - Upselling campaigns
               - Cross-category promotions
               - Referral incentives
            
            4. **Budget-Conscious Segments:**
               - Value bundles
               - Seasonal discounts
               - Loyalty reward points
            """)
        
        st.markdown("---")
        
        # Data table
        st.header("📋 Detailed Customer Data")
        
        # Filter options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            segment_filter = st.multiselect(
                "Filter by segment:",
                segments,
                default=segments
            )
        
        with col2:
            sort_by = st.selectbox("Sort by:", df.columns.tolist())
        
        with col3:
            rows_to_show = st.selectbox("Rows to display:", [10, 25, 50, 100, 'All'])
        
        # Apply filters
        filtered_display = df[df['Segment'].isin(segment_filter)]
        filtered_display = filtered_display.sort_values(sort_by, ascending=False)
        
        if rows_to_show != 'All':
            filtered_display = filtered_display.head(rows_to_show)
        
        st.dataframe(filtered_display, use_container_width=True, height=400)
        
        # Export options
        st.markdown("---")
        st.header("💾 Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export full data with segments
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Complete Data with Segments",
                data=csv,
                file_name=f'customer_segments_{datetime.now().strftime("%Y%m%d")}.csv',
                mime='text/csv',
            )
        
        with col2:
            # Export segment summary
            segment_summary = df.groupby('Segment')[selected_features].mean()
            segment_summary['CustomerCount'] = df.groupby('Segment').size()
            csv_summary = segment_summary.to_csv().encode('utf-8')
            
            st.download_button(
                label="📊 Download Segment Summary",
                data=csv_summary,
                file_name=f'segment_summary_{datetime.now().strftime("%Y%m%d")}.csv',
                mime='text/csv',
            )

else:
    # Welcome screen
    st.info("👋 **Welcome to Customer Segmentation Analysis!**")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 What is Customer Segmentation?
        
        Customer segmentation divides your customer base into distinct groups based on:
        - **Demographics** (age, gender, income)
        - **Behavior** (purchase patterns, frequency)
        - **Engagement** (website visits, email response)
        
        ### ✨ Key Benefits
        
        - 🎯 Targeted marketing campaigns
        - 💰 Improved customer retention
        - 📈 Increased revenue per customer
        - 🎁 Personalized product recommendations
        - 📊 Better resource allocation
        """)
    
    with col2:
        st.markdown("""
        ### 📋 Required Data Format
        
        Your CSV/Excel file should contain:
        
        | Column | Type | Example |
        |--------|------|---------|
        | CustomerID | Text | CUST0001 |
        | Age | Number | 35 |
        | Gender | Text | Male/Female |
        | AnnualIncome | Number | 50000 |
        | AnnualSpending | Number | 15000 |
        | PurchaseFrequency | Number | 12 |
        | Recency | Number | 30 |
        | Tenure | Number | 24 |
        
        **Optional:** WebsiteVisits, EmailOpenRate, PreferredCategory
        """)
    
    st.markdown("---")
    
    st.subheader("📝 Sample Data Preview")
    sample_preview = pd.DataFrame({
        'CustomerID': ['CUST0001', 'CUST0002', 'CUST0003'],
        'Age': [35, 42, 28],
        'Gender': ['Male', 'Female', 'Female'],
        'AnnualIncome': [55000, 78000, 45000],
        'AnnualSpending': [12000, 18000, 9000],
        'PurchaseFrequency': [15, 22, 8],
        'Recency': [10, 5, 45],
        'Tenure': [24, 36, 12]
    })
    st.dataframe(sample_preview, use_container_width=True, hide_index=True)
    
    st.info("💡 **Tip:** Start with the sample data to explore the features, then upload your own data!")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>👥 Customer Segmentation Analysis | Powered by K-Means Clustering & Python</p>
        <p>Data Analyst Internship Project - Task 2</p>
    </div>
    """,
    unsafe_allow_html=True
)