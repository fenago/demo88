import pandas as pd
import numpy as np
import json
import io
import base64
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request, jsonify
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Add required functions to Jinja2 environment
app.jinja_env.globals.update(zip=zip, list=list)

# Add custom filter for safe rounding
def safe_round(value, precision=0):
    if value is None or value == '':
        return ''
    try:
        return round(float(value), precision)
    except (ValueError, TypeError):
        return value
        
app.jinja_env.filters['safe_round'] = safe_round

# Default data source
DEFAULT_DATA_URL = "https://raw.githubusercontent.com/fenago/datasets/refs/heads/main/taxstats2015.csv"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/tax_analysis', methods=['POST'])
def tax_analysis():
    try:
        # Get parameters from form
        data_url = request.form.get('data_url', DEFAULT_DATA_URL)
        n_clusters = int(request.form.get('n_clusters', 3))
        analysis_type = request.form.get('analysis_type', 'taxpayer')
        
        # Load data based on analysis type
        if analysis_type == 'taxpayer':
            df = pd.read_csv(data_url, usecols=['Postcode', 'Average net tax', 'Average total deductions'])
            feature_cols = ['Average net tax', 'Average total deductions']
        else:  # business analysis
            df = pd.read_csv(data_url, usecols=['Postcode', 'Average total business income', 'Average total business expenses'])
            feature_cols = ['Average total business income', 'Average total business expenses']
        
        # Clean data (remove rows with missing or negative values)
        for col in feature_cols:
            df = df[df[col] > 0].dropna()
        
        # Get X data for clustering
        X = df[feature_cols]
        
        # Standardize data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['cluster'] = kmeans.fit_predict(X_scaled)
        
        # Get cluster centroids (transform back to original scale)
        centroids = scaler.inverse_transform(kmeans.cluster_centers_)
        
        # Calculate cluster statistics in a simplified format
        stats = []
        for cluster_id, group in df.groupby('cluster'):
            stats.append({
                'cluster': int(cluster_id),
                'count': len(group),
                'feature1_mean': float(group[feature_cols[0]].mean()),
                'feature2_mean': float(group[feature_cols[1]].mean())
            })
        cluster_stats_dict = stats
        
        # Generate plots
        plots = generate_plots(df, feature_cols, centroids)
        
        # Prepare response data with simplified structures
        sample_data = []
        for _, row in df.head(10).iterrows():
            sample_data.append({
                'postcode': row['Postcode'],
                'feature1': float(row[feature_cols[0]]),
                'feature2': float(row[feature_cols[1]]),
                'cluster': int(row['cluster'])
            })
            
        # Create a simplified result dictionary
        result = {
            'message': 'Analysis completed successfully',
            'cluster_stats': cluster_stats_dict,
            'plots': plots,
            'sample_data': sample_data,
            'feature_names': {
                'feature1': feature_cols[0],
                'feature2': feature_cols[1]
            }
        }
        
        return render_template('results.html', result=result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/elbow_method', methods=['POST'])
def elbow_method():
    try:
        # Get parameters from form
        data_url = request.form.get('data_url', DEFAULT_DATA_URL)
        analysis_type = request.form.get('analysis_type', 'taxpayer')
        max_clusters = int(request.form.get('max_clusters', 10))
        
        # Load data based on analysis type
        if analysis_type == 'taxpayer':
            df = pd.read_csv(data_url, usecols=['Postcode', 'Average net tax', 'Average total deductions'])
            feature_cols = ['Average net tax', 'Average total deductions']
        else:  # business analysis
            df = pd.read_csv(data_url, usecols=['Postcode', 'Average total business income', 'Average total business expenses'])
            feature_cols = ['Average total business income', 'Average total business expenses']
        
        # Clean data (remove rows with missing or negative values)
        for col in feature_cols:
            df = df[df[col] > 0].dropna()
        
        # Get X data for clustering
        X = df[feature_cols]
        
        # Standardize data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Calculate inertia for different numbers of clusters
        inertia = []
        K = range(1, max_clusters + 1)
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X_scaled)
            inertia.append(kmeans.inertia_)
        
        # Generate elbow plot
        plt.figure(figsize=(10, 6))
        plt.plot(K, inertia, 'bo-')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Inertia')
        plt.title('Elbow Method For Optimal k')
        plt.grid(True)
        
        # Convert plot to base64 image
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        elbow_plot = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        # Prepare response
        result = {
            'message': 'Elbow method analysis completed',
            'elbow_plot': elbow_plot,
            'inertia_values': inertia,
            'k_values': list(K)
        }
        
        return render_template('elbow_results.html', result=result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def generate_plots(df, feature_cols, centroids):
    plots = {}
    
    # 1. Scatter plot of features colored by cluster
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=feature_cols[0], y=feature_cols[1], hue='cluster', data=df, palette='viridis')
    
    # Add centroids to the plot
    centroids_df = pd.DataFrame(centroids, columns=feature_cols)
    centroids_df['cluster'] = range(len(centroids_df))
    plt.scatter(centroids_df[feature_cols[0]], centroids_df[feature_cols[1]], 
               s=200, marker='X', c='red', label='Centroids')
    
    plt.title(f'Clusters based on {feature_cols[0]} and {feature_cols[1]}')
    plt.xlabel(feature_cols[0])
    plt.ylabel(feature_cols[1])
    plt.legend()
    
    # Convert plot to base64 image
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plots['scatter'] = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    
    # 2. Box plot of feature distribution per cluster
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='cluster', y=feature_cols[0], data=df, palette='viridis')
    plt.title(f'Distribution of {feature_cols[0]} across Clusters')
    plt.xlabel('Cluster')
    plt.ylabel(feature_cols[0])
    
    # Convert plot to base64 image
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plots['boxplot_feature1'] = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    
    # 3. Box plot of second feature distribution per cluster
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='cluster', y=feature_cols[1], data=df, palette='viridis')
    plt.title(f'Distribution of {feature_cols[1]} across Clusters')
    plt.xlabel('Cluster')
    plt.ylabel(feature_cols[1])
    
    # Convert plot to base64 image
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plots['boxplot_feature2'] = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    
    return plots

if __name__ == '__main__':
    app.run(debug=True)
