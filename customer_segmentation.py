# ============================================================
# PRODIGY INFOTECH - Machine Learning Internship
# Task 02: K-Means Clustering - Customer Segmentation
# Dataset: Mall Customer Segmentation
# Features: Age, Annual Income, Spending Score
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# 1. LOAD DATASET
# ─────────────────────────────────────────
print("=" * 55)
print("  PRODIGY INFOTECH - ML Task 02")
print("  Customer Segmentation (K-Means Clustering)")
print("=" * 55)

df = pd.read_csv('Mall_Customers.csv')
print(f"\n✅ Dataset loaded successfully!")
print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\n📋 First 5 rows:")
print(df.head())
print(f"\n📊 Dataset Info:")
print(df.info())
print(f"\n📈 Basic Statistics:")
print(df.describe().round(2))

# ─────────────────────────────────────────
# 2. DATA PREPROCESSING
# ─────────────────────────────────────────
# Rename columns for easier access
df.columns = ['CustomerID', 'Gender', 'Age', 'AnnualIncome', 'SpendingScore']

# Encode Gender
df['Gender_Encoded'] = df['Gender'].map({'Male': 0, 'Female': 1})

print(f"\n🔧 Preprocessing done!")
print(f"   Gender encoded: Male=0, Female=1")
print(f"   Missing values: {df.isnull().sum().sum()}")

# ─────────────────────────────────────────
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ─────────────────────────────────────────
colors = ['#4361ee', '#f72585', '#7209b7', '#4cc9f0', '#3a0ca3']

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('Prodigy InfoTech — Task 02: EDA - Customer Segmentation',
             fontsize=15, fontweight='bold', color='#3a0ca3')

# Plot 1: Gender Distribution
gender_counts = df['Gender'].value_counts()
axes[0, 0].pie(gender_counts, labels=gender_counts.index,
               autopct='%1.1f%%', colors=['#4361ee', '#f72585'],
               startangle=90, wedgeprops=dict(edgecolor='white', linewidth=2))
axes[0, 0].set_title('Gender Distribution')

# Plot 2: Age Distribution
axes[0, 1].hist(df['Age'], bins=20, color='#7209b7', edgecolor='white', alpha=0.85)
axes[0, 1].set_title('Age Distribution')
axes[0, 1].set_xlabel('Age')
axes[0, 1].set_ylabel('Count')

# Plot 3: Annual Income Distribution
axes[0, 2].hist(df['AnnualIncome'], bins=20, color='#4361ee', edgecolor='white', alpha=0.85)
axes[0, 2].set_title('Annual Income Distribution')
axes[0, 2].set_xlabel('Annual Income (k$)')
axes[0, 2].set_ylabel('Count')

# Plot 4: Spending Score Distribution
axes[1, 0].hist(df['SpendingScore'], bins=20, color='#f72585', edgecolor='white', alpha=0.85)
axes[1, 0].set_title('Spending Score Distribution')
axes[1, 0].set_xlabel('Spending Score (1-100)')
axes[1, 0].set_ylabel('Count')

# Plot 5: Income vs Spending Score
axes[1, 1].scatter(df['AnnualIncome'], df['SpendingScore'],
                   alpha=0.6, color='#4cc9f0', s=60, edgecolors='white', linewidth=0.5)
axes[1, 1].set_title('Annual Income vs Spending Score')
axes[1, 1].set_xlabel('Annual Income (k$)')
axes[1, 1].set_ylabel('Spending Score')

# Plot 6: Correlation Heatmap
corr = df[['Age', 'AnnualIncome', 'SpendingScore']].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            ax=axes[1, 2], square=True, linewidths=0.5)
axes[1, 2].set_title('Correlation Heatmap')

plt.tight_layout()
plt.savefig('eda_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ EDA plot saved as 'eda_analysis.png'")

# ─────────────────────────────────────────
# 4. FEATURE SELECTION & SCALING
# ─────────────────────────────────────────
# Using Annual Income and Spending Score (classic mall segmentation)
X = df[['AnnualIncome', 'SpendingScore']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("\n✅ Features scaled using StandardScaler")

# ─────────────────────────────────────────
# 5. FIND OPTIMAL K — ELBOW METHOD
# ─────────────────────────────────────────
print("\n🔍 Finding optimal number of clusters...")

inertia = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
    print(f"   K={k} | Inertia: {kmeans.inertia_:.2f} | Silhouette: {silhouette_score(X_scaled, kmeans.labels_):.4f}")

# Plot Elbow + Silhouette
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle('Prodigy InfoTech — Task 02: Optimal K Selection',
              fontsize=14, fontweight='bold', color='#3a0ca3')

# Elbow Curve
axes2[0].plot(list(K_range), inertia, 'bo-', linewidth=2, markersize=8, color='#4361ee')
axes2[0].axvline(x=5, color='red', linestyle='--', linewidth=2, label='Optimal K=5')
axes2[0].set_title('Elbow Method')
axes2[0].set_xlabel('Number of Clusters (K)')
axes2[0].set_ylabel('Inertia (WCSS)')
axes2[0].legend()
axes2[0].grid(True, alpha=0.3)

# Silhouette Score
axes2[1].plot(list(K_range), silhouette_scores, 'rs-', linewidth=2, markersize=8, color='#f72585')
axes2[1].axvline(x=5, color='blue', linestyle='--', linewidth=2, label='Optimal K=5')
axes2[1].set_title('Silhouette Score')
axes2[1].set_xlabel('Number of Clusters (K)')
axes2[1].set_ylabel('Silhouette Score')
axes2[1].legend()
axes2[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('elbow_silhouette.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ Elbow & Silhouette plot saved as 'elbow_silhouette.png'")

# ─────────────────────────────────────────
# 6. TRAIN K-MEANS WITH K=5
# ─────────────────────────────────────────
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

print(f"\n✅ K-Means model trained with K={optimal_k}")
print(f"\n📊 Cluster Distribution:")
print(df['Cluster'].value_counts().sort_index())

# ─────────────────────────────────────────
# 7. CLUSTER ANALYSIS
# ─────────────────────────────────────────
cluster_summary = df.groupby('Cluster')[['Age', 'AnnualIncome', 'SpendingScore']].mean().round(2)
print(f"\n📈 Cluster Summary (Mean Values):")
print(cluster_summary)

# Cluster labels based on characteristics
cluster_labels = {
    0: 'Careful Spenders',
    1: 'High Value Targets',
    2: 'Careless Spenders',
    3: 'Standard Customers',
    4: 'Sensible Customers'
}

# ─────────────────────────────────────────
# 8. VISUALIZATION OF CLUSTERS
# ─────────────────────────────────────────
cluster_colors = ['#4361ee', '#f72585', '#7209b7', '#4cc9f0', '#f77f00']

fig3, axes3 = plt.subplots(1, 2, figsize=(18, 7))
fig3.suptitle('Prodigy InfoTech — Task 02: K-Means Customer Segments',
              fontsize=14, fontweight='bold', color='#3a0ca3')

# Plot A: Income vs Spending Score Clusters
for i in range(optimal_k):
    cluster_data = df[df['Cluster'] == i]
    axes3[0].scatter(cluster_data['AnnualIncome'], cluster_data['SpendingScore'],
                     c=cluster_colors[i], s=80, alpha=0.8,
                     edgecolors='white', linewidth=0.5,
                     label=f'Cluster {i}: {cluster_labels.get(i, "")}')

# Plot centroids
centroids_original = scaler.inverse_transform(kmeans.cluster_centers_)
axes3[0].scatter(centroids_original[:, 0], centroids_original[:, 1],
                 c='black', s=200, marker='*', zorder=5, label='Centroids')
axes3[0].set_title('Customer Segments\n(Annual Income vs Spending Score)')
axes3[0].set_xlabel('Annual Income (k$)')
axes3[0].set_ylabel('Spending Score (1-100)')
axes3[0].legend(loc='upper left', fontsize=8)
axes3[0].grid(True, alpha=0.2)

# Plot B: Age vs Spending Score
for i in range(optimal_k):
    cluster_data = df[df['Cluster'] == i]
    axes3[1].scatter(cluster_data['Age'], cluster_data['SpendingScore'],
                     c=cluster_colors[i], s=80, alpha=0.8,
                     edgecolors='white', linewidth=0.5,
                     label=f'Cluster {i}')

axes3[1].set_title('Customer Segments\n(Age vs Spending Score)')
axes3[1].set_xlabel('Age')
axes3[1].set_ylabel('Spending Score (1-100)')
axes3[1].legend(fontsize=8)
axes3[1].grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('cluster_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ Cluster results plot saved as 'cluster_results.png'")

# ─────────────────────────────────────────
# 9. CLUSTER PROFILE VISUALIZATION
# ─────────────────────────────────────────
fig4, axes4 = plt.subplots(1, 3, figsize=(18, 6))
fig4.suptitle('Prodigy InfoTech — Task 02: Cluster Profiles',
              fontsize=14, fontweight='bold', color='#3a0ca3')

metrics = ['Age', 'AnnualIncome', 'SpendingScore']
titles  = ['Average Age by Cluster', 'Average Annual Income by Cluster',
           'Average Spending Score by Cluster']
ylabels = ['Age', 'Annual Income (k$)', 'Spending Score']

for idx, (metric, title, ylabel) in enumerate(zip(metrics, titles, ylabels)):
    vals = [df[df['Cluster'] == i][metric].mean() for i in range(optimal_k)]
    bars = axes4[idx].bar(range(optimal_k), vals, color=cluster_colors, 
                          edgecolor='white', linewidth=1.5)
    axes4[idx].set_title(title)
    axes4[idx].set_xlabel('Cluster')
    axes4[idx].set_ylabel(ylabel)
    axes4[idx].set_xticks(range(optimal_k))
    axes4[idx].set_xticklabels([f'C{i}' for i in range(optimal_k)])
    for bar, val in zip(bars, vals):
        axes4[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{val:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig('cluster_profiles.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ Cluster profiles plot saved as 'cluster_profiles.png'")

# ─────────────────────────────────────────
# 10. FINAL SUMMARY
# ─────────────────────────────────────────
print("\n" + "=" * 55)
print("  📊 FINAL CLUSTER SUMMARY")
print("=" * 55)
for i in range(optimal_k):
    cluster_data = df[df['Cluster'] == i]
    print(f"\n  Cluster {i} — {cluster_labels.get(i, 'Group ' + str(i))}")
    print(f"  {'─'*40}")
    print(f"  Customers    : {len(cluster_data)}")
    print(f"  Avg Age      : {cluster_data['Age'].mean():.1f} years")
    print(f"  Avg Income   : ${cluster_data['AnnualIncome'].mean():.1f}k")
    print(f"  Avg Spending : {cluster_data['SpendingScore'].mean():.1f}/100")

print("\n" + "=" * 55)
print("✅ Task 02 Complete! Files saved:")
print("   → customer_segmentation.py")
print("   → eda_analysis.png")
print("   → elbow_silhouette.png")
print("   → cluster_results.png")
print("   → cluster_profiles.png")
print("=" * 55)