import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ──────────────────────────────────────────
# 1. Load Data
# ──────────────────────────────────────────
df = pd.read_csv('Dataset/ds_salaries.csv')
print("Dataset Shape:", df.shape)
print(df.head(3))

# ──────────────────────────────────────────
# 2. Encode Categorical Columns
# ──────────────────────────────────────────
le = LabelEncoder()
df_encoded = df.copy()

cat_cols = ['experience_level', 'employment_type', 'company_size',
            'company_location', 'employee_residence']
for col in cat_cols:
    df_encoded[col] = le.fit_transform(df_encoded[col])

# ──────────────────────────────────────────
# 3. Select Features & Scale
# ──────────────────────────────────────────
features = ['work_year', 'salary_in_usd', 'remote_ratio',
            'experience_level', 'employment_type', 'company_size']
X = df_encoded[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nFeature matrix shape:", X_scaled.shape)

# ──────────────────────────────────────────
# 4. Elbow Method
# ──────────────────────────────────────────
inertia = []
K_range = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

# ──────────────────────────────────────────
# 5. Silhouette Score for Each K
# ──────────────────────────────────────────
silhouette_scores = []

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)
    print(f"k={k}, Silhouette Score: {score:.4f}")

# Best k
best_k_idx = np.argmax(silhouette_scores)
best_k = list(K_range)[best_k_idx]
best_score = silhouette_scores[best_k_idx]

print(f"\nOptimal k  : {best_k}")
print(f"Best Silhouette Score: {best_score:.4f}")

# ──────────────────────────────────────────
# 6. Plot Elbow & Silhouette Score
# ──────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('K-Means Clustering - DS Salaries Dataset',
             fontsize=16, fontweight='bold', y=1.01)

# --- Elbow Plot ---
axes[0].plot(list(K_range), inertia, marker='o', color='steelblue',
             linewidth=2, markersize=8)
axes[0].set_title('Elbow Method\n(Optimal K Selection)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Number of Clusters (k)', fontsize=11)
axes[0].set_ylabel('Inertia (WCSS)', fontsize=11)
axes[0].grid(True, alpha=0.3)
axes[0].set_xticks(list(K_range))

# --- Silhouette Score Bar Chart ---
colors = ['red' if k == best_k else 'steelblue' for k in K_range]
axes[1].bar(list(K_range), silhouette_scores, color=colors,
            edgecolor='black', linewidth=0.5)
axes[1].set_title('Silhouette Score per K\n(Higher = Better Clustering)',
                  fontsize=13, fontweight='bold')
axes[1].set_xlabel('Number of Clusters (k)', fontsize=11)
axes[1].set_ylabel('Silhouette Score', fontsize=11)
axes[1].set_xticks(list(K_range))
axes[1].grid(True, alpha=0.3, axis='y')

# Annotate bar values
for k, sc in zip(K_range, silhouette_scores):
    axes[1].text(k, sc + 0.002, f'{sc:.3f}', ha='center',
                 va='bottom', fontsize=9, fontweight='bold')

axes[1].text(best_k, silhouette_scores[best_k_idx] + 0.015,
             f'Best k={best_k}', ha='center', fontsize=10,
             color='red', fontweight='bold')

plt.tight_layout()
plt.savefig('kmeans_clustering.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved as 'kmeans_clustering.png'")

# ──────────────────────────────────────────
# 7. Final Clustering with Best K
# ──────────────────────────────────────────
km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df['Cluster'] = km_final.fit_predict(X_scaled)

print("\n=== Cluster Distribution ===")
print(df['Cluster'].value_counts().sort_index())

print("\n=== Cluster Summary (Mean Salary per Cluster) ===")
print(df.groupby('Cluster')['salary_in_usd'].mean().sort_values(ascending=False))