"""
Example: Enhanced Algorithm Tournament
Shows how to easily expand the current hardcoded tournament to test many more algorithms.
"""

# Enhanced Tournament Code (could replace the current hardcoded version)
enhanced_tournament_code = """
# ENHANCED ALGORITHM TOURNAMENT - Multiple Algorithm Families

from sklearn.cluster import (
    KMeans, GaussianMixture, AgglomerativeClustering,
    DBSCAN, SpectralClustering, Birch
)
from sklearn.metrics import silhouette_score, calinski_harabasz_index, davies_bouldin_score
import numpy as np

tournament_results = []

print("\\n=== ENHANCED ALGORITHM TOURNAMENT ===")

# 1. K-MEANS FAMILY
print("\\n1. Testing K-Means variants...")
for k in range(2, 8):
    # Standard K-Means
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    # Multiple metrics
    sil_score = silhouette_score(X_scaled, labels)
    ch_score = calinski_harabasz_index(X_scaled, labels)
    db_score = davies_bouldin_score(X_scaled, labels)

    tournament_results.append({
        'algorithm': 'KMeans',
        'variant': 'standard',
        'k': k,
        'silhouette_score': sil_score,
        'calinski_harabasz_score': ch_score,
        'davies_bouldin_score': db_score,  # Lower is better
        'labels': labels,
        'model': kmeans
    })

    print(f"  K-Means k={k}: Sil={sil_score:.3f}, CH={ch_score:.1f}, DB={db_score:.3f}")

# 2. GAUSSIAN MIXTURE MODELS
print("\\n2. Testing Gaussian Mixture Models...")
for n in range(2, 8):
    for covariance_type in ['full', 'tied', 'diag', 'spherical']:
        try:
            gmm = GaussianMixture(n_components=n, covariance_type=covariance_type, random_state=42)
            labels = gmm.fit_predict(X_scaled)

            sil_score = silhouette_score(X_scaled, labels)
            bic_score = gmm.bic(X_scaled)
            aic_score = gmm.aic(X_scaled)

            tournament_results.append({
                'algorithm': 'GMM',
                'variant': covariance_type,
                'k': n,
                'silhouette_score': sil_score,
                'bic_score': bic_score,
                'aic_score': aic_score,
                'labels': labels,
                'model': gmm
            })

            print(f"  GMM n={n} ({covariance_type}): Sil={sil_score:.3f}, BIC={bic_score:.1f}")
        except Exception as e:
            print(f"  GMM n={n} ({covariance_type}): ERROR - {e}")

# 3. HIERARCHICAL CLUSTERING
print("\\n3. Testing Hierarchical Clustering...")
for k in range(2, 8):
    for linkage in ['ward', 'complete', 'average']:
        try:
            agg = AgglomerativeClustering(n_clusters=k, linkage=linkage)
            labels = agg.fit_predict(X_scaled)

            sil_score = silhouette_score(X_scaled, labels)
            ch_score = calinski_harabasz_index(X_scaled, labels)

            tournament_results.append({
                'algorithm': 'Hierarchical',
                'variant': linkage,
                'k': k,
                'silhouette_score': sil_score,
                'calinski_harabasz_score': ch_score,
                'labels': labels,
                'model': agg
            })

            print(f"  Hierarchical k={k} ({linkage}): Sil={sil_score:.3f}, CH={ch_score:.1f}")
        except Exception as e:
            print(f"  Hierarchical k={k} ({linkage}): ERROR - {e}")

# 4. DENSITY-BASED CLUSTERING (DBSCAN)
print("\\n4. Testing DBSCAN...")
eps_values = [0.1, 0.3, 0.5, 0.7, 1.0]
min_samples_values = [3, 5, 10]

for eps in eps_values:
    for min_samples in min_samples_values:
        try:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X_scaled)

            # Skip if only one cluster (all noise) or all one cluster
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            if n_clusters < 2:
                continue

            sil_score = silhouette_score(X_scaled, labels)

            tournament_results.append({
                'algorithm': 'DBSCAN',
                'variant': f'eps={eps}_min={min_samples}',
                'k': n_clusters,
                'silhouette_score': sil_score,
                'eps': eps,
                'min_samples': min_samples,
                'labels': labels,
                'model': dbscan
            })

            print(f"  DBSCAN eps={eps} min={min_samples}: {n_clusters} clusters, Sil={sil_score:.3f}")
        except Exception as e:
            print(f"  DBSCAN eps={eps} min={min_samples}: ERROR - {e}")

# 5. SPECTRAL CLUSTERING
print("\\n5. Testing Spectral Clustering...")
for k in range(2, 6):  # Spectral can be slower, so fewer k values
    try:
        spectral = SpectralClustering(n_clusters=k, random_state=42, affinity='rbf')
        labels = spectral.fit_predict(X_scaled)

        sil_score = silhouette_score(X_scaled, labels)

        tournament_results.append({
            'algorithm': 'Spectral',
            'variant': 'rbf',
            'k': k,
            'silhouette_score': sil_score,
            'labels': labels,
            'model': spectral
        })

        print(f"  Spectral k={k}: Sil={sil_score:.3f}")
    except Exception as e:
        print(f"  Spectral k={k}: ERROR - {e}")

# 6. BIRCH CLUSTERING
print("\\n6. Testing BIRCH...")
for k in range(2, 8):
    for threshold in [0.1, 0.3, 0.5]:
        try:
            birch = Birch(n_clusters=k, threshold=threshold)
            labels = birch.fit_predict(X_scaled)

            sil_score = silhouette_score(X_scaled, labels)

            tournament_results.append({
                'algorithm': 'BIRCH',
                'variant': f'thresh={threshold}',
                'k': k,
                'silhouette_score': sil_score,
                'labels': labels,
                'model': birch
            })

            print(f"  BIRCH k={k} thresh={threshold}: Sil={sil_score:.3f}")
        except Exception as e:
            print(f"  BIRCH k={k} thresh={threshold}: ERROR - {e}")

# TOURNAMENT ANALYSIS
print(f"\\n=== TOURNAMENT RESULTS ===")
print(f"Total algorithms tested: {len(tournament_results)}")

if tournament_results:
    # Convert to DataFrame for analysis
    import pandas as pd
    results_df = pd.DataFrame(tournament_results)

    # Find best overall performer (highest silhouette score)
    best_idx = results_df['silhouette_score'].idxmax()
    best_result = results_df.iloc[best_idx]

    print(f"\\n=== TOURNAMENT WINNER ===")
    print(f"Algorithm: {best_result['algorithm']}")
    print(f"Variant: {best_result.get('variant', 'standard')}")
    print(f"Clusters: {best_result['k']}")
    print(f"Silhouette Score: {best_result['silhouette_score']:.4f}")

    # Show top 5 performers
    print(f"\\n=== TOP 5 PERFORMERS ===")
    top_5 = results_df.nlargest(5, 'silhouette_score')
    for idx, row in top_5.iterrows():
        variant_str = f" ({row.get('variant', 'std')})" if 'variant' in row else ""
        print(f"{row['algorithm']}{variant_str} k={row['k']}: {row['silhouette_score']:.4f}")

    # Store winner info for downstream use
    winner_algorithm = best_result['algorithm']
    winner_variant = best_result.get('variant', 'standard')
    winner_k = best_result['k']
    winner_score = best_result['silhouette_score']
    winner_labels = best_result['labels']
    winner_model = best_result['model']
else:
    print("ERROR: No algorithms succeeded in tournament")
"""

print("Enhanced Tournament Features:")
print("✓ 6 Algorithm Families: K-Means, GMM, Hierarchical, DBSCAN, Spectral, BIRCH")
print("✓ Multiple Variants: Different linkage methods, covariance types, parameters")
print("✓ Multiple Metrics: Silhouette, Calinski-Harabasz, Davies-Bouldin, BIC, AIC")
print("✓ Automatic Parameter Tuning: Tests different eps, min_samples, thresholds")
print("✓ Robust Error Handling: Continues tournament even if some algorithms fail")
print("✓ Top-N Analysis: Shows best performers, not just winner")