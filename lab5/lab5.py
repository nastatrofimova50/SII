import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# Загрузка данных
X = pd.read_csv("wiki4HE.csv", delimiter=";", header=None)
X = X.apply(pd.to_numeric, errors='coerce')

# Очистка данных
X_cleaned = X.dropna()

# Масштабирование данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cleaned)

# Выбор подвыборки
sample_data = X.sample(frac=0.5, random_state=42)

# Импутация пропущенных значений
imputer = SimpleImputer(strategy='mean')
X_sample_imputed = imputer.fit_transform(sample_data)
X_sample_scaled = scaler.transform(X_sample_imputed)

# 1: K-means кластеризация
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X_sample_scaled)

# 2: DBSCAN кластеризация
dbscan = DBSCAN(eps=7, min_samples=10)
dbscan_labels = dbscan.fit_predict(X_sample_scaled)

# 3: Agglomerative Clustering
agglo = AgglomerativeClustering(n_clusters=3, linkage="ward")
agglo_labels = agglo.fit_predict(X_sample_scaled)

# Визуализация с помощью PCA
pca_vis = PCA(n_components=2)
X_pca_vis = pca_vis.fit_transform(X_sample_scaled)

plt.figure(figsize=(18, 6))

# K-means визуализация
plt.subplot(1, 3, 1)
plt.scatter(X_pca_vis[:, 0], X_pca_vis[:, 1], c=kmeans_labels, cmap="viridis", marker="o")
plt.title("K-means Clustering")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")

# DBSCAN визуализация
plt.subplot(1, 3, 2)
plt.scatter(X_pca_vis[:, 0], X_pca_vis[:, 1], c=dbscan_labels, cmap="viridis", marker="o")
plt.title("DBSCAN Clustering")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")

# Agglomerative Clustering визуализация
plt.subplot(1, 3, 3)
plt.scatter(X_pca_vis[:, 0], X_pca_vis[:, 1], c=agglo_labels, cmap="viridis", marker="o")
plt.title("Agglomerative Clustering")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")

plt.tight_layout()
plt.show()


def evaluate_clustering(X, labels):
    """
    Оценка качества кластеризации
    """
    if len(np.unique(labels)) > 1:
        silhouette = silhouette_score(X, labels)
        calinski = calinski_harabasz_score(X, labels)
        return silhouette, calinski
    else:
        return None, None


# Оценка K-means для разного количества кластеров
kmeans_results = []
for n_clusters in range(2, 6):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_labels = kmeans.fit_predict(X_sample_scaled)
    silhouette, calinski = evaluate_clustering(X_sample_scaled, kmeans_labels)
    kmeans_results.append((n_clusters, silhouette, calinski))

# Оценка DBSCAN для разных параметров
dbscan_results = []
for eps in np.linspace(6, 9, 5):
    for min_samples in range(5, 21, 5):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan_labels = dbscan.fit_predict(X_sample_scaled)
        silhouette, calinski = evaluate_clustering(X_sample_scaled, dbscan_labels)
        if silhouette is not None:  # Добавляем только валидные результаты
            dbscan_results.append((eps, min_samples, silhouette, calinski))

# Оценка Agglomerative Clustering для разного количества кластеров
agglo_results = []
for n_clusters in range(2, 6):
    agglo = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    agglo_labels = agglo.fit_predict(X_sample_scaled)
    silhouette, calinski = evaluate_clustering(X_sample_scaled, agglo_labels)
    agglo_results.append((n_clusters, silhouette, calinski))


def best_result(results, metric_index=1):
    """
    Получение лучшего результата по указанной метрике.
    metric_index - индекс метрики (1 - silhouette, 2 - calinski)
    """
    # Фильтруем результаты с валидными метриками
    valid_results = [r for r in results if r[metric_index] is not None]
    if valid_results:
        return max(valid_results, key=lambda x: x[metric_index])
    else:
        return None


# Вывод лучших результатов
print("=" * 60)

# KMeans результаты
best_kmeans = best_result(kmeans_results, 1)
if best_kmeans:
    print("\nЛучший результат для KMeans:")
    print(f"Кластеры: {best_kmeans[0]}, "
          f"Silhouette: {best_kmeans[1]:.4f}, "
          f"Calinski-Harabasz: {best_kmeans[2]:.4f}")
else:
    print("\nНе удалось получить валидные результаты для KMeans")

# DBSCAN результаты
best_dbscan = best_result(dbscan_results, 2) if dbscan_results else None
if best_dbscan:
    print("\nЛучший результат для DBSCAN:")
    print(f"eps: {best_dbscan[0]:.2f}, min_samples: {best_dbscan[1]}, "
          f"Silhouette: {best_dbscan[2]:.4f}, "
          f"Calinski-Harabasz: {best_dbscan[3]:.4f}")
else:
    print("\nНе удалось получить валидные результаты для DBSCAN")

# Agglomerative Clustering результаты
best_agglo = best_result(agglo_results, 1)
if best_agglo:
    print("\nЛучший результат для Agglomerative Clustering:")
    print(f"Кластеры: {best_agglo[0]}, "
          f"Silhouette: {best_agglo[1]:.4f}, "
          f"Calinski-Harabasz: {best_agglo[2]:.4f}")
else:
    print("\nНе удалось получить валидные результаты для Agglomerative Clustering")

print("\n" + "=" * 60)

# Дополнительная визуализация лучших моделей
if best_kmeans:
    # Лучшая K-means модель
    best_kmeans_model = KMeans(n_clusters=best_kmeans[0], random_state=42)
    best_kmeans_labels = best_kmeans_model.fit_predict(X_sample_scaled)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.scatter(X_pca_vis[:, 0], X_pca_vis[:, 1], c=best_kmeans_labels, cmap="viridis", marker="o")
    plt.title(f"Лучший K-means (k={best_kmeans[0]})")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")

    if best_agglo:
        plt.subplot(1, 3, 2)
        best_agglo_model = AgglomerativeClustering(n_clusters=best_agglo[0], linkage='ward')
        best_agglo_labels = best_agglo_model.fit_predict(X_sample_scaled)
        plt.scatter(X_pca_vis[:, 0], X_pca_vis[:, 1], c=best_agglo_labels, cmap="plasma", marker="o")
        plt.title(f"Лучший Agglomerative (k={best_agglo[0]})")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")

    plt.tight_layout()
    plt.show()