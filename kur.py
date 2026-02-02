import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import dendrogram, linkage
from kneed import KneeLocator
import warnings

warnings.filterwarnings('ignore')

# Настройки
plt.rcParams['figure.figsize'] = (12, 8)
sns.set_palette("husl")

# 1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv'
data = pd.read_csv(url)

# Удаляем категориальные признаки
X = data.drop(['Channel', 'Region'], axis=1)
category_names = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicatessen']

print(f"Размерность данных: {X.shape}")
print(f"Количество объектов: {len(X)}")

# Масштабирование
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA для визуализации
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 2. K-MEANS
# Определение оптимального K
k_range = range(2, 11)
inertia = []
silhouette_scores = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = kmeans.fit_predict(X_scaled)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, labels))

optimal_k = k_range[np.argmax(silhouette_scores)]

# График метода локтя и силуэтного анализа
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Выбор оптимального K для K-means', fontsize=16, fontweight='bold')

# Метод локтя
axes[0].plot(k_range, inertia, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('Количество кластеров (K)')
axes[0].set_ylabel('Inertia')
axes[0].set_title('Метод локтя (Elbow Method)')
axes[0].grid(True, alpha=0.3)
axes[0].axvline(x=optimal_k, color='r', linestyle='--', alpha=0.7, label=f'K={optimal_k}')
axes[0].legend()

# Силуэтный анализ
axes[1].plot(k_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
axes[1].set_xlabel('Количество кластеров (K)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Силуэтный анализ')
axes[1].grid(True, alpha=0.3)
axes[1].axvline(x=optimal_k, color='r', linestyle='--', alpha=0.7, label=f'K={optimal_k}')
axes[1].legend()

plt.tight_layout()
plt.show()

# Обучение K-means с оптимальным K
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
kmeans_labels = kmeans.fit_predict(X_scaled)

# Визуализация кластеров K-means
plt.figure(figsize=(10, 7))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='tab10',
                      alpha=0.7, edgecolor='k', s=50)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
plt.title(f'K-Means: {optimal_k} кластеров (Silhouette={silhouette_score(X_scaled, kmeans_labels):.3f})')
plt.colorbar(scatter, label='Кластер')
plt.grid(True, alpha=0.3)
plt.show()

# 3. DBSCAN С АВТОМАТИЧЕСКИМ ПОДБОРОМ ПАРАМЕТРОВ
print("\n" + "=" * 60)
print("DBSCAN КЛАСТЕРИЗАЦИЯ С АВТОМАТИЧЕСКИМ ПОДБОРОМ ПАРАМЕТРОВ")
print("=" * 60)

# Анализ расстояний для подбора eps
min_samples = 5
nn = NearestNeighbors(n_neighbors=min_samples)
nn.fit(X_scaled)
distances, indices = nn.kneighbors(X_scaled)
k_distances = np.sort(distances[:, -1])

# Находим "локоть" на графике (точку максимальной кривизны)
try:
    kneedle = KneeLocator(range(len(k_distances)), k_distances,
                          curve='convex', direction='increasing')
    optimal_eps = k_distances[kneedle.elbow] if kneedle.elbow else k_distances[-1] * 0.5
except:
    optimal_eps = np.percentile(k_distances, 90)  # 90-й перцентиль как fallback


plt.figure(figsize=(10, 6))
plt.plot(k_distances, linewidth=2)
plt.axhline(y=optimal_eps, color='r', linestyle='--',
            label=f'Предлагаемый eps = {optimal_eps:.2f}')
plt.xlabel('Точки, отсортированные по расстоянию')
plt.ylabel(f'Расстояние до {min_samples}-го соседа')
plt.title('Метод локтя для выбора eps в DBSCAN')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"Рекомендуемый eps: {optimal_eps:.2f}")

# Пробуем разные комбинации параметров для поиска оптимальных
param_combinations = [
    {'eps': optimal_eps, 'min_samples': 3},
    {'eps': optimal_eps * 1.2, 'min_samples': 4},
    {'eps': optimal_eps * 0.8, 'min_samples': 5},
    {'eps': optimal_eps * 0.6, 'min_samples': 6},
    {'eps': optimal_eps * 1.5, 'min_samples': 3},
]

best_dbscan = None
best_labels = None
best_score = -1
best_params = None
best_n_clusters = 0

for params in param_combinations:
    dbscan_test = DBSCAN(eps=params['eps'], min_samples=params['min_samples'])
    labels_test = dbscan_test.fit_predict(X_scaled)

    n_clusters_test = len(set(labels_test)) - (1 if -1 in labels_test else 0)
    n_noise_test = list(labels_test).count(-1)
    noise_percent = n_noise_test / len(labels_test) * 100

    # Вычисляем Silhouette только если есть хотя бы 2 кластера и не слишком много шума
    if n_clusters_test >= 2 and noise_percent < 40:  # Ограничиваем максимальный процент шума
        valid_mask = labels_test != -1
        if len(set(labels_test[valid_mask])) >= 2:
            sil_score = silhouette_score(X_scaled[valid_mask], labels_test[valid_mask])
        else:
            sil_score = -1
    else:
        sil_score = -1

    # Исправление: правильное форматирование строки
    sil_str = f"{sil_score:.3f}" if sil_score > -1 else "N/A"
    print(f"eps={params['eps']:.2f}, min_samples={params['min_samples']}: "
          f"{n_clusters_test} кластеров, шум={noise_percent:.1f}%, "
          f"Silhouette={sil_str}")

    # Выбираем лучшую комбинацию по Silhouette
    if sil_score > best_score:
        best_score = sil_score
        best_labels = labels_test
        best_params = params
        best_dbscan = dbscan_test
        best_n_clusters = n_clusters_test

# Если не нашли подходящих параметров, используем дефолтные
if best_score == -1:
    print("\nНе удалось найти параметры для получения 2+ кластеров. Использую дефолтные параметры.")
    best_params = {'eps': 0.5, 'min_samples': 5}
    dbscan = DBSCAN(**best_params)
    best_labels = dbscan.fit_predict(X_scaled)
    best_n_clusters = len(set(best_labels)) - (1 if -1 in best_labels else 0)
else:
    dbscan = best_dbscan

dbscan_labels = best_labels
n_clusters_dbscan = best_n_clusters
n_noise = list(dbscan_labels).count(-1)

print(f"\nЛучшие параметры: eps={best_params['eps']:.2f}, "
      f"min_samples={best_params['min_samples']}")
print(f"DBSCAN: обнаружено {n_clusters_dbscan} кластеров")
print(f"Шумовых точек: {n_noise} ({n_noise / len(dbscan_labels) * 100:.1f}%)")

# Визуализация кластеров DBSCAN
plt.figure(figsize=(10, 7))

# Разделяем шум и кластеры для разной визуализации
noise_mask = dbscan_labels == -1
cluster_mask = dbscan_labels != -1

# Шум
if np.any(noise_mask):
    plt.scatter(X_pca[noise_mask, 0], X_pca[noise_mask, 1],
                color='gray', alpha=0.3, s=20, label='Шум', marker='x')

# Кластеры
if np.any(cluster_mask):
    scatter = plt.scatter(X_pca[cluster_mask, 0], X_pca[cluster_mask, 1],
                          c=dbscan_labels[cluster_mask], cmap='tab10',
                          alpha=0.7, edgecolor='k', s=50)

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')


# Вычисляем Silhouette для DBSCAN если возможно
if n_clusters_dbscan >= 2:
    valid_mask = dbscan_labels != -1
    if len(set(dbscan_labels[valid_mask])) >= 2:
        dbscan_silhouette = silhouette_score(X_scaled[valid_mask], dbscan_labels[valid_mask])
        plt.title(f'DBSCAN: {n_clusters_dbscan} кластеров, '
                  f'шум={n_noise / len(dbscan_labels) * 100:.1f}%\n'
                  f'Silhouette={dbscan_silhouette:.3f}')
    else:
        plt.title(f'DBSCAN: {n_clusters_dbscan} кластеров, '
                  f'шум={n_noise / len(dbscan_labels) * 100:.1f}%')
else:
    plt.title(f'DBSCAN: {n_clusters_dbscan} кластеров, '
              f'шум={n_noise / len(dbscan_labels) * 100:.1f}%')

plt.grid(True, alpha=0.3)
if np.any(cluster_mask) and len(set(dbscan_labels[cluster_mask])) > 1:
    plt.colorbar(scatter, label='Кластер')
plt.legend()
plt.show()

# 4. АГЛОМЕРАТИВНАЯ КЛАСТЕРИЗАЦИЯ
# Построение дендрограммы
sample_size = min(100, len(X_scaled))
indices = np.random.choice(range(len(X_scaled)), sample_size, replace=False)
X_sample = X_scaled[indices]

linkage_matrix = linkage(X_sample, method='ward')

plt.figure(figsize=(10, 6))
dendrogram(linkage_matrix, truncate_mode='lastp', p=20)
plt.xlabel('Объекты')
plt.ylabel('Расстояние')
plt.title('Агломеративная кластеризация: Дендрограмма (метод Ward)')
plt.grid(True, alpha=0.3)
plt.show()

# Применение агломеративной кластеризации с тем же K, что и в K-means
agg_clustering = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
agg_labels = agg_clustering.fit_predict(X_scaled)

# Визуализация результатов
plt.figure(figsize=(10, 7))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=agg_labels, cmap='tab10',
                      alpha=0.7, edgecolor='k', s=50)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
plt.title(
    f'Агломеративная кластеризация: {optimal_k} кластеров (Silhouette={silhouette_score(X_scaled, agg_labels):.3f})')
plt.colorbar(scatter, label='Кластер')
plt.grid(True, alpha=0.3)
plt.show()

# 5. СРАВНИТЕЛЬНЫЙ АНАЛИЗ
print("\n" + "=" * 60)
print("СРАВНИТЕЛЬНЫЙ АНАЛИЗ")
print("=" * 60)

# Вычисление метрик для всех алгоритмов
metrics_data = []

# K-means
metrics_data.append({
    'Алгоритм': 'K-Means',
    'Кластеров': optimal_k,
    'Silhouette': silhouette_score(X_scaled, kmeans_labels),
    'Шум, %': 0.0
})

# DBSCAN
if n_clusters_dbscan >= 2:
    valid_mask = dbscan_labels != -1
    if len(np.unique(dbscan_labels[valid_mask])) > 1:
        sil_score = silhouette_score(X_scaled[valid_mask], dbscan_labels[valid_mask])
    else:
        sil_score = -1
else:
    sil_score = -1

metrics_data.append({
    'Алгоритм': 'DBSCAN',
    'Кластеров': n_clusters_dbscan,
    'Silhouette': sil_score,
    'Шум, %': n_noise / len(dbscan_labels) * 100
})

# Агломеративная
metrics_data.append({
    'Алгоритм': 'Агломеративная',
    'Кластеров': optimal_k,
    'Silhouette': silhouette_score(X_scaled, agg_labels),
    'Шум, %': 0.0
})

# Создание таблицы метрик
metrics_df = pd.DataFrame(metrics_data)

# Вывод таблицы
print("\nСРАВНИТЕЛЬНАЯ ТАБЛИЦА МЕТРИК:")
print("-" * 60)
print(f"{'Алгоритм':15} | {'Кластеров':10} | {'Silhouette':12} | {'Шум, %':8}")
print("-" * 60)
for _, row in metrics_df.iterrows():
    silhouette_str = f"{row['Silhouette']:.3f}" if row['Silhouette'] > -1 else "N/A"
    print(f"{row['Алгоритм']:15} | {row['Кластеров']:10} | {silhouette_str:12} | {row['Шум, %']:7.1f}%")
print("-" * 60)

# Определение лучшего алгоритма (исключая DBSCAN с Silhouette = -1)
valid_metrics = metrics_df[metrics_df['Silhouette'] > -1]
if len(valid_metrics) > 0:
    best_algo_idx = valid_metrics['Silhouette'].idxmax()
    best_algo = metrics_df.loc[best_algo_idx, 'Алгоритм']
    best_sil = metrics_df.loc[best_algo_idx, 'Silhouette']
else:
    best_algo = "Не определен"
    best_sil = -1

# Финальный сравнительный график
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('СРАВНИТЕЛЬНЫЙ АНАЛИЗ МЕТОДОВ КЛАСТЕРИЗАЦИИ', fontsize=16, fontweight='bold')


# K-means
scatter1 = axes[0, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels,
                              cmap='tab10', alpha=0.7, edgecolor='k', s=40)
axes[0, 0].set_xlabel('PC1')
axes[0, 0].set_ylabel('PC2')
axes[0, 0].set_title(f'K-Means (K={optimal_k})')
axes[0, 0].grid(True, alpha=0.3)

# DBSCAN
for label in sorted(set(dbscan_labels)):
    if label == -1:
        color = 'gray'
        alpha = 0.3
        s = 15
    else:
        color = plt.cm.tab10(label % 10)
        alpha = 0.7
        s = 40

    mask = dbscan_labels == label
    axes[0, 1].scatter(X_pca[mask, 0], X_pca[mask, 1],
                       color=color, alpha=alpha, s=s, edgecolor='k' if label != -1 else None)
axes[0, 1].set_xlabel('PC1')
axes[0, 1].set_ylabel('PC2')
axes[0, 1].set_title(f'DBSCAN (K={n_clusters_dbscan})')
axes[0, 1].grid(True, alpha=0.3)

# Агломеративная
scatter3 = axes[1, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=agg_labels,
                              cmap='tab10', alpha=0.7, edgecolor='k', s=40)
axes[1, 0].set_xlabel('PC1')
axes[1, 0].set_ylabel('PC2')
axes[1, 0].set_title(f'Агломеративная (K={optimal_k})')
axes[1, 0].grid(True, alpha=0.3)

# Таблица метрик
axes[1, 1].axis('off')

# Создаем таблицу
table_data = []
for _, row in metrics_df.iterrows():
    silhouette_str = f"{row['Silhouette']:.3f}" if row['Silhouette'] > -1 else "N/A"
    table_data.append([
        row['Алгоритм'],
        str(row['Кластеров']),
        silhouette_str,
        f"{row['Шум, %']:.1f}%"
    ])

# Добавляем таблицу
table = axes[1, 1].table(cellText=table_data,
                         colLabels=['Алгоритм', 'Кластеры', 'Silhouette', 'Шум, %'],
                         cellLoc='center',
                         loc='center',
                         bbox=[0.1, 0.3, 0.8, 0.6])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.0)

# Выделяем лучший алгоритм
if best_algo != "Не определен" and best_sil > -1:
    for i, row in enumerate(table_data):
        if row[0] == best_algo:
            best_cell = table[(i + 1), 2]
            best_cell.set_facecolor('#90EE90')  # Светло-зеленый
            break

# Добавляем итоговый вывод
result_text = f'Лучший алгоритм: {best_algo} (Silhouette = {best_sil:.3f})' if best_sil > -1 else 'Невозможно определить лучший алгоритм'
axes[1, 1].text(0.5, 0.05, result_text,
                ha='center', va='center', fontsize=12, fontweight='bold',
                transform=axes[1, 1].transAxes)

plt.tight_layout()
plt.show()

# 6. ВЫВОД
# DBSCAN Silhouette для вывода
dbscan_sil_str = "N/A"
if n_clusters_dbscan >= 2:
    valid_mask = dbscan_labels != -1
    if len(set(dbscan_labels[valid_mask])) > 1:
        dbscan_sil = silhouette_score(X_scaled[valid_mask], dbscan_labels[valid_mask])
        dbscan_sil_str = f"{dbscan_sil:.3f}"

# Сохранение результатов
data_with_clusters = data.copy()
data_with_clusters['KMeans_Cluster'] = kmeans_labels
data_with_clusters['DBSCAN_Cluster'] = dbscan_labels
data_with_clusters['Agglomerative_Cluster'] = agg_labels
