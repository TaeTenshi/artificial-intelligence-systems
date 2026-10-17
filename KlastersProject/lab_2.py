import math
import random
import time
import numpy as np
from sklearn.cluster import KMeans


def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def distance_matrix(lat, lon): # матрица попарных расстояний
    n = len(lat)
    mat = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                mat[i][j] = distance((lat[i], lon[i]), (lat[j], lon[j]))
    return mat


def algorithmic_clustering(lat, lon, k):
    n = len(lat)
    if k > n:
        k = n
    dist = distance_matrix(lat, lon)
    clusters = [[i] for i in range(n)]

    def cluster_dist_max(c1, c2):
        max_d = 0.0
        for i in c1:
            for j in c2:
                d = dist[i][j]
                if d > max_d:
                    max_d = d
        return max_d

    while len(clusters) > k:
        min_d = float('inf') #+бесконечность
        to_merge = None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                d = cluster_dist_max(clusters[i], clusters[j])
                if d < min_d: # если расстояние между кластерами меньше текущего минимума (+бесконечность)
                    min_d = d
                    to_merge = (i, j)
        i1, i2 = to_merge
        clusters[i1].extend(clusters[i2])
        clusters.pop(i2) # pop удаление кластера
    return clusters


def kmeans_scratch(lat, lon, k, max_iter=100):
    n = len(lat)
    if k > n:
        k = n
    points = list(zip(lat, lon))
    min_lat, max_lat = min(lat), max(lat) # границы по широте
    min_lon, max_lon = min(lon), max(lon) # границы по долготе
    centroids = [[random.uniform(min_lat, max_lat), random.uniform(min_lon, max_lon)] for _ in range(k)]

    for _ in range(max_iter): # распределение точек по ближайшим центроидам
        clusters = [[] for _ in range(k)]
        for i in range(n):
            dists = [distance(points[i], c) for c in centroids] # расстояния до каждого из центроидов
            best = dists.index(min(dists))
            clusters[best].append(i)

        new_centroids = [] # новый более подходящий центроид
        for c_idx in range(k):
            if clusters[c_idx]:
                avg_lat = sum(points[i][0] for i in clusters[c_idx]) / len(clusters[c_idx]) # среднее арифметическое широты всех городов кластера
                avg_lon = sum(points[i][1] for i in clusters[c_idx]) / len(clusters[c_idx]) # среднее арифметическое долготы всех городов кластера
                new_centroids.append([avg_lat, avg_lon])

        if new_centroids == centroids:
            break
        centroids = new_centroids # назначаются новые центроиды

    return clusters, centroids


def kmeans_sklearn(lat, lon, k):
    n = len(lat)
    if k > n:
        k = n
    coords = np.column_stack((lat, lon))
    kmeans = KMeans(n_clusters=k, n_init=10) # k кластеров, n_init=10 10 запусков с разными центрами, создание объекта с настройками
    labels = kmeans.fit_predict(coords) # подбирает центроид и даёт индекс подходящего кластера для точки
    clusters = [[] for _ in range(k)] # возвращает список из списков с координатами
    for i, lbl in enumerate(labels): # enumerate получает элемент и его индекс
        clusters[lbl].append(i)
        #print(clusters)
    return clusters, kmeans.cluster_centers_


def compute_inertia(lat, lon, clusters):
    total = 0.0
    for cluster in clusters:
        if not cluster:
            continue
        centroid_lat = sum(lat[i] for i in cluster) / len(cluster) # средняя широта
        centroid_lon = sum(lon[i] for i in cluster) / len(cluster) # средняя долгота
        for i in cluster:
            total += distance((lat[i], lon[i]), (centroid_lat, centroid_lon)) ** 2
    return total


def find_optimal_k_elbow(lat, lon):
    # находит оптимальное K методом локтя через sklearn
    n = len(lat)
    max_k = n - 1

    inertias = []
    for k in range(1, max_k + 1):
        clusters, centers = kmeans_sklearn(lat, lon, k) # возвращает кортеж который сразу делится
        inertias.append(compute_inertia(lat, lon, clusters))  # считаем инерцию

    if len(inertias) < 3: # если меньше трёх точек инерции
        return 1, inertias

    optimal_k = 1
    max_angle = 0 # максимальный угол

    for i in range(1, len(inertias) - 1): # перебираем внутренние точки
        v1 = (1, inertias[i - 1] - inertias[i]) # вектор «до»
        v2 = (1, inertias[i] - inertias[i + 1]) # вектор «после»

        dot = v1[0] * v2[0] + v1[1] * v2[1]
        norm1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
        norm2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)

        cos_angle = dot / (norm1 * norm2)
        angle = math.acos(max(-1, min(1, cos_angle)))

        if angle > max_angle: # ищется максимальный угол
            max_angle = angle
            optimal_k = i + 1

    return optimal_k, inertias


def print_clusters(clusters, names):
    for idx, cluster in enumerate(clusters):          # выводим кластеры
        city_names = [names[i] for i in cluster]         # берём названия городов по индексам
        print(f"  Кластер {idx}: {city_names}")


def generate_test_cases():
    tests = []

    # Тест: 3 облака (города Поволжья и центра)
    gor2 = ["Ульяновск", "Самара", "Казань", "Москва", "Питер", "Нижний Новгород"]
    lat2 = [0, 1, 10, 11, 20, 21]
    lon2 = [0, 1, 20, 21, 0, 1]
    tests.append({"name": "3 облака", "cities": gor2, "lat": lat2, "lon": lon2, "K": 3})

    # Тест: Цепочка (города на одной линии)
    gor3 = ["Ульяновск", "Самара", "Казань", "Москва", "Питер", "Нижний Новгород", "Тверь", "Владимир"]
    lat3 = [i * 5 for i in range(8)]
    lon3 = [i * 5 for i in range(8)]
    tests.append({"name": "Цепочка (8 городов)", "cities": gor3, "lat": lat3, "lon": lon3, "K": 3})

    # Тест: Квадрат (4 города)
    gor_sq = ["Ульяновск", "Самара", "Казань", "Москва"]
    lat_sq = [0, 0, 10, 10]
    lon_sq = [0, 10, 10, 0]
    tests.append({"name": "Квадрат (4 города)", "cities": gor_sq, "lat": lat_sq, "lon": lon_sq, "K": 2})

    return tests


if __name__ == "__main__":
    test_cases = generate_test_cases()

    for test in test_cases:
        print("=" * 60)
        print(f"Тест: {test['name']}")
        lat = test["lat"]
        lon = test["lon"]
        names = test["cities"]
        K = test["K"]
        print(f"  Города: {names}")
        print(f"  Координаты: lat={lat}, lon={lon}")
        print(f"  Заданное K = {K}")

        # Поиск оптимального K методом локтя
        opt_k, inertias = find_optimal_k_elbow(lat, lon)
        print()
        print("Поиск оптимального K (метод локтя):")
        for k, inertia in enumerate(inertias, 1):
            marker = " <-- оптимальное" if k == opt_k else ""
            print(f"  K={k}: инерция = {inertia:.2f}{marker}")
        print(f"  Оптимальное K = {opt_k}")
        print()
        print(f"Результаты кластеризации с K={opt_k}:")
        print()

        start = time.time()
        clusters_hier = algorithmic_clustering(lat, lon, opt_k)
        elapsed_hier = time.time() - start
        inertia_hier = compute_inertia(lat, lon, clusters_hier)
        print(f"Иерархический метод:")
        print(f"  Время: {elapsed_hier:.6f} с, инерция: {inertia_hier:.3f}")
        print_clusters(clusters_hier, names)
        print()

        start = time.time()
        clusters_scratch, centroids_scratch = kmeans_scratch(lat, lon, opt_k)
        elapsed_scratch = time.time() - start
        inertia_scratch = compute_inertia(lat, lon, clusters_scratch)
        print(f"K-means (с нуля):")
        print(f"  Время: {elapsed_scratch:.6f} с, инерция: {inertia_scratch:.3f}")
        print_clusters(clusters_scratch, names)
        print()

        start = time.time()
        clusters_sklearn, centroids_sklearn = kmeans_sklearn(lat, lon, opt_k)
        elapsed_sklearn = time.time() - start
        inertia_sklearn = compute_inertia(lat, lon, clusters_sklearn)
        print(f"K-means (scikit-learn):")
        print(f"  Время: {elapsed_sklearn:.6f} с, инерция: {inertia_sklearn:.3f}")
        print_clusters(clusters_sklearn, names)
        print()