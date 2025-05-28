import numpy as np
from sklearn.cluster import KMeans
from shapely.geometry import Polygon, Point
from scipy.spatial import distance
import matplotlib.pyplot as plt
import random

def generate_grid_points(area_points, step=10.0):
    polygon = Polygon(area_points)
    min_x, min_y, max_x, max_y = polygon.bounds

    grid_points = []
    x = min_x
    while x <= max_x:
        y = min_y
        while y <= max_y:
            p = Point(x, y)
            if polygon.contains(p):
                grid_points.append((x, y))
            y += step
        x += step
    return grid_points

def nearest_neighbor_path(points, start):
    points = points.copy()
    path = [start]
    if start in points:
        points.remove(start)

    current = start
    while points:
        distances = [distance.euclidean(current, p) for p in points]
        nearest_index = np.argmin(distances)
        current = points.pop(nearest_index)
        path.append(current)
    return path

def cluster_points(points, n_clusters=3):
    """ Suskirsto taškus į n klasterių naudojant KMeans """
    coords = np.array(points)
    kmeans = KMeans(n_clusters=n_clusters, n_init='auto')
    labels = kmeans.fit_predict(coords)

    clusters = [[] for _ in range(n_clusters)]
    for point, label in zip(points, labels):
        clusters[label].append(point)
    return clusters

def split_points(points, n=3):
    """ Atsitiktinai padalina taškus į n grupių """
    shuffled = points.copy()
    random.shuffle(shuffled)
    return [shuffled[i::n] for i in range(n)]

def assign_points_to_drones(points, drone_starts):
    """ Priskiria kiekvieną tašką artimiausiam dronui pagal starto tašką """
    assignments = [[] for _ in range(len(drone_starts))]

    for p in points:
        distances = [distance.euclidean(p, s) for s in drone_starts]
        nearest_drone = np.argmin(distances)
        assignments[nearest_drone].append(p)

    return assignments

def plot_multi_paths(paths, area_points, starts):
    plt.figure(figsize=(10, 10))
    
    # Poligonas
    poly = Polygon(area_points)
    x_poly, y_poly = poly.exterior.xy
    plt.plot(x_poly, y_poly, 'k--', label='Teritorijos riba')

    colors = ['blue', 'green', 'orange']

    for i, (path, start) in enumerate(zip(paths, starts)):
        xs, ys = zip(*path)
        color = colors[i % len(colors)]

        # Maršrutas
        plt.plot(xs, ys, color=color, linestyle='-', marker='o',
                 linewidth=1, markersize=3, label=f'Dronas {i+1}')

        # Pradžios taškas (tokios pačios spalvos kaip maršrutas)
        plt.scatter([start[0]], [start[1]], c=color, s=100, edgecolors='black', label=f'Drono {i+1} startas')

        # Linija nuo drono starto iki pirmo taško, jei nesutampa
        if start != path[0]:
            plt.plot([start[0], path[0][0]], [start[1], path[0][1]],
                     color=color, linestyle='dotted', linewidth=1)

    plt.gca().set_aspect('equal')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.title("Dronų maršrutai (Artimiausio kaimyno algoritmas)")
    plt.show()

# ---- Pradžia ----

# Pvz. teritorija
#area = [(0.0, 0.0), (20.0, 60.0), (40.0, 80.0), (90.0, 70.0), (100.0, 20.0), (70.0, 10.0), (40.0, 20.0), (20.0, 10.0)]
#area = [(0.0, 0.0), (0.0, 100.0), (100.0, 50.0), (0.0, 0.0)] #trikampis
area = [(0.0, 0.0), (0.0, 100.0), (100.0, 100.0), (100.0, 0.0)] #kvadratas
#area = [(0.0, 0.0), (40.0, 100.0), (140.0, 100.0), (180.0, 0.0) ] #trapecija
#area = [(0.0, 0.0), (33.0, 66.0), (99.0, 99.0), (165.0, 66.0), (198.0, 0.0), (165.0, -66.0), (99.0, -99.0), (33.0, -66.0) ] #apskritimas

# Tinklo taškai
grid = generate_grid_points(area, step=10)

# Nustatome pradines dronų pozicijas
start_points = [(0, 0), (0, 5), (0, 10)]

# Klasterizuojam taškus
splits = cluster_points(grid, n_clusters=3)

# Surandam artimiausią tašką kiekvienam drono startui savo klasteryje
paths = []
for cluster, start in zip(splits, start_points):
    # Iš klasterio surandam artimiausią tašką starto taškui
    nearest = min(cluster, key=lambda p: distance.euclidean(p, start))
    path = nearest_neighbor_path(cluster, start=nearest)
    paths.append(path)

# Braižom
plot_multi_paths(paths, area, start_points)
