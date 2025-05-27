import numpy as np
import asyncio
from mavsdk import System
from mavsdk.offboard import OffboardError, PositionNedYaw
from scipy.spatial import distance
from scipy.spatial import distance_matrix 
from sklearn.cluster import KMeans
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

# Sugeneruojame taškus teritorijoje pagal grid'ą
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

# K-means algoritmas, suskirstyti taškus į n klasterius
def kmeans_clustering(points, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(points)
    return kmeans.labels_, kmeans.cluster_centers_

# Pritaikyti maršrutą kiekvienam dronui pagal klasterius
def assign_drone_routes(points, labels, n_drones):
    drone_routes = [[] for _ in range(n_drones)]
    
    for idx, label in enumerate(labels):
        drone_routes[label].append(points[idx])

    return drone_routes

# tvaizduoja i nuotrauka kaip sugrupuoti taskai
def visualize_clusters(points, labels, centers):
    plt.figure(figsize=(10, 6))
    plt.scatter(*zip(*points), c=labels, cmap='viridis', marker='o')
    plt.scatter(*zip(*centers), c='red', marker='x', label='Centroidai')
    plt.title("Dronų maršrutai pagal klasterius")
    plt.xlabel("X koordinačių ašis")
    plt.ylabel("Y koordinačių ašis")
    plt.legend()
     # Išsaugoti vaizdą kaip failą
    plt.savefig('clusters_output.png')

# Pritaikyti maršrutą pagal artimiausią kaimyną
def nearest_neighbor(points):
    route = []
    unvisited_points = points.copy()
    current_point = unvisited_points.pop(0)  # Pasirenkame pradžios tašką
    route.append(current_point)
    
    while unvisited_points:
        distances = [distance.euclidean(current_point, point) for point in unvisited_points]
        nearest_idx = np.argmin(distances)
        current_point = unvisited_points.pop(nearest_idx)
        route.append(current_point)
        
    return route

# Skrenda pagal maršrutą
async def fly_area(drone, route, offset):
    print(f"\nDronui {drone}: Maršrutas (NED koordinatės su offset):")
    
    for point in route:
        # Pritaikome offset'ą ir nustatome NED poziciją
        shifted = PositionNedYaw(
            point[0] + offset[0],  # North
            point[1] + offset[1],  # East
            -15.0,                   # Down (Z)
            0.0                    # Yaw
        )
        
        # Išspausdiname NED koordinates su offset
        print(f"Dronui {drone}, sekantis taškas: {point} -> NED: {shifted.north_m}, {shifted.east_m}, {shifted.down_m}")
        
        # Siunčiame poziciją dronui
        await drone.offboard.set_position_ned(shifted)
        await asyncio.sleep(7)  # Lėtas skrydis tarp taškų(tiek daug nes kartais nespeja stabilizuotis pries judant i kita taska)
    
    print(f"Dronas {drone} baigė savo trajektoriją.")

# Prisijungti prie drono
async def connect_drone(mavsdk_server_address, port):
    drone = System(mavsdk_server_address=mavsdk_server_address, port=port)
    #print(f"Jungiamasi prie {mavsdk_server_address}:{port}...")
    #print(f"Jungiamasi prie {addr}...")
    await drone.connect()

    async for state in drone.core.connection_state():
        if state.is_connected:
            #print(f"Dronas prisijungė prie {mavsdk_server_address}:{port}")
            break

    print("Laukiama globalios pozicijos...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print(" Globali pozicija OK.")
            break

    return drone

# Arm & takeoff
async def arm_and_takeoff(drone):
    await drone.action.arm()
    await asyncio.sleep(2)
    await drone.action.takeoff()
    await asyncio.sleep(11)

# Aktivuoja offboard režimą
async def activate_offboard(drone):
    async for position in drone.telemetry.position_velocity_ned():
        current_x = position.position.north_m
        current_y = position.position.east_m
        current_z = position.position.down_m
        break

    initial_position = PositionNedYaw(current_x, current_y, current_z, 0.0)

    try:
        for _ in range(10):
            await drone.offboard.set_position_ned(initial_position)
            await asyncio.sleep(0.1)

        await drone.offboard.start()
        await asyncio.sleep(3)
    except OffboardError as e:
        print(f" Offboard klaida: {e}")
        return False

    return True

# Nusileidžia
async def land_drone(drone):
    await drone.action.land()
    await asyncio.sleep(10)

# Sustabdo offboard režimą
async def stop_offboard_mode(drone):
    try:
        await drone.offboard.stop()
    except Exception as e:
        print(f"Klaida stabdant offboard: {e}")

# Kontroliuoti dronus
async def control_drones(area):
    
    drones_info = [
        ("localhost", 50060),
        ("localhost", 50061),
        ("localhost", 50062)
    ]
   

    '''
    drone_udp = [
        "udp://:14542",
        "udp://:14543",
        "udp://:14544"
    ]
    '''

    #drones = [await connect_drone(addr) for addr in drone_udp]

    # Prisijungiam prie visų dronų
    drones = [await connect_drone(addr, port) for addr, port in drones_info]

# Nustatom dronų pozicijas (offset'ai NED sistemoje)
    drone_offsets = [
    (0.0, 3.0),     # Drone 1 (centered)
    (0.0, 6.0),     # Drone 2
    (0.0, 9.0)     # Drone 3
    ]

    await asyncio.gather(*(arm_and_takeoff(drone) for drone in drones))
    await asyncio.gather(*(activate_offboard(drone) for drone in drones))

    # Sugeneruojam taškus teritorijoje
    points = generate_grid_points(area, step=10.0)

    # Grupuojam pagal K-means klasifikaciją
    n_clusters = len(drones)  # Tiek grupių, kiek dronų
    points_np = np.array(points)  # Konvertuojame į numpy masyvą, kad galėtume naudoti K-means
    labels, centers = kmeans_clustering(points_np, n_clusters)

    # Pasiruošiame maršrutus kiekvienam dronui pagal klasterius
    drone_routes = assign_drone_routes(points, labels, len(drones))

    # Spausdiname, kokie taškai priklauso kiekvienam dronui
    for i, route in enumerate(drone_routes):
        print(f"Dronui {i+1} priskirti taškai:")
        for point in route:
            print(f"  Taškas: {point}")
        print("")

    # Vizualizuojame taškų pasiskirstymą
    visualize_clusters(points, labels, centers)

    # Pasiruošiam maršrutus kiekvienam dronui pagal klasterius, naudojant artimiausio kaimyno metodą
    drone_routes_nn = []
    for route in drone_routes:
        drone_routes_nn.append(nearest_neighbor(route))

    # Spausdinam visus maršrutus, kuriuos paduodam dronams
    for i, route in enumerate(drone_routes_nn):
        print(f"\nDronui {i+1} maršrutas:")
        for point in route:
            print(f"  {point}")

    # Dronai skrenda pagal savo maršrutus
    await asyncio.gather(
         *(fly_area(drones[i], drone_routes_nn[i], drone_offsets[i]) for i in range(len(drones)))
     )


    await asyncio.gather(*(land_drone(drone) for drone in drones))
    await asyncio.gather(*(stop_offboard_mode(drone) for drone in drones))

if __name__ == "__main__":
    area = [(0.0, 0.0),(100, 0.0),(100.0, 100.0),(0.0, 100.0)]
    asyncio.run(control_drones(area))
