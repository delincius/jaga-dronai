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
import streamlit as st
import threading

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

def visualize_multiple_drone_paths_and_area(drone_routes, centers=None, area=None):
    plt.figure(figsize=(10, 10))
    colors = plt.cm.get_cmap('tab10', len(drone_routes))  # Skirtinga spalva kiekvienam dronui

    # Piešimas dronų kelių
    for i, route in enumerate(drone_routes):
        # Konvertuojame iš NED į matplotlib koordinatę: (Y, X)
        transposed_route = [(y, x) for (x, y) in route]
        plt.plot(*zip(*transposed_route), marker='o', label=f'Dronas {i+1}', color=colors(i))

    # Piešimas centroidų, jei pateikti
    if centers is not None and len(centers) > 0:
        transposed_centers = [(y, x) for (x, y) in centers]
        plt.scatter(*zip(*transposed_centers), c='red', marker='x', s=100, label='Centroidai')

    # Piešimas teritorijos (area), jei pateikta
    if area is not None and len(area) > 0:
        # Konvertuojame iš NED į matplotlib koordinates
        transposed_area = [(y, x) for (x, y) in area]
        # Užpildome teritoriją
        transposed_area.append(transposed_area[0])  # Uždaryti teritoriją
        plt.fill(*zip(*transposed_area), color='lightgray', alpha=0.5, label='Teritorija')

    plt.title("Dronų maršrutai (K-Means + NN metodas)")
    plt.xlabel("Rytai (X)")
    plt.ylabel("Šiaurė (Y)")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

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

# Skrenda pagal maršrutą (original version - kept for compatibility)
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

# NEW: Fly route with cancellation support
async def fly_route_with_cancellation(drone, route, offset, cancel_event, drone_number):
    """Fly a single drone route with cancellation checking"""
    print(f"🚁 Drone {drone_number}: Starting route with {len(route)} waypoints")
    
    for i, point in enumerate(route):
        # Check for cancellation before each waypoint
        if cancel_event and cancel_event.is_set():
            print(f"🛑 Drone {drone_number}: Mission cancelled at waypoint {i+1}/{len(route)}")
            raise asyncio.CancelledError(f"Drone {drone_number} mission cancelled")
        
        # Calculate target position with offset
        shifted = PositionNedYaw(
            point[0] + offset[0],  # North
            point[1] + offset[1],  # East
            -15.0,                 # Down (altitude)
            0.0                    # Yaw
        )
        
        print(f"🚁 Drone {drone_number}: Going to waypoint {i+1}/{len(route)} - N:{shifted.north_m:.1f}, E:{shifted.east_m:.1f}")
        
        # Send position command
        await drone.offboard.set_position_ned(shifted)
        
        # Wait at waypoint with periodic cancellation checking
        for _ in range(7):  # 7 seconds total, check every second
            if cancel_event and cancel_event.is_set():
                print(f"🛑 Drone {drone_number}: Mission cancelled while at waypoint {i+1}")
                raise asyncio.CancelledError(f"Drone {drone_number} mission cancelled")
            await asyncio.sleep(1)
    
    print(f"✅ Drone {drone_number}: Completed route successfully")

# NEW: Return drones to home positions
async def return_drones_to_home(drones, drone_offsets):
    """Return all drones to their starting positions"""
    print("🏠 Returning drones to home positions...")
    
    # Return each drone to its starting position (offset from origin)
    home_tasks = []
    for i, (drone, offset) in enumerate(zip(drones, drone_offsets)):
        home_position = PositionNedYaw(
            offset[0],  # North offset
            offset[1],  # East offset
            -15.0,      # Same altitude as mission
            0.0         # Yaw
        )
        print(f"🏠 Sending Drone {i+1} to home: N={offset[0]}, E={offset[1]}")
        home_tasks.append(drone.offboard.set_position_ned(home_position))
    
    # Send all drones home simultaneously
    await asyncio.gather(*home_tasks)
    
    # Wait for drones to reach home positions
    print("🏠 Waiting for drones to reach home positions...")
    await asyncio.sleep(10)  # Give time to reach home
    
    print("✅ All drones returned to home positions")

# NEW: Execute drone routes with cancellation support
async def fly_area_with_cancellation(drones, drone_routes_nn, drone_offsets, cancel_event):
    """Execute drone routes with cancellation support"""
    print(f"🚁 Starting flight execution for {len(drones)} drones...")
    
    # Create tasks for each drone
    flight_tasks = []
    for i in range(len(drones)):
        task = fly_route_with_cancellation(
            drones[i], 
            drone_routes_nn[i], 
            drone_offsets[i], 
            cancel_event,
            i + 1  # Drone number for logging
        )
        flight_tasks.append(task)
    
    # Execute all drone flights concurrently
    await asyncio.gather(*flight_tasks)

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
            print("✅ Globali pozicija OK.")
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
        print(f"❌ Offboard klaida: {e}")
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

# NEW: Main control function with cancellation support
async def control_drones_with_cancellation(area, cancel_event=None):
    """Modified control_drones function that supports cancellation"""
    
    drones_info = [
        ("localhost", 50060),
        ("localhost", 50061),
        ("localhost", 50062)
    ]

    # Connect to all drones
    print("🔗 Connecting to drones...")
    drones = [await connect_drone(addr, port) for addr, port in drones_info]

    # Set drone position offsets (NED system)
    drone_offsets = [
        (0.0, 3.0),     # Drone 1
        (0.0, 6.0),     # Drone 2
        (0.0, 9.0)      # Drone 3
    ]

    try:
        # Arm and takeoff
        print("🚁 Arming and taking off...")
        await asyncio.gather(*(arm_and_takeoff(drone) for drone in drones))
        
        # Check for cancellation after takeoff
        if cancel_event and cancel_event.is_set():
            print("🛑 Mission cancelled during takeoff")
            raise asyncio.CancelledError("Mission cancelled during takeoff")
        
        await asyncio.gather(*(activate_offboard(drone) for drone in drones))

        # Generate mission waypoints
        print("📍 Generating mission waypoints...")
        points = generate_grid_points(area, step=10.0)
        n_clusters = len(drones)
        points_np = np.array(points)
        labels, centers = kmeans_clustering(points_np, n_clusters)
        drone_routes = assign_drone_routes(points, labels, len(drones))
        
        # Optimize routes with nearest neighbor
        drone_routes_nn = []
        for route in drone_routes:
            drone_routes_nn.append(nearest_neighbor(route))

        # Print route assignments
        for i, route in enumerate(drone_routes_nn):
            print(f"\n🚁 Drone {i+1} route ({len(route)} waypoints):")
            for j, point in enumerate(route):
                print(f"  Waypoint {j+1}: {point}")

        # Execute mission with cancellation checking
        print("🚁 Starting mission execution...")
        await fly_area_with_cancellation(drones, drone_routes_nn, drone_offsets, cancel_event)

    except asyncio.CancelledError:
        print("🛑 Mission cancelled - initiating return to home...")
        
        # Return drones to home positions
        try:
            await return_drones_to_home(drones, drone_offsets)
        except Exception as e:
            print(f"❌ Error returning drones home: {e}")
        
        # Land drones
        print("🛬 Landing drones...")
        await asyncio.gather(*(land_drone(drone) for drone in drones))
        
        # Stop offboard mode
        print("🔌 Stopping offboard mode...")
        await asyncio.gather(*(stop_offboard_mode(drone) for drone in drones))
        
        print("✅ Mission cancelled successfully - all drones landed")
        raise  # Re-raise to indicate cancellation
        
    except Exception as e:
        print(f"❌ Unexpected error during mission: {e}")
        raise
    
    else:
        # Normal mission completion
        print("✅ Mission completed normally - landing drones...")
        await asyncio.gather(*(land_drone(drone) for drone in drones))
        await asyncio.gather(*(stop_offboard_mode(drone) for drone in drones))
        print("✅ All drones landed successfully")

# ORIGINAL: Keep original control_drones function for backwards compatibility
async def control_drones(area):
    """Original control_drones function - kept for backwards compatibility"""
    
    drones_info = [
        ("localhost", 50060),
        ("localhost", 50061),
        ("localhost", 50062)
    ]

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