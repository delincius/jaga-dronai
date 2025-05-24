import streamlit as st
import folium
from streamlit_folium import st_folium
import numpy as np
import asyncio
from mavsdk import System
from mavsdk.offboard import OffboardError, PositionNedYaw
from scipy.spatial import distance
from sklearn.cluster import KMeans
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import json
from datetime import datetime
import threading
import concurrent.futures
import time
import sys
import traceback

# Import your existing functions (silently)
try:
    from control_drones_k_means import (
        generate_grid_points, kmeans_clustering, assign_drone_routes,
        nearest_neighbor, fly_area, connect_drone,
        arm_and_takeoff, activate_offboard, land_drone, stop_offboard_mode, control_drones
    )
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Drone Control System",
    page_icon="🚁",
    layout="wide"
)

# Initialize session state
if 'start_point' not in st.session_state:
    st.session_state.start_point = None
if 'boundary_points' not in st.session_state:
    st.session_state.boundary_points = []
if 'mission_running' not in st.session_state:
    st.session_state.mission_running = False
if 'mission_status' not in st.session_state:
    st.session_state.mission_status = "Ready"
if 'map_center' not in st.session_state:
    st.session_state.map_center = [54.8985, 23.9036]  # Kaunas center
if 'map_zoom' not in st.session_state:
    st.session_state.map_zoom = 13
if 'mode' not in st.session_state:
    st.session_state.mode = "View Only"

def console_log(message):
    """Simple console logging that works from threads"""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    full_message = f"[{timestamp}] {message}"
    print(full_message)

# Convert geographic coordinates to local NED
def geo_to_ned(lat, lon, ref_lat, ref_lon):
    """Convert geographic coordinates to NED (North-East-Down) coordinates"""
    R = 6371000
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    ref_lat_rad = np.radians(ref_lat)
    ref_lon_rad = np.radians(ref_lon)
    
    north = R * (lat_rad - ref_lat_rad)
    east = R * np.cos(ref_lat_rad) * (lon_rad - ref_lon_rad)
    
    return north, east

def get_area_coordinates():
    """Convert lat/lon boundary points to NED coordinates relative to start point"""
    if not st.session_state.start_point or len(st.session_state.boundary_points) < 3:
        return None
    
    ref_lat = st.session_state.start_point['lat']
    ref_lon = st.session_state.start_point['lng']
    
    area_coords = []
    for point in st.session_state.boundary_points:
        north, east = geo_to_ned(point['lat'], point['lng'], ref_lat, ref_lon)
        area_coords.append((north, east))
    
    return area_coords

def run_mission_with_debug(area_coordinates):
    """Run mission with debugging - NO SESSION STATE ACCESS"""
    def mission_thread():
        try:
            console_log("🚀 Starting mission...")
            console_log(f"📍 Area coordinates: {area_coordinates}")
            
            # Create new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            console_log("✅ Event loop created")
            
            # Call the function
            console_log("📞 Calling control_drones...")
            loop.run_until_complete(control_drones(area_coordinates))
            console_log("✅ Mission completed successfully!")
            
        except Exception as e:
            console_log(f"❌ Mission failed: {str(e)}")
            console_log(f"❌ Full traceback:\n{traceback.format_exc()}")
            
        finally:
            console_log("🏁 Mission thread finishing...")
            try:
                loop.close()
            except:
                console_log("⚠️ Error closing event loop")
    
    # Start the mission thread
    thread = threading.Thread(target=mission_thread)
    thread.daemon = True
    thread.start()
    console_log("✅ Mission thread started")
    
    return thread

def execute_mission_simulation():
    """Execute mission with debugging"""
    if not st.session_state.start_point or len(st.session_state.boundary_points) < 3:
        st.error("Missing start point or boundary points!")
        return
    
    area_coordinates = get_area_coordinates()
    
    if area_coordinates:
        console_log(f"✅ Starting mission with coordinates: {area_coordinates}")
        st.session_state.mission_running = True
        st.session_state.mission_status = "Mission Started - Check Console"
        
        # Start the mission
        run_mission_with_debug(area_coordinates)
        
        st.success("Mission started! Check your terminal/console for detailed progress.")
        
    else:
        st.error("Could not generate area coordinates!")

# Main UI
st.title("🚁 Drone Control System")
st.markdown("---")

# Sidebar for controls
with st.sidebar:
    st.header("Mission Controls")
    
    # Mode selection
    mode = st.radio(
        "Select Mode:",
        ["Set Start Point", "Add Boundary Points", "View Only"],
        key="mode_radio"
    )
    st.session_state.mode = mode
    
    # Clear buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Start"):
            st.session_state.start_point = None
            st.rerun()
    
    with col2:
        if st.button("Clear Boundary"):
            st.session_state.boundary_points = []
            st.rerun()
    
    st.markdown("---")
    
    # Display current points
    st.subheader("Current Configuration")
    
    if st.session_state.start_point:
        st.success(f"✅ Start Point: {st.session_state.start_point['lat']:.6f}, {st.session_state.start_point['lng']:.6f}")
    else:
        st.warning("⚠️ No start point set")
    
    if st.session_state.boundary_points:
        st.success(f"✅ Boundary Points: {len(st.session_state.boundary_points)}")
        with st.expander("View Boundary Points"):
            for i, point in enumerate(st.session_state.boundary_points):
                st.text(f"{i+1}: {point['lat']:.6f}, {point['lng']:.6f}")
    else:
        st.warning("⚠️ No boundary points set")
    
    st.markdown("---")
    
    # Mission parameters
    st.subheader("Mission Parameters")
    num_drones = st.number_input("Number of Drones", min_value=1, max_value=10, value=3)
    grid_step = st.slider("Grid Step Size (m)", min_value=5, max_value=50, value=10)
    altitude = st.slider("Flight Altitude (m)", min_value=5, max_value=100, value=5)
    
    st.markdown("---")
    
    # Mission control buttons
    if st.session_state.start_point and len(st.session_state.boundary_points) >= 3:
        if not st.session_state.mission_running:
            if st.button("🚀 Start Mission", type="primary"):
                execute_mission_simulation()
                st.rerun()
        else:
            if st.button("🛑 Stop Mission", type="secondary"):
                st.session_state.mission_running = False
                st.session_state.mission_status = "Mission Stopped"
                st.rerun()
    else:
        st.info("Set start point and at least 3 boundary points to start mission")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Mission Area Map")
    
    # Add instruction based on mode
    if st.session_state.mode == "Set Start Point":
        st.info("🎯 Click on the map to set the drone start point")
    elif st.session_state.mode == "Add Boundary Points":
        st.info("📍 Click on the map to add boundary points")
    else:
        st.info("👁️ View mode - map interactions disabled")
    
    # Create map using current session state center and zoom
    m = folium.Map(
        location=st.session_state.map_center,
        zoom_start=st.session_state.map_zoom,
        tiles='OpenStreetMap'
    )
    
    # Add start point if exists
    if st.session_state.start_point:
        folium.Marker(
            [st.session_state.start_point['lat'], st.session_state.start_point['lng']],
            popup="Start Point",
            tooltip="Drone Start Location",
            icon=folium.Icon(color='green', icon='play')
        ).add_to(m)
    
    # Add boundary points and polygon
    if st.session_state.boundary_points:
        for i, point in enumerate(st.session_state.boundary_points):
            folium.Marker(
                [point['lat'], point['lng']],
                popup=f"Boundary Point {i+1}",
                tooltip=f"Point {i+1}",
                icon=folium.Icon(color='red', icon='info-sign')
            ).add_to(m)
        
        if len(st.session_state.boundary_points) >= 3:
            polygon_points = [[p['lat'], p['lng']] for p in st.session_state.boundary_points]
            folium.Polygon(
                locations=polygon_points,
                color='blue',
                weight=2,
                fill=True,
                fillColor='lightblue',
                fillOpacity=0.3,
                popup="Search Area"
            ).add_to(m)
    
    # Display map and capture interactions
    map_data = st_folium(
        m,
        key="drone_map",
        width=700,
        height=500,
        returned_objects=["last_clicked", "center", "zoom"]
    )
    
    # Handle map clicks for adding points
    if map_data.get('last_clicked') is not None and st.session_state.mode != "View Only":
        clicked_lat = map_data['last_clicked']['lat']
        clicked_lng = map_data['last_clicked']['lng']
        
        # ONLY update map state when we're about to refresh (when adding points)
        # This preserves the current view across the refresh
        if map_data.get('center') is not None:
            st.session_state.map_center = [map_data['center']['lat'], map_data['center']['lng']]
        if map_data.get('zoom') is not None:
            st.session_state.map_zoom = map_data['zoom']
        
        if st.session_state.mode == "Set Start Point":
            st.session_state.start_point = {
                'lat': clicked_lat,
                'lng': clicked_lng
            }
            st.success(f"Start point set at: {clicked_lat:.6f}, {clicked_lng:.6f}")
            st.rerun()
        
        elif st.session_state.mode == "Add Boundary Points":
            new_point = {
                'lat': clicked_lat,
                'lng': clicked_lng
            }
            st.session_state.boundary_points.append(new_point)
            st.success(f"Boundary point {len(st.session_state.boundary_points)} added at: {clicked_lat:.6f}, {clicked_lng:.6f}")
            st.rerun()

with col2:
    st.subheader("Mission Status")
    
    # Display current mission status
    if st.session_state.mission_running:
        st.error("🔴 Mission in Progress")
        st.write(f"Status: {st.session_state.mission_status}")
        st.info("Check your terminal/console for detailed progress updates!")
    else:
        if "Failed" in st.session_state.mission_status:
            st.error(f"❌ {st.session_state.mission_status}")
        elif "Completed" in st.session_state.mission_status:
            st.success(f"✅ {st.session_state.mission_status}")
        else:
            st.success("✅ Ready for Mission")
    
    # Show preview if we have enough points
    if st.session_state.start_point and len(st.session_state.boundary_points) >= 3:
        st.subheader("Mission Preview")
        
        # Convert boundary points to NED
        ref_lat = st.session_state.start_point['lat']
        ref_lon = st.session_state.start_point['lng']
        
        ned_boundary = []
        for point in st.session_state.boundary_points:
            north, east = geo_to_ned(point['lat'], point['lng'], ref_lat, ref_lon)
            ned_boundary.append((north, east))
        
        try:
            # Generate and preview grid points
            grid_points = generate_grid_points(ned_boundary, step=grid_step)
            
            if grid_points:
                points_np = np.array(grid_points)
                labels, centers = kmeans_clustering(points_np, num_drones)
                
                # Create preview visualization
                fig, ax = plt.subplots(figsize=(5, 5))
                scatter = ax.scatter(*zip(*grid_points), c=labels, cmap='viridis', marker='o', alpha=0.7, s=20)
                ax.scatter(*zip(*centers), c='red', marker='x', s=80, label='Centroids')
                ax.set_xlabel("North (m)")
                ax.set_ylabel("East (m)")
                ax.set_title("Mission Preview")
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)
                
                # Preview statistics
                st.metric("Preview: Total Points", len(grid_points))
            else:
                st.warning("No valid grid points in preview")
        except Exception as e:
            st.error(f"Preview error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("🚁 **Drone Control System**")

# Optional debug info (hidden by default)
if st.checkbox("Show Debug Info"):
    if st.session_state.start_point and st.session_state.boundary_points:
        area_coords = get_area_coordinates()
        if area_coords:
            st.write("**Area coordinates that will be sent to control_drones:**")
            st.code(f"area_coordinates = {area_coords}")
        else:
            st.write("No valid area coordinates")
    
    # Show current map state
    st.write("**Current map state:**")
    st.write(f"Center: {st.session_state.map_center}")
    st.write(f"Zoom: {st.session_state.map_zoom}")