import streamlit as st
import streamlit.components.v1
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
import subprocess
import os
import signal

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
if 'request_location' not in st.session_state:
    st.session_state.request_location = False
if 'user_location' not in st.session_state:
    st.session_state.user_location = None
if 'camera_processes' not in st.session_state:
    st.session_state.camera_processes = []
if 'mission_thread' not in st.session_state:
    st.session_state.mission_thread = None

# Global variable to store camera processes (avoids Streamlit threading issues)
active_camera_processes = []

def console_log(message):
    """Simple console logging that works from threads"""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    full_message = f"[{timestamp}] {message}"
    print(full_message)

# Camera control functions
def monitor_camera_output(process, drone_number):
    """Monitor and log camera process output"""
    try:
        for line in iter(process.stdout.readline, ''):
            if line:
                console_log(f"📸 [Drone {drone_number}]: {line.strip()}")
    except Exception as e:
        console_log(f"❌ Error monitoring Drone {drone_number}: {e}")

def start_camera_capture(drone_configs, capture_interval=5):
    """Start camera capture processes for all drones"""
    global active_camera_processes
    processes = []
    
    # Check if video_capture.py exists
    video_capture_script = "video_capture.py"
    if not os.path.exists(video_capture_script):
        console_log(f"❌ ERROR: {video_capture_script} not found!")
        console_log("   Please ensure the video capture script is in the same directory as this GUI script.")
        return []
    
    # Create output directory for images
    output_dir = os.path.join("captured_images", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(output_dir, exist_ok=True)
    
    console_log(f"📸 Starting camera capture, saving to: {output_dir}")
    console_log(f"📸 Starting {len(drone_configs)} camera processes...")
    console_log(f"📸 Capture interval: {capture_interval} seconds")
    
    for i, (video_port, drone_number) in enumerate(drone_configs):
        try:
            # Build command to run the video capture script
            cmd = [
                sys.executable,  # Python executable
                video_capture_script,
                "--port", str(video_port),
                "--number", str(drone_number),
                "--interval", str(capture_interval)
            ]
            
            # Set environment variable to specify output directory
            env = os.environ.copy()
            env['OUTPUT_DIR'] = output_dir
            
            console_log(f"🚁 Starting camera for Drone {drone_number}...")
            console_log(f"   Command: {' '.join(cmd)}")
            console_log(f"   Port: {video_port}")
            
            # Start the process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Combine stderr with stdout
                env=env,
                bufsize=1,  # Line buffered
                universal_newlines=True,  # Text mode
                preexec_fn=os.setsid if os.name != 'nt' else None  # Create new process group
            )
            
            # Start a thread to monitor this camera's output
            monitor_thread = threading.Thread(
                target=monitor_camera_output,
                args=(process, drone_number),
                daemon=True
            )
            monitor_thread.start()
            
            processes.append(process)
            console_log(f"✅ Started camera capture for Drone {drone_number} (PID: {process.pid})")
            
            # Give each process time to start properly
            time.sleep(1)
            
        except Exception as e:
            console_log(f"❌ Failed to start camera for Drone {drone_number}: {str(e)}")
            traceback.print_exc()
    
    console_log(f"📸 Total camera processes started: {len(processes)}")
    
    # Store processes globally to avoid Streamlit threading issues
    active_camera_processes = processes
    return processes

def stop_camera_capture(processes):
    """Stop all camera capture processes"""
    console_log(f"📸 Stopping {len(processes)} camera processes...")
    
    for i, process in enumerate(processes):
        try:
            drone_num = i + 1
            console_log(f"🛑 Stopping camera for Drone {drone_num} (PID: {process.pid})...")
            
            if os.name == 'nt':  # Windows
                # On Windows, just terminate
                process.terminate()
            else:  # Unix/Linux/Mac
                # Send SIGTERM to the process group
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            
            # Wait for process to finish (with timeout)
            try:
                process.wait(timeout=5)
                console_log(f"✅ Stopped camera process for Drone {drone_num}")
            except subprocess.TimeoutExpired:
                # Force kill if it doesn't terminate gracefully
                if os.name == 'nt':
                    process.kill()
                else:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                console_log(f"⚠️ Force killed camera process for Drone {drone_num}")
                
        except ProcessLookupError:
            console_log(f"ℹ️ Camera process {i+1} already terminated")
        except Exception as e:
            console_log(f"❌ Error stopping camera process {i+1}: {str(e)}")
    
    # Clear the global processes list
    processes.clear()
    console_log("📸 All camera processes stopped")

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

def run_mission_with_debug(area_coordinates, num_drones, capture_interval=5):
    """Run mission with debugging and camera capture"""
    global active_camera_processes
    
    def mission_thread():
        try:
            console_log("🚀 Starting mission...")
            console_log(f"📍 Area coordinates: {area_coordinates}")
            console_log(f"🚁 Number of drones: {num_drones}")
            console_log(f"📸 Capture interval: {capture_interval} seconds")
            
            # Define camera configurations (port, drone_number)
            # Adjust these ports based on your actual camera stream ports
            camera_configs = [
                (5600, 1),  # Drone 1 camera on port 5600
                (5601, 2),  # Drone 2 camera on port 5601
                (5602, 3),  # Drone 3 camera on port 5602
            ][:num_drones]  # Only use cameras for active drones
            
            # Start camera capture with interval
            camera_processes = start_camera_capture(camera_configs, capture_interval)
            
            # Wait a bit to ensure all cameras are initialized
            console_log("⏳ Waiting for cameras to initialize...")
            time.sleep(2)
            
            # Create new event loop for drone control
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            console_log("✅ Event loop created")
            
            # Call the drone control function
            console_log("📞 Calling control_drones...")
            loop.run_until_complete(control_drones(area_coordinates))
            console_log("✅ Mission completed successfully!")
            
        except Exception as e:
            console_log(f"❌ Mission failed: {str(e)}")
            console_log(f"❌ Full traceback:\n{traceback.format_exc()}")
            
        finally:
            console_log("🏁 Mission thread finishing...")
            
            # Stop camera capture
            if active_camera_processes:
                stop_camera_capture(active_camera_processes)
            
            try:
                loop.close()
            except:
                console_log("⚠️ Error closing event loop")
    
    # Start the mission thread
    thread = threading.Thread(target=mission_thread)
    thread.daemon = True
    thread.start()
    console_log("✅ Mission thread started")
    
    # Store camera process count in session state for UI display
    st.session_state.camera_processes = list(range(num_drones))  # Just store count for UI
    
    return thread

def execute_mission_simulation():
    """Execute mission with debugging and camera capture"""
    if not st.session_state.start_point or len(st.session_state.boundary_points) < 3:
        st.error("Missing start point or boundary points!")
        return
    
    area_coordinates = get_area_coordinates()
    
    if area_coordinates:
        # Get parameters from sidebar
        num_drones = st.session_state.get('num_drones', 3)
        capture_interval = st.session_state.get('capture_interval', 5)
        
        console_log(f"✅ Starting mission with {num_drones} drones")
        console_log(f"✅ Area coordinates: {area_coordinates}")
        
        st.session_state.mission_running = True
        st.session_state.mission_status = "Mission Started - Check Console"
        
        # Start the mission with camera capture
        mission_thread = run_mission_with_debug(area_coordinates, num_drones, capture_interval)
        st.session_state.mission_thread = mission_thread
        
        st.success("Mission started! Check your terminal/console for detailed progress.")
        
    else:
        st.error("Could not generate area coordinates!")

def stop_mission():
    """Stop the mission and camera capture"""
    global active_camera_processes
    console_log("🛑 Stopping mission...")
    
    # Stop camera processes
    if active_camera_processes:
        stop_camera_capture(active_camera_processes)
    
    # Update status
    st.session_state.mission_running = False
    st.session_state.mission_status = "Mission Stopped"
    st.session_state.camera_processes = []
    
    console_log("✅ Mission stopped")

# Main UI
st.title("🚁 Drone Control System")
st.markdown("---")

# Handle geolocation request
if st.session_state.request_location:
    # Simple auto-location with immediate callback
    location_html = """
    <script>
    function setLocationAndReload(lat, lng) {
        // Set URL parameters and reload
        const url = new URL(window.location);
        url.searchParams.set('auto_lat', lat);
        url.searchParams.set('auto_lng', lng);
        window.location.href = url.toString();
    }
    
    if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition(
            function(position) {
                const lat = position.coords.latitude;
                const lng = position.coords.longitude;
                console.log('Location found:', lat, lng);
                setLocationAndReload(lat, lng);
            },
            function(error) {
                console.error('Location error:', error);
                alert('Location access denied or unavailable. Please enable location access and try again.');
                // Go back without setting location
                const url = new URL(window.location);
                url.searchParams.delete('auto_lat');
                url.searchParams.delete('auto_lng');
                window.location.href = url.toString();
            },
            {
                enableHighAccuracy: true,
                timeout: 10000,
                maximumAge: 300000
            }
        );
    } else {
        alert('Geolocation is not supported by this browser.');
        window.history.back();
    }
    </script>
    <div style="padding: 20px; text-align: center;">
        <div style="font-size: 18px;">🌍 Getting your location...</div>
        <div style="margin-top: 10px; color: #666;">Please allow location access when prompted</div>
    </div>
    """
    
    st.components.v1.html(location_html, height=100)

# Check for auto-location from URL parameters
try:
    query_params = st.query_params
    if 'auto_lat' in query_params and 'auto_lng' in query_params:
        lat = float(query_params['auto_lat'])
        lng = float(query_params['auto_lng'])
        
        # Set the location automatically
        st.session_state.start_point = {'lat': lat, 'lng': lng}
        st.session_state.map_center = [lat, lng] 
        st.session_state.map_zoom = 16
        st.session_state.request_location = False
        
        # Clear the URL parameters
        st.query_params.clear()
        
        st.success(f"✅ Start point set to your location: {lat:.6f}, {lng:.6f}")
        st.rerun()
except:
    # Handle any query param errors silently
    pass

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
    
    # Clear buttons and location button
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Clear Start"):
            st.session_state.start_point = None
            st.rerun()
    
    with col2:
        if st.button("Clear Boundary"):
            st.session_state.boundary_points = []
            st.rerun()
            
    with col3:
        if st.button("📍 Try My Location"):
            # This will trigger the geolocation JavaScript
            st.session_state.request_location = True
            st.rerun()
    
    st.info("💡 **Tip**: For accurate positioning, switch to 'Set Start Point' mode and click directly on the map where you are located. GPS location can be inaccurate.")
    
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
    num_drones = st.number_input("Number of Drones", min_value=1, max_value=10, value=3, key="num_drones")
    grid_step = st.slider("Grid Step Size (m)", min_value=5, max_value=50, value=10)
    altitude = st.slider("Flight Altitude (m)", min_value=5, max_value=100, value=5)
    
    # Camera settings
    st.markdown("---")
    st.subheader("Camera Settings")
    capture_interval = st.slider("Image Capture Interval (s)", min_value=1, max_value=30, value=5, key="capture_interval")
    st.info(f"📸 Cameras will capture images every {capture_interval} seconds during flight")
    
    st.markdown("---")
    
    # Mission control buttons
    if st.session_state.start_point and len(st.session_state.boundary_points) >= 3:
        if not st.session_state.mission_running:
            if st.button("🚀 Start Mission", type="primary"):
                execute_mission_simulation()
                st.rerun()
        else:
            if st.button("🛑 Stop Mission", type="secondary"):
                stop_mission()
                st.rerun()
    else:
        st.info("Set start point and at least 3 boundary points to start mission")
    
    # Display captured images info
    if st.session_state.mission_running:
        st.markdown("---")
        st.subheader("📸 Camera Status")
        st.info(f"Recording from {len(st.session_state.camera_processes)} cameras")
        st.text("Images saved to:")
        st.code("captured_images/[timestamp]")

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
        
        # Show camera status
        if st.session_state.camera_processes:
            st.success(f"📸 {len(st.session_state.camera_processes)} cameras recording")
        
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
st.markdown("🚁 **Drone Control System** | 📸 **With Integrated Camera Capture**")

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
    
    # Show camera process info
    if st.session_state.camera_processes:
        st.write("**Active camera processes:**")
        st.write(f"Count: {len(st.session_state.camera_processes)}")