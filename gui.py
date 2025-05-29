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
        generate_grid_points, kmeans_clustering, assign_drone_routes, visualize_multiple_drone_paths_and_area,
        nearest_neighbor, fly_area, connect_drone,
        arm_and_takeoff, activate_offboard, land_drone, stop_offboard_mode, control_drones, control_drones_with_cancellation
    )
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Drone Control System with Red Object Detection",
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
if 'mission_cancel_event' not in st.session_state:
    st.session_state.mission_cancel_event = None
if 'detection_process' not in st.session_state:
    st.session_state.detection_process = None
if 'detection_thread' not in st.session_state:
    st.session_state.detection_thread = None
if 'detection_results' not in st.session_state:
    st.session_state.detection_results = []
if 'last_detection_check' not in st.session_state:
    st.session_state.last_detection_check = 0
if 'detection_notifications' not in st.session_state:
    st.session_state.detection_notifications = []
if 'detection_markers' not in st.session_state:
    st.session_state.detection_markers = []
if 'map_refresh_key' not in st.session_state:
    st.session_state.map_refresh_key = 0
# SIMPLE FIX: Only track markers count to trigger refresh
if 'last_marker_count' not in st.session_state:
    st.session_state.last_marker_count = 0

# Global variables
active_camera_processes = []
active_detection_process = None
detection_results_file = "detection_results.json"
detection_thread_running = False
processed_detections = set()

# Global mission configuration for background threads
mission_config = {
    'start_point': None,
    'boundary_points': []
}

def console_log(message):
    """Simple console logging that works from threads"""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    full_message = f"[{timestamp}] {message}"
    print(full_message)

# Camera control functions (same as before)
def monitor_camera_output(process, drone_number):
    """Monitor and log camera process output"""
    try:
        for line in iter(process.stdout.readline, ''):
            if line:
                console_log(f"📹 [Drone {drone_number}]: {line.strip()}")
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
    
    console_log(f"📹 Starting camera capture, saving to: {output_dir}")
    console_log(f"📹 Starting {len(drone_configs)} camera processes...")
    console_log(f"📹 Capture interval: {capture_interval} seconds")
    
    for i, (video_port, drone_number) in enumerate(drone_configs):
        try:
            cmd = [
                sys.executable,
                video_capture_script,
                "--port", str(video_port),
                "--number", str(drone_number),
                "--interval", str(capture_interval)
            ]
            
            env = os.environ.copy()
            env['OUTPUT_DIR'] = output_dir
            
            console_log(f"📹 Starting camera for Drone {drone_number}...")
            console_log(f"   Command: {' '.join(cmd)}")
            console_log(f"   Port: {video_port}")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                bufsize=1,
                universal_newlines=True,
                preexec_fn=os.setsid if os.name != 'nt' else None
            )
            
            monitor_thread = threading.Thread(
                target=monitor_camera_output,
                args=(process, drone_number),
                daemon=True
            )
            monitor_thread.start()
            
            processes.append(process)
            console_log(f"✅ Started camera capture for Drone {drone_number} (PID: {process.pid})")
            
            time.sleep(1)
            
        except Exception as e:
            console_log(f"❌ Failed to start camera for Drone {drone_number}: {str(e)}")
            traceback.print_exc()
    
    console_log(f"📹 Total camera processes started: {len(processes)}")
    
    active_camera_processes = processes
    return processes, output_dir

def stop_camera_capture(processes):
    """Stop all camera capture processes"""
    console_log(f"📹 Stopping {len(processes)} camera processes...")
    
    for i, process in enumerate(processes):
        try:
            drone_num = i + 1
            console_log(f"📹 Stopping camera for Drone {drone_num} (PID: {process.pid})...")
            
            if os.name == 'nt':
                process.terminate()
            else:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            
            try:
                process.wait(timeout=5)
                console_log(f"✅ Stopped camera process for Drone {drone_num}")
            except subprocess.TimeoutExpired:
                if os.name == 'nt':
                    process.kill()
                else:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                console_log(f"⚠️ Force killed camera process for Drone {drone_num}")
                
        except ProcessLookupError:
            console_log(f"ℹ️ Camera process {i+1} already terminated")
        except Exception as e:
            console_log(f"❌ Error stopping camera process {i+1}: {str(e)}")
    
    processes.clear()
    console_log("📹 All camera processes stopped")

# Detection service functions
def start_detection_service(output_dir):
    """Start the red object detection service"""
    global active_detection_process
    
    detection_script = "red_object_detection.py"
    if not os.path.exists(detection_script):
        console_log(f"❌ ERROR: {detection_script} not found!")
        console_log("   Please ensure the detection script is in the same directory as this GUI script.")
        return None
    
    try:
        if os.path.exists(detection_results_file):
            os.remove(detection_results_file)
        
        cmd = [
            sys.executable,
            detection_script,
            "--folder", output_dir,
            "--pattern", "*.jpg",
            "--mode", "monitor"
        ]
        
        console_log(f"🔍 Starting red object detection service...")
        console_log(f"   Command: {' '.join(cmd)}")
        console_log(f"   Monitoring folder: {output_dir}")
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
            preexec_fn=os.setsid if os.name != 'nt' else None
        )
        
        monitor_thread = threading.Thread(
            target=monitor_detection_output,
            args=(process,),
            daemon=True
        )
        monitor_thread.start()
        
        active_detection_process = process
        console_log(f"✅ Started detection service (PID: {process.pid})")
        
        return process
        
    except Exception as e:
        console_log(f"❌ Failed to start detection service: {str(e)}")
        traceback.print_exc()
        return None

def monitor_detection_output(process):
    """Monitor and log detection process output"""
    try:
        for line in iter(process.stdout.readline, ''):
            if line:
                console_log(f"🔍 [Detection]: {line.strip()}")
    except Exception as e:
        console_log(f"❌ Error monitoring detection process: {e}")

def stop_detection_service(process):
    """Stop the detection service process"""
    if not process:
        return
    
    try:
        console_log(f"🔍 Stopping detection service (PID: {process.pid})...")
        
        if os.name == 'nt':
            process.terminate()
        else:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        
        try:
            process.wait(timeout=5)
            console_log(f"✅ Stopped detection service")
        except subprocess.TimeoutExpired:
            if os.name == 'nt':
                process.kill()
            else:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            console_log(f"⚠️ Force killed detection service")
            
    except ProcessLookupError:
        console_log(f"ℹ️ Detection service already terminated")
    except Exception as e:
        console_log(f"❌ Error stopping detection service: {str(e)}")

def check_for_new_detections():
    """Check for new detection results"""
    try:
        if not os.path.exists(detection_results_file):
            return []
        
        with open(detection_results_file, 'r') as f:
            all_results = json.load(f)
        
        return all_results[-10:] if all_results else []
        
    except Exception as e:
        console_log(f"❌ Error checking detection results: {e}")
        return []

def detection_monitor_thread():
    """Background thread to monitor detection results"""
    global detection_thread_running, processed_detections
    
    console_log("🔍 Starting detection monitoring thread...")
    detection_thread_running = True
    processed_detections = set()
    
    while detection_thread_running:
        try:
            new_detections = check_for_new_detections()
            
            for detection in new_detections:
                detection_id = f"{detection['filename']}_{detection['timestamp']}"
                
                if detection_id in processed_detections:
                    continue
                
                processed_detections.add(detection_id)
                
                if detection['has_red_objects']:
                    drone_number = detection.get('drone_number', 1)
                    
                    # SIMPLE FIX: Use the mission config directly
                    estimated_location = get_drone_position_simple(drone_number)
                    
                    if not estimated_location:
                        console_log("⚠️ Using fallback location")
                        estimated_location = (54.8985 + (drone_number * 0.001), 23.9036 + (drone_number * 0.001))
                    
                    notification = {
                        'timestamp': detection['timestamp'],
                        'message': f"🔴 RED OBJECT DETECTED!",
                        'details': f"Drone {drone_number}: {detection['object_count']} objects ({detection['confidence']:.1f}% confidence)",
                        'filename': detection['filename'],
                        'drone_number': drone_number,
                        'estimated_lat': estimated_location[0],
                        'estimated_lng': estimated_location[1]
                    }
                    
                    try:
                        # Write notification
                        with open("detection_notifications.json", 'w') as f:
                            json.dump([notification], f, indent=2)
                        
                        # IMPROVED: Better marker file handling with validation
                        markers_file = "detection_markers.json"
                        existing_markers = []
                        
                        if os.path.exists(markers_file):
                            try:
                                with open(markers_file, 'r') as f:
                                    content = f.read().strip()
                                    if content:
                                        existing_markers = json.loads(content)
                                    else:
                                        existing_markers = []
                            except (json.JSONDecodeError, ValueError) as e:
                                console_log(f"⚠️ Corrupted marker file, recreating: {e}")
                                existing_markers = []
                        
                        marker = {
                            'lat': float(estimated_location[0]),
                            'lng': float(estimated_location[1]),
                            'timestamp': detection['timestamp'],
                            'drone_number': drone_number,
                            'confidence': detection['confidence'],
                            'object_count': detection['object_count'],
                            'filename': detection['filename']
                        }
                        existing_markers.append(marker)
                        
                        # Keep only last 20 markers
                        if len(existing_markers) > 20:
                            existing_markers = existing_markers[-20:]
                        
                        # Write with proper formatting and validation
                        try:
                            with open(markers_file, 'w') as f:
                                json.dump(existing_markers, f, indent=2, ensure_ascii=False)
                        except Exception as write_error:
                            console_log(f"❌ Error writing markers file: {write_error}")
                            # Fallback: try to write without formatting
                            with open(markers_file, 'w') as f:
                                json.dump(existing_markers, f)
                        
                        console_log(f"🔴 NEW DETECTION: Drone {drone_number} found {detection['object_count']} red objects ({detection['confidence']:.1f}% confidence)")
                        console_log(f"📍 Marker added at {estimated_location[0]:.6f}, {estimated_location[1]:.6f}")
                        console_log(f"📍 Total markers in file: {len(existing_markers)}")
                        
                        # IMPROVED: Write trigger with marker count for validation
                        with open("map_refresh_trigger.txt", 'w') as f:
                            f.write(f"{len(existing_markers)}")
                        
                    except Exception as e:
                        console_log(f"❌ Error writing detection files: {e}")
                        traceback.print_exc()
            
            if len(processed_detections) > 200:
                processed_detections = set(list(processed_detections)[-100:])
            
            time.sleep(2)
            
        except Exception as e:
            console_log(f"❌ Error in detection monitoring: {e}")
            time.sleep(5)
    
    console_log("🔍 Detection monitoring thread stopped")

def get_drone_position_simple(drone_number):
    """SIMPLE drone position calculation"""
    try:
        start_point = mission_config.get('start_point')
        boundary_points = mission_config.get('boundary_points', [])
        
        if not start_point:
            return None
        
        start_lat = float(start_point['lat'])
        start_lng = float(start_point['lng'])
        
        if len(boundary_points) >= 3:
            lats = [float(p['lat']) for p in boundary_points]
            lngs = [float(p['lng']) for p in boundary_points]
            
            lat_range = max(lats) - min(lats)
            lng_range = max(lngs) - min(lngs)
            
            if drone_number == 1:
                offset_lat = lat_range * 0.3
                offset_lng = lng_range * 0.1
            elif drone_number == 2:
                offset_lat = lat_range * 0.1
                offset_lng = lng_range * 0.4
            else:
                offset_lat = -lat_range * 0.2
                offset_lng = -lng_range * 0.1
            
            return (start_lat + offset_lat, start_lng + offset_lng)
        
        # Simple fallback
        offset = drone_number * 0.002
        return (start_lat + offset, start_lng + offset)
        
    except Exception as e:
        console_log(f"❌ Error calculating drone position: {e}")
        return None

def clear_detection_markers():
    """Clear all detection markers"""
    global processed_detections
    
    try:
        files_to_remove = [
            "detection_markers.json", 
            "detection_notifications.json",
            "map_refresh_trigger.txt"
        ]
        
        for file in files_to_remove:
            if os.path.exists(file):
                os.remove(file)
        
        st.session_state.detection_markers = []
        st.session_state.detection_notifications = []
        st.session_state.last_marker_count = 0
        
        processed_detections = set()
        
        console_log("🗑️ All detection markers cleared")
        return True
    except Exception as e:
        console_log(f"❌ Error clearing markers: {e}")
        return False

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
    """Run mission with debugging, camera capture, and detection service"""
    global active_camera_processes, active_detection_process

    cancel_event = st.session_state.mission_cancel_event

    def mission_thread():
        global detection_thread_running
    
        try:
            console_log("🚁 Starting mission...")
            console_log(f"🚁 Area coordinates: {area_coordinates}")
            console_log(f"🚁 Number of drones: {num_drones}")
            console_log(f"🚁 Capture interval: {capture_interval} seconds")
            
            camera_configs = [
                (5601, 1),
                (5602, 2),
                (5603, 3),
            ][:num_drones]
            
            camera_processes, output_dir = start_camera_capture(camera_configs, capture_interval)
            detection_process = start_detection_service(output_dir)
            
            detection_thread_running = True
            detection_thread = threading.Thread(target=detection_monitor_thread, daemon=True)
            detection_thread.start()
            
            console_log("🚁 Waiting for systems to initialize...")
            time.sleep(3)
            
            st.session_state.last_detection_check = time.time()
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            console_log("🚁 Event loop created")
            
            console_log("🚁 Calling control_drones...")
            loop.run_until_complete(control_drones_with_cancellation(
                area_coordinates, 
                cancel_event
            ))
            console_log("✅ Mission completed successfully!")
        
        except asyncio.CancelledError:
            console_log("✅ Mission was cancelled and handled successfully")

        except Exception as e:
            console_log(f"❌ Mission failed: {str(e)}")
            console_log(f"❌ Full traceback:\n{traceback.format_exc()}")
        
        finally:
            console_log("🚁 Mission thread finishing...")
            
            detection_thread_running = False
            
            if active_camera_processes:
                stop_camera_capture(active_camera_processes)
            
            if active_detection_process:
                stop_detection_service(active_detection_process)
            
            try:
                loop.close()
            except:
                console_log("❌ Error closing event loop")
            
            console_log("🚁 Mission thread completed")
    
    thread = threading.Thread(target=mission_thread)
    thread.daemon = True
    thread.start()
    console_log("🚁 Mission thread started")
    
    st.session_state.camera_processes = list(range(num_drones))
    
    return thread

def execute_mission_simulation():
    """Execute mission with debugging, camera capture, and detection service"""
    global mission_config, processed_detections
    
    if not st.session_state.start_point or len(st.session_state.boundary_points) < 3:
        st.error("Missing start point or boundary points!")
        return
    
    mission_config['start_point'] = st.session_state.start_point.copy()
    mission_config['boundary_points'] = [p.copy() for p in st.session_state.boundary_points]
    
    processed_detections = set()
    
    area_coordinates = get_area_coordinates()
    
    if area_coordinates:
        num_drones = st.session_state.get('num_drones', 3)
        capture_interval = st.session_state.get('capture_interval', 5)
        
        console_log(f"🚁 Starting mission with {num_drones} drones")
        console_log(f"🚁 Area coordinates: {area_coordinates}")
        console_log(f"🚁 Mission config stored for background threads")
        
        st.session_state.mission_cancel_event = threading.Event()
        
        st.session_state.detection_results = []
        st.session_state.detection_notifications = []
        
        st.session_state.mission_running = True
        st.session_state.mission_status = "Mission Started - Check Console"
        
        mission_thread = run_mission_with_debug(area_coordinates, num_drones, capture_interval)
        st.session_state.mission_thread = mission_thread
        
        st.success("Mission started! Detection markers will appear when objects are found.")
        
    else:
        st.error("Could not generate area coordinates!")

def stop_mission():
    """Stop the mission and signal drones to return home"""
    global active_camera_processes, active_detection_process, detection_thread_running, mission_config, processed_detections
    console_log("🛑 Mission cancellation requested...")
    
    detection_thread_running = False
    
    if st.session_state.mission_cancel_event:
        st.session_state.mission_cancel_event.set()
        console_log("🛑 Cancellation signal sent to drones")
    
    if active_camera_processes:
        stop_camera_capture(active_camera_processes)
    
    if active_detection_process:
        stop_detection_service(active_detection_process)
    
    try:
        files_to_clear = ["detection_notifications.json", "detection_markers.json", "map_refresh_trigger.txt"]
        for file in files_to_clear:
            if os.path.exists(file):
                os.remove(file)
    except:
        pass
    
    mission_config['start_point'] = None
    mission_config['boundary_points'] = []
    processed_detections = set()
    
    st.session_state.mission_status = "Cancelling Mission - Drones Returning Home..."
    st.session_state.camera_processes = []
    
    console_log("🛑 Mission cancellation initiated")

# Main UI
st.title("🚁 Drone Control System with Red Object Detection")
st.markdown("---")

# Detection alerts
notifications_file = "detection_notifications.json"
current_detection = None
if os.path.exists(notifications_file):
    try:
        with open(notifications_file, 'r') as f:
            file_notifications = json.load(f)
        
        if file_notifications:
            current_detection = file_notifications[-1]
            st.session_state.detection_notifications = [current_detection]
    except:
        pass

if current_detection:
    alert_col1, alert_col2, alert_col3 = st.columns([6, 2, 1])
    with alert_col1:
        st.error(f"🔴 {current_detection['details']} - {current_detection['filename']}")
    with alert_col2:
        timestamp = datetime.fromisoformat(current_detection['timestamp'])
        st.text(f"⏰ {timestamp.strftime('%H:%M:%S')}")
    with alert_col3:
        if st.button("✖", help="Clear Alert"):
            if os.path.exists(notifications_file):
                os.remove(notifications_file)
            st.rerun()

# Handle geolocation request
if st.session_state.request_location:
    location_html = """
    <script>
    function setLocationAndReload(lat, lng) {
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
        <div style="font-size: 18px;">📍 Getting your location...</div>
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
        
        st.session_state.start_point = {'lat': lat, 'lng': lng}
        st.session_state.map_center = [lat, lng] 
        st.session_state.map_zoom = 16
        st.session_state.request_location = False
        
        st.query_params.clear()
        
        st.success(f"📍 Start point set to your location: {lat:.6f}, {lng:.6f}")
        st.rerun()
except:
    pass

# Sidebar for controls
with st.sidebar:
    st.header("Mission Controls")
    
    mode = st.radio(
        "Select Mode:",
        ["Set Start Point", "Add Boundary Points", "View Only"],
        key="mode_radio"
    )
    st.session_state.mode = mode
    
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
    if st.button("🗑️ Clear Detection Markers", type="secondary"):
        if clear_detection_markers():
            st.success("All detection markers cleared!")
            st.rerun()
        else:
            st.error("Failed to clear markers")
    
    st.markdown("---")
    
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
    
    st.subheader("Mission Parameters")
    num_drones = st.number_input("Number of Drones", min_value=1, max_value=10, value=3, key="num_drones")
    grid_step = st.slider("Grid Step Size (m)", min_value=5, max_value=50, value=10)
    altitude = st.slider("Flight Altitude (m)", min_value=5, max_value=100, value=5)
    
    st.markdown("---")
    st.subheader("Camera & Detection Settings")
    capture_interval = st.slider("Image Capture Interval (s)", min_value=1, max_value=30, value=5, key="capture_interval")
    
    st.markdown("---")
    
    if st.session_state.start_point and len(st.session_state.boundary_points) >= 3:
        if not st.session_state.mission_running:
            if st.button("🚀 Start Mission", type="primary"):
                execute_mission_simulation()
                st.rerun()
        else:
            if st.button("🛑 Cancel Mission & Return Home", type="secondary"):
                stop_mission()
                st.rerun()
    else:
        st.info("Set start point and at least 3 boundary points to start mission")
    
    if hasattr(st.session_state, 'mission_thread') and st.session_state.mission_thread:
        if not st.session_state.mission_thread.is_alive() and st.session_state.mission_running:
            st.session_state.mission_running = False
            st.session_state.camera_processes = []
            st.session_state.mission_status = "Mission Completed"
            st.rerun()

    if st.session_state.mission_running:
        st.markdown("---")
        st.subheader("📊 System Status")
        
        if st.session_state.camera_processes:
            st.success(f"📹 {len(st.session_state.camera_processes)} cameras recording")
        
        if st.session_state.detection_results:
            total_detections = sum(1 for r in st.session_state.detection_results if r['has_red_objects'])
            st.info(f"🔍 Detection active: {total_detections} red objects found")
        else:
            st.info("🔍 Detection service running")
        
        st.text("Images saved to:")
        st.code("captured_images/[timestamp]")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Mission Area Map")
    
    if st.session_state.mode == "Set Start Point":
        st.info("📍 Click on the map to set the drone start point")
    elif st.session_state.mode == "Add Boundary Points":
        st.info("🎯 Click on the map to add boundary points")
    else:
        st.info("👀 View mode - map interactions disabled")
    
    # SIMPLE FIX: Check if new markers were added (trigger refresh)
    should_refresh_map = False
    trigger_file = "map_refresh_trigger.txt"
    current_marker_count = 0
    
    if os.path.exists(trigger_file):
        try:
            with open(trigger_file, 'r') as f:
                current_marker_count = int(f.read().strip())
        except:
            current_marker_count = 0
    
    if current_marker_count != st.session_state.last_marker_count:
        should_refresh_map = True
        st.session_state.last_marker_count = current_marker_count
        st.session_state.map_refresh_key += 1
        console_log(f"🔄 Map refresh triggered by new detection markers (count: {current_marker_count})")
    
    # IMPROVED: Load detection markers with better error handling
    markers_file = "detection_markers.json"
    current_markers = []
    if os.path.exists(markers_file):
        try:
            with open(markers_file, 'r') as f:
                current_markers = json.load(f)
            console_log(f"📍 Loaded {len(current_markers)} markers from file")
        except Exception as e:
            console_log(f"❌ Error loading detection markers: {e}")
            current_markers = []
    
    # Debug: Show marker details if debug mode
    if st.session_state.get('debug_markers', False):
        st.write("**Debug - Current markers:**")
        for i, marker in enumerate(current_markers):
            st.text(f"Marker {i+1}: {marker.get('lat', 'N/A'):.6f}, {marker.get('lng', 'N/A'):.6f}")
        if len(current_markers) > 0:
            st.success(f"🔴 {len(current_markers)} detection marker(s) on map")
    
    # Create map
    m = folium.Map(
        location=st.session_state.map_center,
        zoom_start=st.session_state.map_zoom,
        tiles='OpenStreetMap'
    )

    # Add start point
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

    # IMPROVED: Add detection markers with better validation and logging
    markers_added = 0
    for i, marker in enumerate(current_markers):
        try:
            lat = float(marker['lat'])
            lng = float(marker['lng'])
            
            if (-90 <= lat <= 90) and (-180 <= lng <= 180):
                timestamp = datetime.fromisoformat(marker['timestamp'])
                popup_text = f"""
                🔴 Red Object Detected<br>
                Drone: {marker['drone_number']}<br>
                Objects: {marker['object_count']}<br>
                Confidence: {marker['confidence']:.1f}%<br>
                Time: {timestamp.strftime('%H:%M:%S')}<br>
                File: {marker.get('filename', 'Unknown')}<br>
                Location: {lat:.6f}, {lng:.6f}
                """
                
                folium.Marker(
                    [lat, lng],
                    popup=popup_text,
                    tooltip=f"🔴 Drone {marker['drone_number']} Detection ({marker['confidence']:.1f}%)",
                    icon=folium.Icon(color='orange', icon='exclamation-sign')
                ).add_to(m)
                markers_added += 1
            else:
                console_log(f"⚠️ Invalid marker coordinates: {lat}, {lng}")
        except (ValueError, TypeError, KeyError) as e:
            console_log(f"❌ Error adding marker {i+1}: {e} - Data: {marker}")
    
    if markers_added > 0:
        console_log(f"📍 Successfully added {markers_added}/{len(current_markers)} markers to map")
    
    # Manual refresh button
    if st.button("🔄 Refresh Map Now", key="refresh_map_btn"):
        st.session_state.map_refresh_key += 1
        console_log("🔄 Manual map refresh triggered")
        st.rerun()
    
    # FIXED: Display map with NO center/zoom tracking to prevent resets
    map_data = st_folium(
        m,
        key=f"drone_map_{st.session_state.map_refresh_key}",
        width=700,
        height=500,
        returned_objects=["last_clicked", "center", "zoom"] # ADD center and zoom to preserve map state
    )

    # ONLY handle clicks for adding points - NO map view tracking
    if map_data.get('last_clicked') is not None and st.session_state.mode != "View Only":
        clicked_lat = map_data['last_clicked']['lat']
        clicked_lng = map_data['last_clicked']['lng']
        
        # PRESERVE current map position and zoom before rerun
        if map_data.get('center') and map_data.get('zoom'):
            st.session_state.map_center = [map_data['center']['lat'], map_data['center']['lng']]
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
    
    if st.session_state.mission_running:
        if "Cancelling" in st.session_state.mission_status:
            st.warning(f"⚠️ {st.session_state.mission_status}")
        else:
            st.success(f"🚁 Mission in Progress")
        st.write(f"Status: {st.session_state.mission_status}")
        st.info("Check your terminal/console for detailed progress updates!")
        
        if st.session_state.camera_processes:
            st.success(f"📹 {len(st.session_state.camera_processes)} cameras recording")
        
        if st.session_state.detection_results:
            red_count = sum(1 for r in st.session_state.detection_results if r['has_red_objects'])
            st.info(f"🔍 Detection active: {red_count}/{len(st.session_state.detection_results)} images with red objects")
        
    else:
        if "Failed" in st.session_state.mission_status:
            st.error(f"❌ {st.session_state.mission_status}")
        elif "Completed" in st.session_state.mission_status or "Cancelled" in st.session_state.mission_status:
            st.success(f"✅ {st.session_state.mission_status}")
        else:
            st.success("✅ Ready for Mission")
    
    if st.session_state.detection_results:
        st.subheader("🔍 Recent Detections")
        
        recent_results = st.session_state.detection_results[-5:]
        for result in reversed(recent_results):
            timestamp = datetime.fromisoformat(result['timestamp'])
            
            if result['has_red_objects']:
                st.error(f"🔴 {result['filename']}")
                st.text(f"Time: {timestamp.strftime('%H:%M:%S')}")
                st.text(f"Confidence: {result['confidence']:.1f}%")
                st.text(f"Objects: {result['object_count']}")
            else:
                st.success(f"⚪ {result['filename']}")
                st.text(f"Time: {timestamp.strftime('%H:%M:%S')}")
    
    if st.session_state.start_point and len(st.session_state.boundary_points) >= 3:
        st.subheader("Mission Preview")
        
        ref_lat = st.session_state.start_point['lat']
        ref_lon = st.session_state.start_point['lng']
        
        ned_boundary = []
        for point in st.session_state.boundary_points:
            north, east = geo_to_ned(point['lat'], point['lng'], ref_lat, ref_lon)
            ned_boundary.append((north, east))
        
        try:
            grid_points = generate_grid_points(ned_boundary, step=grid_step)
            
            if grid_points:
                points_np = np.array(grid_points)
                labels, centers = kmeans_clustering(points_np, num_drones)
                
                drone_routes = assign_drone_routes(points_np, labels, num_drones)
                drone_routes_nn = []
                for route in drone_routes:
                    drone_routes_nn.append(nearest_neighbor(route))
                
                st.success(f"✅ Mission Plan Ready")
                
                visualize_multiple_drone_paths_and_area(drone_routes_nn, centers, get_area_coordinates())
                
        except Exception as e:
            st.error(f"Preview error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("🚁 **Drone Control System** | 📹 **Camera Capture** | 🔍 **Red Object Detection**")

# SIMPLE FIX: Only refresh when new markers are detected
if should_refresh_map:
    time.sleep(1)  # Brief pause before refresh
    st.rerun()

# Optional debug info
debug_mode = st.checkbox("Show Debug Info")
if debug_mode:
    st.session_state.debug_markers = st.checkbox("Debug Detection Markers", help="Show detailed marker information")
    
    if st.session_state.start_point and st.session_state.boundary_points:
        area_coords = get_area_coordinates()
        if area_coords:
            st.write("**Area coordinates that will be sent to control_drones:**")
            st.code(f"area_coordinates = {area_coords}")
        else:
            st.write("No valid area coordinates")
    
    st.write("**Current map state:**")
    st.write(f"Center: {st.session_state.map_center}")
    st.write(f"Zoom: {st.session_state.map_zoom}")
    st.write(f"Marker count: {current_marker_count}")
    
    # Debug marker file contents
    if os.path.exists("detection_markers.json"):
        try:
            with open("detection_markers.json", 'r') as f:
                marker_data = json.load(f)
            st.write(f"**Markers in file: {len(marker_data)}**")
            if len(marker_data) > 0:
                st.json(marker_data[-3:])  # Show last 3 markers
        except Exception as e:
            st.error(f"Error reading marker file: {e}")
    
    if st.session_state.detection_results:
        st.write("**Detection results:**")
        st.json(st.session_state.detection_results[-3:])

# Additional functions for drone control with cancellation
async def return_drones_to_home(drones, actual_home_positions):
    """Return all drones to their actual takeoff positions"""
    print("🏠 Returning drones to actual home positions...")
    
    home_tasks = []
    for i, (drone, home_pos) in enumerate(zip(drones, actual_home_positions)):
        home_position = PositionNedYaw(
            home_pos[0],
            home_pos[1],
            2.0,  # FIXED: Go to ground level (positive = below takeoff point)
            0.0
        )
        print(f"🏠 Sending Drone {i+1} to ground level: N={home_pos[0]:.2f}, E={home_pos[1]:.2f}, D=2.0m")
        home_tasks.append(drone.offboard.set_position_ned(home_position))
    
    await asyncio.gather(*home_tasks)
    
    print("🏠 Waiting for drones to reach ground level...")
    await asyncio.sleep(25)  # More time to descend to ground
    
    print("✅ All drones at ground level")

async def control_drones_with_cancellation(area, cancel_event):
    """Modified control_drones function that supports cancellation"""
    
    drones_info = [
        ("localhost", 50060),
        ("localhost", 50061),
        ("localhost", 50062)
    ]

    console_log("🔗 Connecting to drones...")
    drones = [await connect_drone(addr, port) for addr, port in drones_info]

    drone_offsets = [
        (0.0, 3.0),
        (0.0, 6.0),
        (0.0, 9.0)
    ]

    actual_home_positions = []

    try:
        console_log("🚁 Arming and taking off...")
        await asyncio.gather(*(arm_and_takeoff(drone) for drone in drones))
        
        print("📍 Recording actual takeoff positions...")
        for i, drone in enumerate(drones):
            async for position in drone.telemetry.position_velocity_ned():
                actual_pos = (
                    position.position.north_m + drone_offsets[i][0],
                    position.position.east_m + drone_offsets[i][1]
                )
                actual_home_positions.append(actual_pos)
                print(f"🏠 Drone {i+1} home position: N={actual_pos[0]:.2f}, E={actual_pos[1]:.2f}")
                break

        if cancel_event.is_set():
            console_log("🛑 Mission cancelled during takeoff")
            raise asyncio.CancelledError("Mission cancelled during takeoff")
        
        await asyncio.gather(*(activate_offboard(drone) for drone in drones))

        console_log("📍 Generating mission waypoints...")
        points = generate_grid_points(area, step=10.0)
        n_clusters = len(drones)
        points_np = np.array(points)
        labels, centers = kmeans_clustering(points_np, n_clusters)
        drone_routes = assign_drone_routes(points, labels, len(drones))
        
        drone_routes_nn = []
        for route in drone_routes:
            drone_routes_nn.append(nearest_neighbor(route))

        console_log("🚁 Starting mission execution...")
        await fly_area_with_cancellation(drones, drone_routes_nn, drone_offsets, cancel_event)

    except asyncio.CancelledError:
        print("🛑 Mission cancelled - initiating return to home...")
        
        try:
            if actual_home_positions:
                await return_drones_to_home(drones, actual_home_positions)
            else:
                print("⚠️ No recorded home positions, using manual descent")
                # Manual descent to ground level
                for i, drone in enumerate(drones):
                    try:
                        ground_position = PositionNedYaw(
                            drone_offsets[i][0],
                            drone_offsets[i][1],
                            1.0,  # Ground level
                            0.0
                        )
                        await drone.offboard.set_position_ned(ground_position)
                    except Exception as e:
                        print(f"❌ Error descending Drone {i+1}: {e}")
                await asyncio.sleep(20)
        except Exception as e:
            print(f"❌ Error returning drones home: {e}")
        
        print("🛬 Landing drones...")
        
        # FIXED: First descend to ground level in offboard mode
        print("📉 Descending to ground level...")
        for i, drone in enumerate(drones):
            try:
                ground_position = PositionNedYaw(
                    actual_home_positions[i][0] if actual_home_positions else drone_offsets[i][0],
                    actual_home_positions[i][1] if actual_home_positions else drone_offsets[i][1],
                    1.0,  # Just above ground level
                    0.0
                )
                await drone.offboard.set_position_ned(ground_position)
            except Exception as e:
                print(f"❌ Error descending Drone {i+1}: {e}")
        
        await asyncio.sleep(15)  # Wait for descent
        
        # Then stop offboard and land
        for i, drone in enumerate(drones):
            try:
                print(f"🛬 Landing Drone {i+1}...")
                # Stop offboard mode first
                await stop_offboard_mode(drone)
                await asyncio.sleep(1)
                # Then initiate landing
                await drone.action.land()
                await asyncio.sleep(3)
            except Exception as e:
                print(f"❌ Error landing Drone {i+1}: {e}")

        print("🛬 Waiting for drones to land completely...")
        await asyncio.sleep(20)  # More time for complete landing

        print("🔓 Ensuring drones are disarmed...")
        for i, drone in enumerate(drones):
            try:
                # Check if drone is still armed and disarm if needed
                async for is_armed in drone.telemetry.armed():
                    if is_armed:
                        print(f"🔓 Disarming Drone {i+1}...")
                        await drone.action.disarm()
                        await asyncio.sleep(2)
                    else:
                        print(f"✅ Drone {i+1} already disarmed")
                    break  # Only check once
            except Exception as e:
                print(f"❌ Error disarming Drone {i+1}: {e}")
        
        console_log("✅ Mission cancelled successfully - all drones landed and disarmed")
        raise
        
    except Exception as e:
        console_log(f"❌ Unexpected error during mission: {e}")
        raise
    
    else:
        console_log("✅ Mission completed normally - landing drones...")
        
        # FIXED: First descend to near ground level, then land
        console_log("📉 Descending drones to ground level...")
        descend_tasks = []
        for i, drone in enumerate(drones):
            ground_position = PositionNedYaw(
                actual_home_positions[i][0] if actual_home_positions else drone_offsets[i][0],
                actual_home_positions[i][1] if actual_home_positions else drone_offsets[i][1],
                1.0,  # Just above ground
                0.0
            )
            descend_tasks.append(drone.offboard.set_position_ned(ground_position))
        
        await asyncio.gather(*descend_tasks)
        await asyncio.sleep(15)  # Wait for descent
        
        # Stop offboard mode first, then land
        for i, drone in enumerate(drones):
            try:
                console_log(f"🔌 Stopping offboard mode for Drone {i+1}...")
                await stop_offboard_mode(drone)
                await asyncio.sleep(1)
            except Exception as e:
                console_log(f"❌ Error stopping offboard for Drone {i+1}: {e}")
        
        # Land all drones
        for i, drone in enumerate(drones):
            try:
                console_log(f"🛬 Landing Drone {i+1}...")
                await drone.action.land()
                await asyncio.sleep(3)
            except Exception as e:
                console_log(f"❌ Error landing Drone {i+1}: {e}")
                
        console_log("🛬 Waiting for complete landing...")
        await asyncio.sleep(20)  # More time for landing
        
        # Ensure all drones are disarmed
        for i, drone in enumerate(drones):
            try:
                async for is_armed in drone.telemetry.armed():
                    if is_armed:
                        console_log(f"🔓 Disarming Drone {i+1}...")
                        await drone.action.disarm()
                        await asyncio.sleep(2)
                    else:
                        console_log(f"✅ Drone {i+1} already disarmed")
                    break
            except Exception as e:
                console_log(f"❌ Error checking/disarming Drone {i+1}: {e}")
                
        console_log("✅ All drones landed and disarmed successfully")

async def fly_area_with_cancellation(drones, drone_routes_nn, drone_offsets, cancel_event):
    """Execute drone routes with cancellation support"""
    console_log(f"🚁 Starting flight execution for {len(drones)} drones...")
    
    flight_tasks = []
    for i in range(len(drones)):
        task = fly_route_with_cancellation(
            drones[i], 
            drone_routes_nn[i], 
            drone_offsets[i], 
            cancel_event,
            i + 1
        )
        flight_tasks.append(task)
    
    await asyncio.gather(*flight_tasks)

async def fly_route_with_cancellation(drone, route, offset, cancel_event, drone_number):
    """Fly a single drone route with cancellation checking"""
    console_log(f"🚁 Drone {drone_number}: Starting route with {len(route)} waypoints")
    
    for i, point in enumerate(route):
        if cancel_event.is_set():
            console_log(f"🛑 Drone {drone_number}: Mission cancelled at waypoint {i+1}/{len(route)}")
            raise asyncio.CancelledError(f"Drone {drone_number} mission cancelled")
        
        shifted = PositionNedYaw(
            point[0] + offset[0],
            point[1] + offset[1],
            -30.0,  # Back to reasonable flight altitude
            0.0
        )
        
        console_log(f"🚁 Drone {drone_number}: Going to waypoint {i+1}/{len(route)} - N:{shifted.north_m:.1f}, E:{shifted.east_m:.1f}")
        
        await drone.offboard.set_position_ned(shifted)
        
        for _ in range(7):
            if cancel_event.is_set():
                console_log(f"🛑 Drone {drone_number}: Mission cancelled while at waypoint {i+1}")
                raise asyncio.CancelledError(f"Drone {drone_number} mission cancelled")
            await asyncio.sleep(1)
    
    console_log(f"✅ Drone {drone_number}: Completed route successfully")