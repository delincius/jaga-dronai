#!/usr/bin/env python

import cv2
import gi
import numpy as np
import time
import os
from datetime import datetime
import argparse
import signal
import sys

gi.require_version('Gst', '1.0')
from gi.repository import Gst


class Video():
    """BlueRov video capture class constructor

    Attributes:
        port (int): Video UDP port
        video_codec (string): Source h264 parser
        video_decode (string): Transform YUV (12bits) to BGR (24bits)
        video_pipe (object): GStreamer top-level pipeline
        video_sink (object): Gstreamer sink element
        video_sink_conf (string): Sink configuration
        video_source (string): Udp source ip and port
    """

    def __init__(self, port=5600):
        """Summary

        Args:
            port (int, optional): UDP port
        """

        Gst.init(None)

        self.port = port
        self._frame = None

        # [Software component diagram](https://www.ardusub.com/software/components.html)
        # UDP video stream (:5600)
        self.video_source = 'udpsrc port={}'.format(self.port)
        # [Rasp raw image](http://picamera.readthedocs.io/en/release-0.7/recipes2.html#raw-image-capture-yuv-format)
        # Cam -> CSI-2 -> H264 Raw (YUV 4-4-4 (12bits) I420)
        self.video_codec = '! application/x-rtp, payload=96 ! rtph264depay ! h264parse ! avdec_h264'
        # Python don't have nibble, convert YUV nibbles (4-4-4) to OpenCV standard BGR bytes (8-8-8)
        self.video_decode = \
            '! decodebin ! videoconvert ! video/x-raw,format=(string)BGR ! videoconvert'
        # Create a sink to get data
        self.video_sink_conf = \
            '! appsink emit-signals=true sync=false max-buffers=2 drop=true'

        self.video_pipe = None
        self.video_sink = None

        self.run()

    def start_gst(self, config=None):
        """ Start gstreamer pipeline and sink
        Pipeline description list e.g:
            [
                'videotestsrc ! decodebin', \
                '! videoconvert ! video/x-raw,format=(string)BGR ! videoconvert',
                '! appsink'
            ]

        Args:
            config (list, optional): Gstreamer pileline description list
        """

        if not config:
            config = \
                [
                    'videotestsrc ! decodebin',
                    '! videoconvert ! video/x-raw,format=(string)BGR ! videoconvert',
                    '! appsink'
                ]

        command = ' '.join(config)
        self.video_pipe = Gst.parse_launch(command)
        self.video_pipe.set_state(Gst.State.PLAYING)
        self.video_sink = self.video_pipe.get_by_name('appsink0')

    @staticmethod
    def gst_to_opencv(sample):
        """Transform byte array into np array

        Args:
            sample (TYPE): Description

        Returns:
            TYPE: Description
        """
        buf = sample.get_buffer()
        caps = sample.get_caps()
        array = np.ndarray(
            (
                caps.get_structure(0).get_value('height'),
                caps.get_structure(0).get_value('width'),
                3
            ),
            buffer=buf.extract_dup(0, buf.get_size()), dtype=np.uint8)
        return array

    def frame(self):
        """ Get Frame

        Returns:
            iterable: bool and image frame, cap.read() output
        """
        return self._frame

    def frame_available(self):
        """Check if frame is available

        Returns:
            bool: true if frame is available
        """
        return type(self._frame) != type(None)

    def run(self):
        """ Get frame to update _frame
        """

        self.start_gst(
            [
                self.video_source,
                self.video_codec,
                self.video_decode,
                self.video_sink_conf
            ])

        self.video_sink.connect('new-sample', self.callback)

    def callback(self, sink):
        sample = sink.emit('pull-sample')
        new_frame = self.gst_to_opencv(sample)
        self._frame = new_frame

        return Gst.FlowReturn.OK


def save_image(frame, output_dir, drone_number, image_count):
    """Save the current frame as an image file

    Args:
        frame: The frame to save
        output_dir (str): Directory to save images
        drone_number (int): Drone identifier
        image_count (int): Image counter
    """
    # Create drone-specific subdirectory
    drone_dir = os.path.join(output_dir, f"drone_{drone_number}")
    if not os.path.exists(drone_dir):
        os.makedirs(drone_dir)
    
    # Generate filename with timestamp and counter
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(drone_dir, f"drone{drone_number}_img{image_count:04d}_{timestamp}.jpg")
    
    # Save the image
    try:
        cv2.imwrite(filename, frame)
        print(f"[Drone {drone_number}] Saved image #{image_count}: {filename}")
        return True
    except Exception as e:
        print(f"[Drone {drone_number}] Error saving image: {e}")
        return False


# Global flag for graceful shutdown
running = True

def signal_handler(sig, frame):
    """Handle termination signals gracefully"""
    global running
    print(f"\n[Signal {sig}] Shutting down camera capture...")
    running = False


if __name__ == '__main__':
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    parser = argparse.ArgumentParser(description='Drone camera capture script')
    parser.add_argument('--port', '-p', type=int, required=True, 
                       help='Port where the UDP stream is coming from')
    parser.add_argument('--number', '-n', type=int, required=True, 
                       help='Drone number (1, 2, 3, ...)')
    parser.add_argument('--interval', '-i', type=int, default=5,
                       help='Image capture interval in seconds (default: 5)')
    
    args = parser.parse_args()
    
    # Get output directory from environment variable or use default
    output_directory = os.environ.get('OUTPUT_DIR', 'captured_images')
    
    print(f"[Drone {args.number}] Starting camera capture")
    print(f"[Drone {args.number}] Port: {args.port}")
    print(f"[Drone {args.number}] Output directory: {output_directory}")
    print(f"[Drone {args.number}] Capture interval: {args.interval} seconds")
    
    try:
        # Create the video object
        video = Video(port=args.port)
        
        # Initialize counters and timers
        last_capture_time = time.time()
        image_count = 0
        capture_interval = args.interval
        
        print(f"[Drone {args.number}] Camera initialized, starting capture loop...")
        
        while running:
            # Wait for the next frame
            if not video.frame_available():
                time.sleep(0.1)  # Small delay to prevent busy waiting
                continue

            frame = video.frame()
            
            # Check if it's time to save an image
            current_time = time.time()
            if current_time - last_capture_time >= capture_interval:
                image_count += 1
                if save_image(frame, output_directory, args.number, image_count):
                    last_capture_time = current_time
                
            # Small delay to prevent excessive CPU usage
            time.sleep(0.05)
            
    except Exception as e:
        print(f"[Drone {args.number}] Fatal error: {e}")
        sys.exit(1)
    
    print(f"[Drone {args.number}] Camera capture stopped")
    print(f"[Drone {args.number}] Total images captured: {image_count}")