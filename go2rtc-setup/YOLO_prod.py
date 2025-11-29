import os
import sys
import errno
import time
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

# --- CONFIG ---
WIDTH, HEIGHT = 640, 480
FPS = 15
PIPE_RAW = '/tmp/pipe_raw'  # Raw video
PIPE_YOLO = '/tmp/pipe_yolo' # Annotated video

# --- LOGGING SETUP ---
# We use stderr for logs so they show up in systemctl status
def log(msg):
    sys.stderr.write(f"{msg}\n")
    sys.stderr.flush()

log("Initializing YOLO model...")
model = YOLO("/home/robo_coyote/Documents/YOLO_test/best.pt")
model.overrides['verbose'] = False 

def ensure_pipe(path):
    """Creates a named pipe if it doesn't exist."""
    if not os.path.exists(path):
        try:
            os.mkfifo(path)
            # Set permissions so go2rtc can read it even if users differ
            os.chmod(path, 0o666) 
            log(f"Created pipe: {path}")
        except OSError as e:
            log(f"Failed to create pipe {path}: {e}")


def write_to_pipe(fd, data):
    """
    Writes data to a pipe. 
    STRATEGY: 
    1. If pipe is full at start: Drop frame (prevent lag).
    2. If write starts: Force finish (prevent corruption).
    """
    if fd is None: 
        return None
    
    data_bytes = data if isinstance(data, bytes) else data.tobytes()
    total_len = len(data_bytes)
    written = 0
    mv = memoryview(data_bytes)

    try:
        while written < total_len:
            try:
                # write returns number of bytes actually written
                n = os.write(fd, mv[written:])
                written += n
            except OSError as e:
                # EAGAIN means "Pipe Full"
                if e.errno == errno.EAGAIN:
                    # CASE A: We haven't written anything yet.
                    if written == 0:
                        # Drop the frame entirely. Better to skip 1 frame 
                        # than corrupt the video stream.
                        return fd 
                    
                    # CASE B: We partially wrote the frame.
                    # We are committed. We MUST finish writing, or the stream
                    # becomes byte-misaligned. We sleep tiny amount and retry.
                    time.sleep(0.001)
                    continue
                
                elif e.errno == errno.EPIPE:
                    # Reader (FFmpeg) closed the connection
                    os.close(fd)
                    return None
                else:
                    # Some other crash
                    try: os.close(fd)
                    except: pass
                    return None
                    
        return fd
        
    except Exception as e:
        log(f"Pipe Error: {e}")
        return None

def open_pipe_nb(path):
    """Attempts to open a pipe non-blocking. Returns fd or None."""
    try:
        fd = os.open(path, os.O_WRONLY | os.O_NONBLOCK)
        return fd
    except OSError as e:
        if e.errno == errno.ENXIO: # No reader connected
            return None
        return None

def run_pipeline():
    ensure_pipe(PIPE_RAW)
    ensure_pipe(PIPE_YOLO)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    
    try:
        pipeline.start(config)
    except Exception as e:
        log(f"Failed to start RealSense: {e}")
        return

    log("Stream started. Filling pipes...")

    fd_raw = None
    fd_yolo = None

    try:
        while True:
            # 1. Get Frame
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame: continue
            frame = np.asanyarray(color_frame.get_data())
            
            # 2. Manage Connections (Lazy Open)
            if fd_raw is None: fd_raw = open_pipe_nb(PIPE_RAW)
            if fd_yolo is None: fd_yolo = open_pipe_nb(PIPE_YOLO)

            # 3. Write RAW
            if fd_raw is not None:
                fd_raw = write_to_pipe(fd_raw, frame.tobytes())

            # 4. Process YOLO & Write
            # Only process YOLO if someone is actually watching the YOLO stream
            # to save CPU, OR process always if you want consistent logs.
            # Here we process always:
            results = model(frame, conf=0.25, verbose=False)
            annotated = results[0].plot()

            if fd_yolo is not None:
                fd_yolo = write_to_pipe(fd_yolo, annotated.tobytes())

    finally:
        pipeline.stop()
        # Do NOT remove pipes. Let them persist.
        if fd_raw: os.close(fd_raw)
        if fd_yolo: os.close(fd_yolo)
        log("Pipeline stopped.")

if __name__ == "__main__":
    run_pipeline()