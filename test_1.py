import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from robomaster.robot import Robot
from skimage.color import rgb2hsv
from skimage.measure import label, regionprops

# --- Global Sensor Variables ---
accumulated_yaw = 0.0
last_yaw = None
current_distance = 2000.0  # Initialized at 2 meters
readings = []  # For plot: (angle, distance)

# --- Control Parameters (Anti-drift) ---
KP_YAW = 1.5  # Aggressiveness of heading correction

# --- Memory Variable for Unknown Obstacle ---
last_turn_was_right = False 

# --- SENSOR CALLBACKS (Gyroscope / IMU) ---
def attitude_handler(attitude_info):
    global accumulated_yaw, last_yaw
    yaw = attitude_info[0]
    
    if last_yaw is None:
        last_yaw = yaw
        return
        
    delta = yaw - last_yaw
    
    # Handle wrap-around between 180 and -180
    if delta > 180: delta -= 360
    elif delta < -180: delta += 360
    
    # accumulated_yaw stores absolute heading from start
    accumulated_yaw += delta
    last_yaw = yaw

def distance_handler(data):
    global current_distance, readings, accumulated_yaw
    dist = data[0]
    current_distance = dist if dist != 65534 else 2000.0
    readings.append((abs(accumulated_yaw), current_distance))

# --- VISIONE ---
def detect_color(img_rgb: np.ndarray):
    img_float = img_rgb.astype(np.float32) / 255.0
    hsv = rgb2hsv(img_float)
    
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    
    mask_orange = (H >= 0.02) & (H <= 0.15) & (S > 0.50) & (V > 0.20)
    mask_blue   = (H >= 0.55) & (H <= 0.70) & (S > 0.40) & (V > 0.20)
    mask_black  = (V < 0.20)
    
    MIN_PIXELS = 10000 
    
    if np.sum(mask_blue) > MIN_PIXELS: return "blue"
    if np.sum(mask_black) > MIN_PIXELS: return "black"
    if np.sum(mask_orange) > MIN_PIXELS: return "orange"
    
    return None

def guess_unknown_color(img_rgb: np.ndarray) -> str:
    h, w = img_rgb.shape[:2]
    center_roi = img_rgb[h//2 - 20 : h//2 + 20, w//2 - 20 : w//2 + 20]
    avg_color = np.mean(center_roi, axis=(0, 1))
    R, G, B = avg_color
    
    if R > G and R > B: return "Red/Brown"
    if G > R and G > B: return "Green"
    if B > R and B > G: return "Blue/Purple"
    if R > 200 and G > 200 and B > 200: return "White"
    return "Unknown"

# --- MANEUVERS AND OVERLAY ---
def draw_overlay(frame, color_seen, state_msg, z_correction=0.0):
    display = frame.copy()
    cv2.putText(display, f"State: {state_msg}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(display, f"Color Seen: {color_seen}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(display, f"ToF Distance: {current_distance:.0f} mm", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Show gyro error and applied correction on screen
    cv2.putText(display, f"Gyro Err: {accumulated_yaw:+.1f} deg", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2)
    cv2.putText(display, f"Z Correction: {z_correction:+.1f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2)
    return display

def maneuver_and_pump_camera(robot, x_speed, y_speed, duration):
    global accumulated_yaw
    start_time = time.time()
    
    while time.time() - start_time < duration:
        # Compute Z correction in real time (if drifting left, push right)
        z_speed = -KP_YAW * accumulated_yaw
        z_speed = float(np.clip(z_speed, -30.0, 30.0)) # Limit to avoid aggressive turning
        
        robot.chassis.drive_speed(x=x_speed, y=y_speed, z=z_speed)
        
        frame = robot.camera.read_video_frame(strategy="newest")
        if frame is not None:
            display = frame.copy()
            cv2.putText(display, "DODGE IN PROGRESS...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            # Show correction during dodge
            cv2.putText(display, f"Z Corr: {z_speed:.1f}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow("RoboMaster - Live", display)
            cv2.waitKey(1)
        time.sleep(0.05)
        
    robot.chassis.drive_speed(x=0, y=0, z=0)

def dodge_obstacle(robot, direction_y):
    # Dodge sideways (x=0, y=direction)
    maneuver_and_pump_camera(robot, x_speed=0.0, y_speed=direction_y, duration=1.5)
    # Move forward (x=0.3, y=0)
    maneuver_and_pump_camera(robot, x_speed=0.3, y_speed=0.0, duration=2.0)
    # Return to center (x=0, y=-direction)
    maneuver_and_pump_camera(robot, x_speed=0.0, y_speed=-direction_y, duration=1.5)

# ==========================================
# --- MAIN EXECUTION ---
# ==========================================
robot = Robot()
robot.initialize(conn_type="sta", sn="LOCAL")

try:
    print("Initializing Sensors and Camera...")
    # Subscribe attitude at 50Hz (faster readings = faster correction)
    robot.chassis.sub_attitude(freq=50, callback=attitude_handler)
    robot.sensor.sub_distance(freq=10, callback=distance_handler)
    robot.camera.start_video_stream(display=False, resolution="720p")
    time.sleep(1.5)
    
    print("\nRobot ready! Press Q in the video window to stop and generate the plot.\n")

    while True:
        frame = robot.camera.read_video_frame(strategy="newest")
        if frame is None:
            time.sleep(0.01)
            continue

        rgb_frame = frame[:, :, ::-1]
        color_seen = detect_color(rgb_frame)
        z_correction = 0.0 # Valore di default per l'overlay
        
        # --- NAVIGATION AND DODGE LOGIC ---
        
        # 1. BLUE CASE (Approach to 6cm / 60mm and STOP)
        if color_seen == "blue":
            if current_distance > 60:
                state_msg = "BLUE seen: approaching..."
                z_correction = -KP_YAW * accumulated_yaw
                z_correction = float(np.clip(z_correction, -15.0, 15.0))
                robot.chassis.drive_speed(x=0.2, y=0.0, z=z_correction)
            else:
                state_msg = "BLUE target reached (6cm)!"
                robot.chassis.drive_speed(x=0.0, y=0.0, z=0.0)
                
        # 2. DODGE: Sensor detects obstacle at 10 cm (100 mm) or closer
        elif current_distance <= 100:
            if color_seen == "black":
                state_msg = "BLACK obstacle at 10cm: Dodge RIGHT"
                print(f"\n[{current_distance}mm] Detected BLACK. Dodging right.")
                dodge_obstacle(robot, direction_y=0.3)
                last_turn_was_right = True
                
            elif color_seen == "orange":
                state_msg = "ORANGE obstacle at 10cm: Dodge LEFT"
                print(f"\n[{current_distance}mm] Detected ORANGE. Dodging left.")
                dodge_obstacle(robot, direction_y=-0.3)
                last_turn_was_right = False
                
            else:
                # Sees an obstacle pole but unknown color
                guessed_color = guess_unknown_color(rgb_frame)
                state_msg = f"Obstacle ({guessed_color}) at 10cm!"
                
                if last_turn_was_right:
                    print(f"\n[{current_distance}mm] Obstacle {guessed_color}. Last turn: Right -> Now dodge LEFT.")
                    dodge_obstacle(robot, direction_y=-0.3)
                    last_turn_was_right = False
                else:
                    print(f"\n[{current_distance}mm] Obstacle {guessed_color}. Last turn: Left -> Now dodge RIGHT.")
                    dodge_obstacle(robot, direction_y=0.3)
                    last_turn_was_right = True

        # 3. NO CLOSE OBSTACLE (> 100 mm)
        else:
            if color_seen in ["black", "orange"]:
                state_msg = f"Seeing {color_seen.upper()} ahead..."
            else:
                state_msg = "Steady straight progress"
            
            # --- Continuous auto-correction on main path ---
            z_correction = -KP_YAW * accumulated_yaw
            z_correction = float(np.clip(z_correction, -15.0, 15.0)) # gentler when straight
            
            robot.chassis.drive_speed(x=0.3, y=0.0, z=z_correction)

        # --- Aggiorna Finestra Video ---
        display_frame = draw_overlay(frame, str(color_seen), state_msg, z_correction)
        cv2.imshow("RoboMaster - Live", display_frame)

        # Exit on Q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nShutdown requested by user.")
            break

        time.sleep(0.05)

except KeyboardInterrupt:
    print("\nManual interruption (Ctrl+C).")
finally:
    print("\nShutting down motors and sensors...")
    robot.chassis.drive_speed(x=0, y=0, z=0)
    robot.sensor.unsub_distance()
    robot.chassis.unsub_attitude()
    cv2.destroyAllWindows()
    try:
        robot.camera.stop_video_stream()
        time.sleep(0.5)
        robot.close()
    except:
        pass

# ==========================================
# --- FINAL PLOT GENERATION ---
# ==========================================
print("\nGenerating sensor plot...")
if readings:
    angles = [r[0] for r in readings]
    distances = [r[1] for r in readings]
    distances_clean = [d if d != 2000.0 else np.nan for d in distances]
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(distances_clean)), distances_clean, '-o', markersize=4, color="b", label="Obstacle distance")
    plt.title("Distance profile recorded during the run")
    plt.xlabel("Time (sensor readings)")
    plt.ylabel("Distance (mm)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    print("No sensor data acquired for the plot.")