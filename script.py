import time
import threading
import numpy as np
import cv2
import matplotlib.pyplot as plt
from robomaster import robot as rm_robot
from robomaster.robot import Robot


# CONSTANTS AND PARAMETERS 
KP_YAW             = 2.0
MAX_Z              = 90.0

OBSTACLE_DIST_MM   = 300
BLUE_STOP_MM       = 60
CLEAR_THRESHOLD_MM = 1000

DODGE_Y_SPEED      = 0.40
FORWARD_SPEED      = 0.30
MAX_FWD_TIME       = 10.0

MIN_STRAFE_CM      = 8.0
MIN_ALONGSIDE_CM   = 10.0
EXTRA_PASS_CM      = 45.0
RETURN_FACTOR      = 1.08

MIN_PIXELS         = 10000

COLOR_RANGES = {
    "blue":   ((  99, 102,  51), (126, 255, 255)),
    "black":  ((   0,   0,   0), (179, 255,  50)),
    "orange": ((   3, 127,  51), ( 27, 255, 255)),
    "red":    ((   0, 120,  70), ( 10, 255, 255)),  
    "red2":   (( 170, 120,  70), (180, 255, 255)),  
    "green":  ((  36,  80,  40), ( 86, 255, 255)),
    "white":  ((   0,   0, 200), (179,  30, 255)),
    "yellow": ((  20, 100,  80), ( 35, 255, 255)),
}


# GLOBAL THREAD-SAFE STATE 
_lock               = threading.Lock()
accumulated_yaw     = 0.0
target_yaw          = 0.0
last_yaw            = None
current_distance    = 2000.0
readings            = []
last_turn_was_right = False


# CALLBACKS SENSORS 
def attitude_handler(attitude_info):
    global accumulated_yaw, last_yaw
    yaw = attitude_info[0]
    if last_yaw is None:
        last_yaw = yaw
        return
    delta = yaw - last_yaw
    if delta >  180: delta -= 360
    if delta < -180: delta += 360
    with _lock:
        accumulated_yaw += delta
    last_yaw = yaw


def distance_handler(data):
    global current_distance, readings
    dist = data[0]
    with _lock:
        current_distance = dist if dist != 65534 else 2000.0
        readings.append((abs(accumulated_yaw), current_distance))


# HELPER YAW 
def get_z_correction(max_z: float = MAX_Z) -> float:
    with _lock:
        error = accumulated_yaw - target_yaw
    return float(np.clip(-KP_YAW * error, -max_z, max_z))


def reset_yaw_reference():
    global accumulated_yaw, last_yaw, target_yaw
    with _lock:
        accumulated_yaw = 0.0
        last_yaw        = None
        target_yaw      = 0.0


# LIVE VISION
# def detect_color(img_rgb: np.ndarray):
#     roi = img_rgb[ROI_Y:ROI_Y2, ROI_X:ROI_X2]          
#     hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
#     mask_blue   = cv2.inRange(hsv, ( 99, 102,  51), (126, 255, 255))
#     mask_black  = cv2.inRange(hsv, (  0,   0,   0), (179, 255,  50))
#     mask_orange = cv2.inRange(hsv, (  3, 127,  51), ( 27, 255, 255))
#     counts = {
#         "blue":   cv2.countNonZero(mask_blue),
#         "black":  cv2.countNonZero(mask_black),
#         "orange": cv2.countNonZero(mask_orange),
#     }
#     dominant = max(counts, key=counts.get)
#     return dominant if counts[dominant] > MIN_PIXELS else None

def detect_color(img_rgb: np.ndarray):
    roi = img_rgb[ROI_Y:ROI_Y2, ROI_X:ROI_X2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)

    counts = {}
    for name, (lower, upper) in COLOR_RANGES.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        counts[name] = cv2.countNonZero(mask)

    # Merge 'red' and 'red2' into a single 'red'
    counts["red"] = counts.pop("red", 0) + counts.pop("red2", 0)

    # All colours detected above the threshold
    detected = {k: v for k, v in counts.items() if v > MIN_PIXELS}

    if not detected:
        return None, {}

    dominant = max(detected, key=detected.get)
    return dominant, detected



# ROI (Region of Interest) for colour detection
ROI_SIZE   = 250
ROI_OFFSET_Y = 80   
ROI_X      = (1280 - ROI_SIZE) // 2  
ROI_Y      = ( 720 - ROI_SIZE) // 2 - ROI_OFFSET_Y
ROI_X2     = ROI_X + ROI_SIZE          
ROI_Y2     = ROI_Y + ROI_SIZE          


def guess_unknown_color(img_rgb: np.ndarray) -> str:
    roi      = img_rgb[ROI_Y:ROI_Y2, ROI_X:ROI_X2]     
    R, G, B  = np.mean(roi, axis=(0, 1))
    if R > G and R > B:                  return "Red"
    if G > R and G > B:                  return "Green"
    if B > R and B > G:                  return "Blue"
    if R > 200 and G > 200 and B > 200:  return "White"
    return "Unknown"



# CORRECT LINEAR MOTION 
def drive_corrected(robot, x: float, y: float, duration: float, label: str = ""):
    start = time.time()
    while time.time() - start < duration:
        robot.chassis.drive_speed(x=x, y=y, z=get_z_correction())
        frame = robot.camera.read_video_frame(strategy="newest")
        if frame is not None:
            cv2.imshow("RoboMaster - Live", draw_overlay(frame, "", label, {}))
            cv2.waitKey(1)
        time.sleep(0.02)
    robot.chassis.drive_speed(x=0, y=0, z=0)


# PROCEDURA DODGE 
def dodge_obstacle(robot, direction_y: float, obstacle_color: str):
    """
    P1 --> Move sideways (min. 8 cm) until the ToF sensor is clear
    P2 --> Turn 90° facing the obstacle
    P3 --> Move forward sideways: ignore the ToF sensor for 10 cm, then wait
          until the obstacle appears and then disappears from the sensor
    P4 --> EXTRA_PASS_CM extra distance beyond the edge
    P5 --> Turn back to the original heading
    P6 --> Return to the central path
    """
    print(obstacle_color)
    global target_yaw
    #side = "RIGHT" if direction_y > 0 else "LEFT"
    side = "RIGHT" if obstacle_color in ("Red") else "LEFT"
    print(f"\n[DODGE] ── Start maneuver towards {side} | Obstacle: {obstacle_color}")

    robot.chassis.drive_speed(x=0, y=0, z=0)
    # FIX #1: sleep BEFORE start_dodge → lateral_duration measures
    # only the time when the robot is actually moving
    time.sleep(0.2)

    min_strafe_secs    = (MIN_STRAFE_CM    / 100.0) / DODGE_Y_SPEED
    min_alongside_secs = (MIN_ALONGSIDE_CM / 100.0) / FORWARD_SPEED
    extra_pass_secs    = (EXTRA_PASS_CM    / 100.0) / FORWARD_SPEED

    # STEP 1: Lateral deviation 
    # Measure the trade AFTER it has settled, so that lateral_duration reflects only the movement
    print(f"[P1] Lateral deviation (min {MIN_STRAFE_CM:.0f} cm then free ToF)...")
    start_dodge = time.time()
    clear_count = 0

    while time.time() - start_dodge < 6.0:
        robot.chassis.drive_speed(x=0.0, y=direction_y, z=get_z_correction())
        elapsed = time.time() - start_dodge
        with _lock:
            dist = current_distance

        if elapsed >= min_strafe_secs:
            clear_count = clear_count + 1 if dist > CLEAR_THRESHOLD_MM else 0
            if clear_count >= 3:
                print(f"[P1] Free route after {elapsed:.2f}s.")
                break

        frame = robot.camera.read_video_frame(strategy="newest")
        if frame is not None:
            cv2.imshow("RoboMaster - Live",
                       draw_overlay(frame, obstacle_color,
                                    f"P1 SCARTO-->{side} | {elapsed:.1f}s"))
            cv2.waitKey(1)
        time.sleep(0.02)

    robot.chassis.drive_speed(x=0, y=0, z=0)

    # Additional margin to fit the total width of the frame
    # (included in `lateral_duration` --> will be correctly taken into account in P6)
    drive_corrected(robot, x=0.0, y=direction_y, duration=0.5,
                    label="P1 CHASSIS MARGIN")
    lateral_duration = time.time() - start_dodge
    print(f"[P1] Total lateral deflection: {lateral_duration:.2f}s "
          f"about {lateral_duration * DODGE_Y_SPEED * 100:.1f} cm")

    # STEP 2: Rotate 90° towards the obstacle 
    # direction_y > 0 (RIGHT) --> obstacle now on the left --> yaw_shift = -90
    # direction_y < 0 (LEFT) --> obstacle now on the right --> yaw_shift = +90
    yaw_shift = -90.0 if direction_y > 0 else +90.0
    print(f"[P2] Rotation {yaw_shift:+.0f}° towards the obstacle...")
    with _lock:
        target_yaw += yaw_shift
    drive_corrected(robot, x=0.0, y=0.0, duration=1.8,
                    label=f"P2 ROTATION {yaw_shift:+.0f}°")

    # STEP 3: Sideways movement 
    # After -90° rotation: +y robot = original goal direction --> fwd_y = +FORWARD
    # After +90° rotation: -y robot = original goal direction --> fwd_y = -FORWARD
    fwd_y = +FORWARD_SPEED if direction_y > 0 else -FORWARD_SPEED
    print(f"[P3] Coaching (ignore ToF {MIN_ALONGSIDE_CM:.0f} cm, then wait for it to disappear)...")
    start_fwd        = time.time()
    obstacle_in_view = False

    while time.time() - start_fwd < MAX_FWD_TIME:
        robot.chassis.drive_speed(x=0.0, y=fwd_y, z=get_z_correction())
        elapsed = time.time() - start_fwd
        with _lock:
            dist = current_distance

        frame = robot.camera.read_video_frame(strategy="newest")
        if frame is not None:
            phase_label = (f"P3 IGNORE ToF | {elapsed:.1f}s"
                           if elapsed < min_alongside_secs
                           else f"P3 SEARCH FOR OBSTACLE | {dist:.0f}mm")
            cv2.imshow("RoboMaster - Live",
                       draw_overlay(frame, obstacle_color, phase_label))
            cv2.waitKey(1)

        if elapsed >= min_alongside_secs:
            if dist <= CLEAR_THRESHOLD_MM:
                obstacle_in_view = True
            if obstacle_in_view and dist > CLEAR_THRESHOLD_MM:
                print(f"[P3] Obstacle exited the ToF after {elapsed:.2f}s.")
                break

        time.sleep(0.02)

    robot.chassis.drive_speed(x=0, y=0, z=0)

    # STEP 4: EXTRA_PASS_CM – extra beyond the edge 
    print(f"[P4] {EXTRA_PASS_CM:.0f} cm extra beyond the edge...")
    drive_corrected(robot, x=0.0, y=fwd_y, duration=extra_pass_secs,
                    label=f"P4 +{EXTRA_PASS_CM:.0f}cm")

    # STEP 5: Re-align the original heading 
    print("[P5] Realignment of original heading...")
    with _lock:
        target_yaw -= yaw_shift   # target_yaw torna a 0
    drive_corrected(robot, x=0.0, y=0.0, duration=1.8,
                    label="P5 REALIGNMENT")

    # FIX #2: waiting for stabilisation after P5 rotation\
    # Without this, P6 starts whilst the robot is still in micro-rotation\
    # and the lateral return trajectory is crooked
    time.sleep(0.3)

    # STEP 6: Return to the central path 
    # Move exactly lateral_duration * RETURN_FACTOR in the direction 
    # opposite to the deviation --> return to the starting line
    return_duration = lateral_duration * RETURN_FACTOR
    print(f"[P6] Back to the route ({return_duration:.2f}s "
          f"about {return_duration * DODGE_Y_SPEED * 100:.1f} cm)...")
    drive_corrected(robot, x=0.0, y=-direction_y, duration=return_duration,
                    label="P6 RETURN JOURNEY")

    time.sleep(0.2)
    print("[DODGE] ── Maneuver complete. Resuming route.\n")


# OVERLAY VIDEO 
def draw_overlay(frame, color_seen, state_msg, detected_colors: dict = None):
    display = frame.copy()
    with _lock:
        dist  = current_distance
        yaw   = accumulated_yaw
        t_yaw = target_yaw

    cv2.rectangle(display, (ROI_X, ROI_Y), (ROI_X2, ROI_Y2), (0, 255, 255), 2)
    cv2.putText(display, "ROI", (ROI_X + 5, ROI_Y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

    lines = [
        (f"State  : {state_msg}",                            (0, 255, 0)),
        (f"Dominant: {color_seen}",                          (255, 255, 0)),
        (f"Dist   : {dist:.0f} mm",                          (0, 255, 255)),
        (f"Yaw    : {yaw:+.1f}° --> Target: {t_yaw:+.0f}°", (255, 100, 100)),
    ]
    for i, (text, color) in enumerate(lines):
        cv2.putText(display, text, (10, 30 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    # Show all detected colours along with their pixel counts
    if detected_colors:
        all_colors_str = "  ".join(
            f"{k}:{v//1000}k" for k, v in sorted(detected_colors.items(),
                                                   key=lambda x: -x[1])
        )
        cv2.putText(display, all_colors_str, (10, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    bar_len   = int(np.clip(dist / max(OBSTACLE_DIST_MM, 1) * 200, 0, 200))
    bar_color = (0, 255, 0) if dist > OBSTACLE_DIST_MM else (0, 0, 255)
    cv2.rectangle(display, (10, 185), (10 + bar_len, 205), bar_color, -1)
    cv2.rectangle(display, (10, 185), (210, 205), (200, 200, 200), 1)
    return display


# MAIN
robot = Robot()
robot.initialize(conn_type="sta", sn="LOCAL")

try:
    print("Initialisation of sensors and camera...")
    robot.set_robot_mode(mode=rm_robot.FREE)
    robot.chassis.sub_attitude(freq=50, callback=attitude_handler)
    robot.sensor.sub_distance(freq=10, callback=distance_handler)
    robot.camera.start_video_stream(display=False, resolution="720p")
    try:
        robot.gimbal.recenter().wait_for_completed(timeout=2)
    except Exception:
        pass
    time.sleep(1.5)
    reset_yaw_reference()
    print("\nRobot is ready. Press Q to stop.\n")

    while True:
        frame = robot.camera.read_video_frame(strategy="newest")
        if frame is None:
            time.sleep(0.01)
            continue

        rgb_frame  = frame[:, :, ::-1]
        color_seen, detected_colors = detect_color(rgb_frame)

        with _lock:
            dist = current_distance

        # 1. BLUE TARGET 
        if color_seen == "blue":
            if dist > BLUE_STOP_MM:
                state_msg = "BLUE: Approaching..."
                speed_x   = 0.20 if dist > 150 else 0.05
                robot.chassis.drive_speed(x=speed_x, y=0.0,
                                          z=get_z_correction(max_z=15.0))
            else:
                state_msg = "BLUE TARGET REACHED ✓"
                robot.chassis.drive_speed(x=0, y=0, z=0)

        # 2. OBSTACLE DETECTED 
        elif dist <= OBSTACLE_DIST_MM:
            robot.chassis.drive_speed(x=0, y=0, z=0)

            direction_y = -DODGE_Y_SPEED if last_turn_was_right else +DODGE_Y_SPEED
            side        = "RIGHT" if direction_y > 0 else "LEFT"

            KNOWN_OBSTACLES = ("black", "orange", "green", "white", "yellow")
            if color_seen in KNOWN_OBSTACLES:
                label = color_seen.upper()
            else:
                label = guess_unknown_color(rgb_frame)
                color_seen = label
                
            state_msg = f"OBSTACLE {label} --> {side}"
            print(f"[MAIN] {state_msg}")

            dodge_obstacle(robot, direction_y=direction_y,
                           obstacle_color=color_seen or "Unknown")
            last_turn_was_right = (direction_y > 0)
            reset_yaw_reference()

        # 3. PROGRESS 
        else:
            state_msg = f"Progress | {color_seen}" if color_seen else "Progress"
            robot.chassis.drive_speed(x=FORWARD_SPEED, y=0.0,
                                      z=get_z_correction())

        cv2.imshow("RoboMaster - Live",
                draw_overlay(frame, str(color_seen), state_msg, detected_colors))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nStop required.")
            break
        time.sleep(0.02)

except KeyboardInterrupt:
    print("\Manual interruption.")
finally:
    print("\nShutdown...")
    robot.chassis.drive_speed(x=0, y=0, z=0)
    robot.sensor.unsub_distance()
    robot.chassis.unsub_attitude()
    cv2.destroyAllWindows()
    try:
        robot.camera.stop_video_stream()
        time.sleep(0.5)
        robot.close()
    except Exception:
        pass


# Final Plot
print("\nGeneration plot...")
if readings:
    dists_c = [d if d < 1999 else np.nan for _, d in readings]
    plt.figure(figsize=(12, 5))
    plt.plot(range(len(dists_c)), dists_c, '-o', markersize=3,
             color="steelblue", label="ToF Distance")
    plt.axhline(OBSTACLE_DIST_MM,   color='red',    linestyle='--',
                label=f"Dodge threshold ({OBSTACLE_DIST_MM} mm)")
    plt.axhline(CLEAR_THRESHOLD_MM, color='orange', linestyle='--',
                label=f"Clear threshold ({CLEAR_THRESHOLD_MM} mm)")
    plt.axhline(BLUE_STOP_MM,       color='green',  linestyle='--',
                label=f"Blue stop ({BLUE_STOP_MM} mm)")
    plt.title("Distance profile during the run")
    plt.xlabel("ToF sensors")
    plt.ylabel("Distance (mm)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    print("No data acquired.")
