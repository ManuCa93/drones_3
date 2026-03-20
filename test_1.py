import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from robomaster.robot import Robot
from skimage.color import rgb2hsv
from skimage.measure import label, regionprops

# --- Variabili Globali Sensori ---
accumulated_yaw = 0.0
last_yaw = None
current_distance = 2000.0  # Inizializzato a 2 metri
readings = []  # Per il grafico: (angolo, distanza)

# --- Parametri di Controllo (Antiscivolo) ---
KP_YAW = 1.5  # Quanto forte corregge la traiettoria se va storto

# --- Variabile di Memoria per l'Ostacolo Sconosciuto ---
last_turn_was_right = False 

# --- CALLBACKS SENSORI (Il Giroscopio / IMU) ---
def attitude_handler(attitude_info):
    global accumulated_yaw, last_yaw
    yaw = attitude_info[0]
    
    if last_yaw is None:
        last_yaw = yaw
        return
        
    delta = yaw - last_yaw
    
    # Gestione del passaggio tra 180 e -180
    if delta > 180: delta -= 360
    elif delta < -180: delta += 360
    
    # accumulated_yaw tiene in memoria l'angolo assoluto rispetto alla partenza
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
    
    if R > G and R > B: return "Rosso/Marrone"
    if G > R and G > B: return "Verde"
    if B > R and B > G: return "Azzurro/Viola"
    if R > 200 and G > 200 and B > 200: return "Bianco"
    return "Sconosciuto"

# --- MANOVRE E OVERLAY ---
def draw_overlay(frame, color_seen, state_msg, z_correction=0.0):
    display = frame.copy()
    cv2.putText(display, f"Stato: {state_msg}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(display, f"Colore Visto: {color_seen}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(display, f"Distanza ToF: {current_distance:.0f} mm", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Mostriamo a schermo l'errore del giroscopio e la correzione applicata
    cv2.putText(display, f"Gyro Err: {accumulated_yaw:+.1f} deg", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2)
    cv2.putText(display, f"Z Correzione: {z_correction:+.1f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2)
    return display

def maneuver_and_pump_camera(robot, x_speed, y_speed, duration):
    global accumulated_yaw
    start_time = time.time()
    
    while time.time() - start_time < duration:
        # Calcolo correzione Z in tempo reale (se va a sinistra, lo spinge a destra)
        z_speed = -KP_YAW * accumulated_yaw
        z_speed = float(np.clip(z_speed, -30.0, 30.0)) # Limite per non farlo girare troppo forte
        
        robot.chassis.drive_speed(x=x_speed, y=y_speed, z=z_speed)
        
        frame = robot.camera.read_video_frame(strategy="newest")
        if frame is not None:
            display = frame.copy()
            cv2.putText(display, "SCHIVATA IN CORSO...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            # Mostra la correzione anche durante la schivata
            cv2.putText(display, f"Z Corr: {z_speed:.1f}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow("RoboMaster - Live", display)
            cv2.waitKey(1)
        time.sleep(0.05)
        
    robot.chassis.drive_speed(x=0, y=0, z=0)

def dodge_obstacle(robot, direction_y):
    # Schiva di lato (x=0, y=direzione)
    maneuver_and_pump_camera(robot, x_speed=0.0, y_speed=direction_y, duration=1.5)
    # Va avanti (x=0.3, y=0)
    maneuver_and_pump_camera(robot, x_speed=0.3, y_speed=0.0, duration=2.0)
    # Torna al centro (x=0, y=-direzione)
    maneuver_and_pump_camera(robot, x_speed=0.0, y_speed=-direction_y, duration=1.5)

# ==========================================
# --- ESECUZIONE PRINCIPALE ---
# ==========================================
robot = Robot()
robot.initialize(conn_type="sta", sn="LOCAL")

try:
    print("Inizializzazione Sensori e Camera...")
    # Sub freq a 50Hz per il giroscopio (più veloce legge, più rapida è la correzione)
    robot.chassis.sub_attitude(freq=50, callback=attitude_handler)
    robot.sensor.sub_distance(freq=10, callback=distance_handler)
    robot.camera.start_video_stream(display=False, resolution="720p")
    time.sleep(1.5)
    
    print("\nRobot pronto! Premi Q nella finestra video per fermare e generare il grafico.\n")

    while True:
        frame = robot.camera.read_video_frame(strategy="newest")
        if frame is None:
            time.sleep(0.01)
            continue

        rgb_frame = frame[:, :, ::-1]
        color_seen = detect_color(rgb_frame)
        z_correction = 0.0 # Valore di default per l'overlay
        
        # --- LOGICA DI NAVIGAZIONE E SCHIVATA ---
        
        # 1. CASO BLU (Avvicinati fino a 6cm / 60mm e FERMATI)
        if color_seen == "blue":
            if current_distance > 60:
                state_msg = "Vedo BLU: Mi avvicino..."
                z_correction = -KP_YAW * accumulated_yaw
                z_correction = float(np.clip(z_correction, -15.0, 15.0))
                robot.chassis.drive_speed(x=0.2, y=0.0, z=z_correction)
            else:
                state_msg = "Obiettivo BLU raggiunto (6cm)!"
                robot.chassis.drive_speed(x=0.0, y=0.0, z=0.0)
                
        # 2. SCHIVATA: Il sensore rileva ostacolo a 10 cm (100 mm) O MENO
        elif current_distance <= 100:
            if color_seen == "black":
                state_msg = "Ostacolo NERO a 10cm: Schivo a DESTRA"
                print(f"\n[{current_distance}mm] Rilevato NERO. Schivo a Destra.")
                dodge_obstacle(robot, direction_y=0.3)
                last_turn_was_right = True
                
            elif color_seen == "orange":
                state_msg = "Ostacolo ARANCIONE a 10cm: Schivo a SINISTRA"
                print(f"\n[{current_distance}mm] Rilevato ARANCIONE. Schivo a Sinistra.")
                dodge_obstacle(robot, direction_y=-0.3)
                last_turn_was_right = False
                
            else:
                # Vede un palo ma non riconosce il colore
                guessed_color = guess_unknown_color(rgb_frame)
                state_msg = f"Ostacolo ({guessed_color}) a 10cm!"
                
                if last_turn_was_right:
                    print(f"\n[{current_distance}mm] Ostacolo {guessed_color}. Ultima svolta: Destra -> Ora schivo a SINISTRA.")
                    dodge_obstacle(robot, direction_y=-0.3)
                    last_turn_was_right = False
                else:
                    print(f"\n[{current_distance}mm] Ostacolo {guessed_color}. Ultima svolta: Sinistra -> Ora schivo a DESTRA.")
                    dodge_obstacle(robot, direction_y=0.3)
                    last_turn_was_right = True

        # 3. NESSUN OSTACOLO VICINO (> 100 mm)
        else:
            if color_seen in ["black", "orange"]:
                state_msg = f"Vedo {color_seen.upper()} in lontananza..."
            else:
                state_msg = "Avanzamento Dritto Stabilizzato"
            
            # --- Auto-raddrizzamento continuo sulla via principale ---
            z_correction = -KP_YAW * accumulated_yaw
            z_correction = float(np.clip(z_correction, -15.0, 15.0)) # Più dolce quando va dritto
            
            robot.chassis.drive_speed(x=0.3, y=0.0, z=z_correction)

        # --- Aggiorna Finestra Video ---
        display_frame = draw_overlay(frame, str(color_seen), state_msg, z_correction)
        cv2.imshow("RoboMaster - Live", display_frame)

        # Esci premendo Q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nChiusura richiesta dall'utente.")
            break

        time.sleep(0.05)

except KeyboardInterrupt:
    print("\nInterruzione manuale (Ctrl+C).")
finally:
    print("\nSpegnimento motori e sensori...")
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
# --- GENERAZIONE GRAFICO FINALE ---
# ==========================================
print("\nGenerazione del grafico del sensore in corso...")
if readings:
    angles = [r[0] for r in readings]
    distances = [r[1] for r in readings]
    distances_clean = [d if d != 2000.0 else np.nan for d in distances]
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(distances_clean)), distances_clean, '-o', markersize=4, color="b", label="Distanza ostacoli")
    plt.title("Profilo di distanza rilevato durante il percorso")
    plt.xlabel("Tempo (letture del sensore)")
    plt.ylabel("Distanza (mm)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    print("Nessun dato acquisito dal sensore per il grafico.")