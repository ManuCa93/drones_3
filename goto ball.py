import time
import numpy as np
import cv2
from robomaster.robot import Robot
from skimage.measure import label, regionprops
from skimage.color import rgb2hsv



# --- PARAMETRI DA REGOLARE ---
TARGET_W = 300  # <--- I pixel di larghezza della palla quando è a 40cm!
CENTER_X = 640
KP_YAW   = 0.02   
MAX_Z    = 15.0   
KP_X     = 0.003  # Kp per l'avanzamento
MAX_X    = 0.2    # Limite velocità X (molto lento per sicurezza)
# -----------------------------

robot = Robot()
robot.initialize(conn_type="sta", sn="3JKCK5C00307ZD")

def detect_ball(img_rgb: np.ndarray) -> np.ndarray:
    """
    Ritorna una maschera booleana (True/False) per i pixel ARANCIONI.
    """
    # Convertiamo i pixel in float tra 0.0 e 1.0 (richiesto da rgb2hsv)
    img_float = img_rgb.astype(np.float32) / 255.0
    hsv = rgb2hsv(img_float)

    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    # ARANCIONE: Hue compreso tra 0.04 e 0.12
    hue_mask = (H >= 0.02) & (H <= 0.15)

    # Saturazione alta (l'arancione è un colore vivo) e Valore per escludere il nero
    sat_mask = S > 0.50
    val_mask = V > 0.20

    # Combiniamo tutto con l'AND logico (&)
    mask = hue_mask & sat_mask & val_mask
    return mask

def detect_ball_centroid(img_rgb: np.ndarray) -> dict | None:
    """
    Trova il centroide e la dimensione della palla arancione nell'immagine.
    """
    # 1. Ottieni la maschera binaria dalla funzione di prima
    mask = detect_ball(img_rgb)

    # 2. Etichetta le componenti connesse (trova i "gruppi" di pixel)
    labeled_mask = label(mask)

    # 3. Calcola le proprietà di ogni gruppo
    regions = regionprops(labeled_mask)

    # Se non ci sono gruppi (non ha visto niente di arancione), ritorna None
    if not regions:
        return None

    # 4. Prendi solo il gruppo con l'area (numero di pixel) più grande
    largest = max(regions, key=lambda r: r.area)

    # 5. Restituisci i dati. 
    # NOTA BENE: skimage restituisce (riga, colonna), ovvero (Asse Y, Asse X)!
    return {
        "centroid": largest.centroid,  # (y, x)
        "area":     largest.area,      # numero di pixel
        "bbox":     largest.bbox       # (min_y, min_x, max_y, max_x)
    }

def draw_overlay(frame, result, x_speed, z_speed):
    """Draws detection box, centroid, and telemetry onto the frame."""
    display = frame.copy()
    h, w = display.shape[:2]

    cx_center = w // 2
    cy_center = h // 2
    cv2.line(display, (cx_center - 20, cy_center), (cx_center + 20, cy_center), (255, 255, 0), 1)
    cv2.line(display, (cx_center, cy_center - 20), (cx_center, cy_center + 20), (255, 255, 0), 1)

    if result is not None:
        # ✅ Cast everything to plain int
        cx = int(result["centroid"][1])
        cy = int(result["centroid"][0])
        r0, c0, r1, c1 = [int(v) for v in result["bbox"]]
        current_w = c1 - c0
        error_dist = TARGET_W - current_w

        color = (0, 255, 0) if abs(error_dist) < 20 else (0, 100, 255) 
        cv2.rectangle(display, (c0, r0), (c1, r1), color, 2)
        cv2.circle(display, (cx, cy), 5, (0, 0, 255), -1)
        cv2.line(display, (cx_center, cy_center), (cx, cy), (200, 200, 200), 1)
        cv2.putText(display, f"W: {current_w}px", (c0, r0 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        telemetry = [
            f"Dist err: {error_dist:+.0f} px",
            f"Yaw  err: {cx - CENTER_X:+.0f} px",
            f"V_x:  {x_speed:+.3f} m/s",
            f"V_z:  {z_speed:+.1f} deg/s",
        ]
        for i, line in enumerate(telemetry):
            cv2.putText(display, line, (10, 25 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display, line, (10, 25 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    else:
        cv2.putText(display, "PALLA PERSA", (w // 2 - 90, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    return display


try:
    robot.camera.start_video_stream(display=False, resolution="720p")
    time.sleep(1)

    print("Test REACH avviato. Premi Q nella finestra o Ctrl+C per uscire.")

    x_speed, z_speed = 0.0, 0.0  # keep last values for overlay when ball lost

    while True:
        frame = robot.camera.read_video_frame(strategy="newest")
        if frame is None:
            time.sleep(0.01)
            continue

        rgb_frame = frame[:, :, ::-1]
        result = detect_ball_centroid(rgb_frame)

        if result is not None:
            cx = result["centroid"][1]
            r0, c0, r1, c1 = result["bbox"]
            current_w = c1 - c0

            error_yaw  = cx - CENTER_X
            z_speed    = float(np.clip(KP_YAW * error_yaw, -MAX_Z, MAX_Z))

            error_dist = TARGET_W - current_w
            x_speed    = float(np.clip(KP_X * error_dist, -MAX_X, MAX_X))

            robot.chassis.drive_speed(x=x_speed, y=0, z=z_speed)
            print(f"\rDist Err: {error_dist:+.0f}px | V_x: {x_speed:+.2f} m/s | V_z: {z_speed:+.1f} °/s  ",
                  end="", flush=True)
        else:
            x_speed, z_speed = 0.0, 0.0
            robot.chassis.drive_speed(x=0, y=0, z=0)
            print("\rPalla persa, stop immediato.                        ", end="", flush=True)

        # --- Live camera window ---
        display_frame = draw_overlay(frame, result, x_speed, z_speed)
        cv2.imshow("RoboMaster – Live", display_frame)

        # Esci premendo Q o ESC nella finestra
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            print("\nChiusura dalla finestra.")
            break

        time.sleep(0.05)

except KeyboardInterrupt:
    print("\nStop!")
finally:
    robot.chassis.drive_speed(x=0, y=0, z=0)
    cv2.destroyAllWindows()
    try:
        robot.camera.stop_video_stream()
        time.sleep(0.5)
        robot.close()
    except Exception:
        pass
    print("Motori fermati")