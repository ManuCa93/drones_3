# RoboMaster Autonomous Navigation & Obstacle Avoidance

Autonomous navigation system for a DJI RoboMaster EP simulated in **CoppeliaSIM**, using real-time colour detection and ToF distance sensing to navigate towards a blue target while dodging colour-coded obstacles.

## Demo

A full demonstration of the robot completing the course is available in **`video.mov`**.

## How It Works

The robot drives forward until it encounters an obstacle or a blue target. A central ROI in the camera feed is converted to HSV and compared against predefined colour ranges. The dominant colour determines what happens next:

- **Blue** — the goal. The robot slows down and stops at close range.
- **Red / Orange / Yellow / Black** — obstacle; the robot dodges to the **right**.
- **Green / White** — obstacle; the robot dodges to the **left**.

Obstacle avoidance follows a six-phase manoeuvre:

1. **Lateral strafe** — move sideways until the ToF sensor reads clear.
2. **90° rotation** — turn to face parallel to the obstacle.
3. **Alongside pass** — travel alongside until the obstacle leaves the sensor's field of view.
4. **Extra clearance** — continue a fixed distance beyond the obstacle edge.
5. **Re-align heading** — rotate back to the original bearing.
6. **Return to path** — strafe back to the centreline.

A proportional yaw controller (`KP_YAW = 2.0`) corrects heading drift throughout every phase, using accumulated gyro data to stay on target.

## Project Structure

```
.
├── main.py          # Full navigation and obstacle-avoidance logic
├── video.mov        # Recorded demonstration run
└── README.md
```

## Key Parameters

| Parameter | Value | Purpose |
|---|---|---|
| `OBSTACLE_DIST_MM` | 300 mm | Distance at which an obstacle triggers a dodge |
| `BLUE_STOP_MM` | 60 mm | Stopping distance from the blue target |
| `CLEAR_THRESHOLD_MM` | 1000 mm | ToF reading considered "clear" |
| `FORWARD_SPEED` | 0.30 m/s | Cruising speed |
| `DODGE_Y_SPEED` | 0.40 m/s | Lateral strafe speed during dodges |
| `MIN_PIXELS` | 10 000 | Minimum pixel count to register a colour |
| `EXTRA_PASS_CM` | 45 cm | Safety margin past an obstacle edge |

## Colour Detection

Detection uses HSV thresholding over a 250×250 px ROI centred in the 720p frame. The supported colours and their HSV ranges are defined in `COLOR_RANGES`. Two separate ranges for red (low and high hue) are merged into a single count.

## Requirements

- Python 3.8+
- CoppeliaSIM (with RoboMaster EP scene)
- `robomaster` SDK
- `numpy`
- `opencv-python`
- `matplotlib`

## Running

1. Launch CoppeliaSIM and load the RoboMaster scene.
2. Start the simulation.
3. Run the script:

```bash
python main.py
```

Press **Q** in the live video window to stop. A distance-profile plot is generated automatically at the end of the run.

## Output

At shutdown the script plots every ToF reading collected during the run, overlaid with the dodge, clear, and blue-stop thresholds — useful for analysing how the robot reacted to each obstacle.
