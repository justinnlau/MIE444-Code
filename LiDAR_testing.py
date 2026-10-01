import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from rplidar import RPLidar

PORT_NAME = 'COM5'  # Update this to match your system's port

lidar = RPLidar(PORT_NAME, baudrate=115200)

plt.ion()
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_facecolor('black')

# Draw the grid and limits ONCE outside the loop
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(1))
ax.set_xlim(-15, 15)
ax.set_ylim(-15, 15)
ax.grid(True, color='dimgray', linestyle='--', linewidth=0.5)

ax.scatter(0, 0, c='yellow', s=50, label='LiDAR')

# Create an empty scatter plot object to update later
points_plot = ax.scatter([], [], c='lime', s=10, alpha=0.8)

# Initialize a history buffer to create a "radar memory" effect
history_x = []
history_y = []

try:
    print("LiDAR spinning... Close window or press Ctrl+C to stop.")
    for scan in lidar.iter_scans(max_buf_meas=5000):
        angles = np.radians([meas[1] for meas in scan])
        distances = np.array([meas[2] for meas in scan]) / 25.4  

        x = distances * np.sin(angles)
        y = distances * np.cos(angles)

        # Add new points to the history buffer
        history_x.extend(x)
        history_y.extend(y)

        # Keep only the last 1500 points (roughly 3 or 4 full 360-degree spins)
        if len(history_x) > 1500:
            history_x = history_x[-1500:]
            history_y = history_y[-1500:]

        # Plot the accumulated history rather than just the current fragmented scan
        points_plot.set_offsets(np.c_[history_x, history_y])
        
        # Flush UI events
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

except KeyboardInterrupt:
    print("Stopping...")
finally:
    lidar.stop()
    lidar.stop_motor()
    lidar.disconnect()
    plt.close()