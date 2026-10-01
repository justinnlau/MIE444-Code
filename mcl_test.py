import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from rplidar import RPLidar

PORT_NAME = 'COM5'
CELL_SIZE = 0.0762  # 3 inches per tile in meters
NUM_PARTICLES = 400  # Increased slightly for better global multi-hypothesis coverage
NUM_RAYS_TO_CHECK = 12
MAX_RANGE = 4.0
CLOSE_THRESHOLD = 0.0254  # 1 inch in meters

# 32x16 Maze Map (Floor visually only, no longer used for wall collisions)
maze_matrix = np.array([
    [0,1,1,0,0,1,0,1,1,1,0,0,1,0,0,1,0,1,0,0,1,1,0,1,0,1,0,0,0,1,0,0],
    [0,0,1,0,0,0,1,0,0,1,1,0,1,0,0,1,1,0,0,1,1,1,0,0,0,1,1,0,1,0,0,0],
    [1,0,1,1,0,0,1,1,1,1,1,0,1,1,1,0,1,0,1,0,0,1,0,0,0,0,1,1,0,1,0,1],
    [0,0,0,0,1,0,1,0,0,1,1,1,1,1,1,1,0,0,1,0,0,1,0,1,1,1,0,0,1,0,0,0],
    [0,0,0,0,1,0,1,0,1,1,1,0,0,1,1,0,1,0,0,0,1,0,1,1,0,0,0,1,1,1,1,0],
    [1,1,1,1,1,1,0,0,1,1,1,1,0,1,1,0,0,1,1,1,1,0,1,1,1,0,1,0,1,0,0,0],
    [1,0,0,1,0,0,1,1,1,1,0,0,0,0,0,1,0,0,1,0,0,1,0,1,0,0,1,0,1,1,1,0],
    [0,0,1,1,1,0,0,1,1,1,0,1,0,1,1,0,0,1,1,0,1,0,1,1,0,1,1,1,0,0,1,1],
    [0,0,1,0,0,1,1,1,0,0,0,0,0,1,0,1,1,1,0,0,0,1,1,0,0,1,1,1,1,1,0,1],
    [0,0,1,1,1,0,0,0,0,0,1,1,1,1,0,1,1,0,0,0,0,1,0,1,1,1,1,1,0,1,1,1],
    [1,0,0,0,0,1,1,1,0,1,1,1,0,1,1,0,1,0,0,0,1,0,0,1,0,0,1,0,1,1,1,0],
    [0,1,0,1,1,1,0,0,1,1,1,0,0,1,1,0,0,1,1,1,1,0,0,0,1,0,0,0,1,0,0,0],
    [0,0,1,0,1,0,0,0,0,1,0,0,1,1,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,0,0,1],
    [1,1,1,1,1,0,0,0,0,0,1,1,0,0,1,1,0,0,0,0,0,0,1,1,1,0,0,0,0,0,1,0],
    [0,0,1,0,0,0,1,1,1,1,1,0,0,1,1,1,1,0,1,1,1,1,1,0,1,0,0,0,1,1,0,0],
    [1,1,0,0,1,1,1,1,0,1,1,0,0,1,0,0,0,0,1,1,1,0,0,1,1,0,0,1,1,0,1,0]
])

ROWS, COLS = maze_matrix.shape
MAX_X = COLS * CELL_SIZE
MAX_Y = ROWS * CELL_SIZE

# ----------------- GEOMETRIC WALL DEFINITIONS -----------------
red_boxes = [
    (16, 0, 4, 4),   # Top block 1
    (24, 0, 4, 4),   # Top block 2
    (8, 4, 4, 4),    # Mid-left block
    (4, 8, 4, 4),    # Lower-mid continuous block part 1
    (12, 8, 8, 4),   # Lower-mid continuous block part 2
    (24, 8, 4, 8)    # Right-most tall block
]

scaled_lines = []

# 1. Outer boundary walls
scaled_lines.extend([
    (0, 0, MAX_X, 0),
    (MAX_X, 0, MAX_X, MAX_Y),
    (MAX_X, MAX_Y, 0, MAX_Y),
    (0, MAX_Y, 0, 0)
])

# 2. Inner obstacle boxes
for c, r, w, h in red_boxes:
    x1, y1 = c * CELL_SIZE, r * CELL_SIZE
    x2, y2 = (c + w) * CELL_SIZE, (r + h) * CELL_SIZE
    scaled_lines.extend([
        (x1, y1, x2, y1),
        (x2, y1, x2, y2),
        (x2, y2, x1, y2),
        (x1, y2, x1, y1)
    ])

def ray_intersect_lines(rx, ry, rtheta, max_range):
    """Simulates a LiDAR laser ray bouncing off the closest red line segment."""
    dx = math.cos(rtheta)
    dy = math.sin(rtheta)
    min_t = max_range
    
    for (x1, y1, x2, y2) in scaled_lines:
        vx = x2 - x1
        vy = y2 - y1
        det = dy * vx - dx * vy
        
        if abs(det) < 1e-6:
            continue
            
        px = x1 - rx
        py = y1 - ry
        
        t = (py * vx - px * vy) / det
        u = (dx * py - dy * px) / det
        
        if 0.001 < t < min_t and 0.0 <= u <= 1.0:
            min_t = t
            
    return min_t

def is_inside_obstacle(px, py):
    """Safety check so particles don't spawn inside a solid wall block."""
    if px < 0 or px > MAX_X or py < 0 or py > MAX_Y:
        return True
    for (c, r, w, h) in red_boxes:
        if (c * CELL_SIZE <= px <= (c + w) * CELL_SIZE) and \
           (r * CELL_SIZE <= py <= (r + h) * CELL_SIZE):
            return True
    return False

def initialize_particles():
    particles = []
    while len(particles) < NUM_PARTICLES:
        px = np.random.uniform(0, MAX_X)
        py = np.random.uniform(0, MAX_Y)
        if not is_inside_obstacle(px, py):
            ptheta = np.random.uniform(0, 2 * math.pi)
            particles.append([px, py, ptheta, 1.0/NUM_PARTICLES])
    return np.array(particles)

def update_weights(particles, real_scan_data):
    weights = np.zeros(NUM_PARTICLES)
    for i, p in enumerate(particles):
        if is_inside_obstacle(p[0], p[1]):
            weights[i] = 0.0
            continue
            
        prob = 1.0
        for angle_deg, real_dist in real_scan_data:
            if real_dist < 0.15 or real_dist > MAX_RANGE:
                continue
            
            global_angle = p[2] + math.radians(angle_deg)
            expected_dist = ray_intersect_lines(p[0], p[1], global_angle, MAX_RANGE)
            
            error = abs(expected_dist - real_dist)
            # Slightly wider variance (0.15 instead of 0.05) to maintain multi-modal hypotheses in open/similar spaces
            prob *= math.exp(-(error**2) / 0.15)
        
        weights[i] = prob
    
    w_sum = np.sum(weights)
    if w_sum == 0:
        weights.fill(1.0/NUM_PARTICLES)
    else:
        weights /= w_sum
    return weights

def resample(particles, weights):
    # Standard low-variance / multinomial resampling
    indices = np.random.choice(range(NUM_PARTICLES), size=NUM_PARTICLES, p=weights)
    new_particles = particles[indices].copy()
    
    # Add noise for exploration
    new_particles[:, 0] += np.random.normal(0, 0.02, NUM_PARTICLES)
    new_particles[:, 1] += np.random.normal(0, 0.02, NUM_PARTICLES)
    new_particles[:, 2] += np.random.normal(0, 0.08, NUM_PARTICLES)
    new_particles[:, 2] %= (2 * math.pi)
    
    # Random particle injection (10% of particles) to re-introduce hypotheses if lost or exploring open areas
    num_random = int(0.10 * NUM_PARTICLES)
    for i in range(num_random):
        while True:
            px = np.random.uniform(0, MAX_X)
            py = np.random.uniform(0, MAX_Y)
            if not is_inside_obstacle(px, py):
                new_particles[i, 0] = px
                new_particles[i, 1] = py
                new_particles[i, 2] = np.random.uniform(0, 2 * math.pi)
                break
                
    new_particles[:, 3] = 1.0 / NUM_PARTICLES
    return new_particles

# ----------------- MAIN LOOP -----------------
plt.ion()
fig, ax = plt.subplots(figsize=(12, 6))
particles = initialize_particles()

print(f"Connecting to LiDAR on {PORT_NAME}...")
lidar = RPLidar(PORT_NAME, baudrate=115200)

try:
    print("MCL Visualizer Running with Multi-Hypothesis Tracking & Wall Proximity Monitor.")
    for scan in lidar.iter_scans(max_buf_meas=1500):
        scan_dict = {int(meas[1]): meas[2]/1000.0 for meas in scan}
        real_scan_data = []
        for target_angle in range(0, 360, int(360/NUM_RAYS_TO_CHECK)):
            closest_angle = min(scan_dict.keys(), key=lambda k: abs(k - target_angle), default=None)
            if closest_angle is not None and abs(closest_angle - target_angle) < 5:
                real_scan_data.append((closest_angle, scan_dict[closest_angle]))

        if len(real_scan_data) < 3:
            continue
            
        # Check minimum distance to wall from real scan data
        min_lidar_dist = min([d for _, d in real_scan_data if d > 0.05], default=MAX_RANGE)
        too_close = min_lidar_dist < CLOSE_THRESHOLD
            
        weights = update_weights(particles, real_scan_data)
        particles = resample(particles, weights)
        
        ax.clear()
        
        # 1. Render base grid map
        ax.imshow(maze_matrix, cmap='binary', origin='upper', 
                  extent=[0, MAX_X, MAX_Y, 0])
                  
        # 2. Draw outer boundary
        outer_rect = Rectangle((0, 0), MAX_X, MAX_Y, 
                               linewidth=2, edgecolor='crimson', facecolor='none')
        ax.add_patch(outer_rect)
        
        # 3. Draw internal red boundary boxes
        for (c, r, w, h) in red_boxes:
            rect = Rectangle((c*CELL_SIZE, r*CELL_SIZE), w*CELL_SIZE, h*CELL_SIZE, 
                             linewidth=2.5, edgecolor='crimson', facecolor='none', zorder=5)
            ax.add_patch(rect)
        
        # 4. Plot particles as small dots
        ax.scatter(particles[:, 0], particles[:, 1], 
                   c='lime', s=3, alpha=0.7, zorder=10, label="Particles")
        
        # 5. Compute Heading angle from best particle
        best_idx = np.argmax(weights)
        best_heading_deg = math.degrees(particles[best_idx, 2]) % 360
        
        ax.set_xlim(-0.2, MAX_X + 0.2)
        ax.set_ylim(MAX_Y + 0.2, -0.2)
        
        # Title and Proximity Warning Indicator
        title_str = f"Active MCL | Heading: {best_heading_deg:.1f}°"
        if too_close:
            title_str += " | ⚠️ TOO CLOSE TO WALL!"
            ax.set_title(title_str, fontsize=12, fontweight='bold', color='red')
        else:
            ax.set_title(title_str, fontsize=11, fontweight='bold', color='black')
        
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

except KeyboardInterrupt:
    print("Stopping...")
finally:
    lidar.stop()
    lidar.stop_motor()
    lidar.disconnect()
    plt.close()