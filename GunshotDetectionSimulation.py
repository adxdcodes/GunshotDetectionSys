import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Constants
SPEED_OF_SOUND = 343.0  # Speed of sound in m/s
MIC_DISTANCE = 1  # Reduced distance between microphones in meters
GUNSHOT_AREA_SIZE = 4  # Increased area size for gunshot (meters)

# 3D positions of microphones (centered around the origin)
mic_positions = np.array([
    [MIC_DISTANCE, 0, 0],  # Right of center (X+)
    [-MIC_DISTANCE, 0, 0],  # Left of center (X-)
    [0, MIC_DISTANCE, 0],  # Front of center (Y+)
    [0, -MIC_DISTANCE, 0],  # Back of center (Y-)
    [0, 0, MIC_DISTANCE],  # Above center (Z+)
    [0, 0, -MIC_DISTANCE]  # Below center (Z-)
])


# Function to simulate gunshot and calculate delays and amplitudes
def simulate_gunshot_3d(gunshot_position):
    distances = np.linalg.norm(mic_positions - gunshot_position, axis=1)
    time_delays = distances / SPEED_OF_SOUND
    amplitudes = 1 / (distances + 1e-6)  # Avoid division by zero
    return time_delays, amplitudes


# Function to calculate Direction of Arrival (DOA)
def calculate_doa(time_delays, mic_positions):
    ref_position = mic_positions[0]
    vector_to_gunshot = np.zeros(3)
    for i in range(1, len(mic_positions)):
        distance_diff = (time_delays[i] - time_delays[0]) * SPEED_OF_SOUND
        vector_to_gunshot += (mic_positions[i] - ref_position) * distance_diff

    vector_to_gunshot /= np.linalg.norm(vector_to_gunshot)
    azimuth = np.arctan2(vector_to_gunshot[1], vector_to_gunshot[0]) * 180 / np.pi
    elevation = np.arcsin(vector_to_gunshot[2]) * 180 / np.pi

    return azimuth, elevation


# Function to handle clicking in 3D space
def on_click(event, fig, ax, ax_circle):
    if event.inaxes == ax:
        gunshot_x_range = GUNSHOT_AREA_SIZE
        gunshot_y_range = GUNSHOT_AREA_SIZE
        gunshot_z_range = GUNSHOT_AREA_SIZE

        gunshot_x = event.xdata
        gunshot_y = event.ydata

        if gunshot_x is not None and gunshot_y is not None:
            gunshot_x += gunshot_x_range * (2 * np.random.random() - 1)
            gunshot_y += gunshot_y_range * (2 * np.random.random() - 1)
            gunshot_z = gunshot_z_range * (2 * np.random.random() - 1)

            gunshot_position = np.array([gunshot_x, gunshot_y, gunshot_z])

            time_delays, amplitudes = simulate_gunshot_3d(gunshot_position)
            azimuth, elevation = calculate_doa(time_delays, mic_positions)

            ax.scatter(gunshot_position[0], gunshot_position[1], gunshot_position[2], c='r', marker='x', s=100,
                       label="Gunshot")
            for mic_position in mic_positions:
                ax.plot([gunshot_position[0], mic_position[0]],
                        [gunshot_position[1], mic_position[1]],
                        [gunshot_position[2], mic_position[2]], 'k--')

            fig.canvas.draw()

            update_doa_circle(ax_circle, azimuth)

            print("Gunshot Position (meters):", gunshot_position)
            print("Time Delays (seconds):", time_delays)
            print("Amplitudes:", amplitudes)
            print(f"Direction of Arrival - Azimuth: {azimuth:.2f} degrees, Elevation: {elevation:.2f} degrees\n")


# Function to update the DOA in the 360-degree circle
def update_doa_circle(ax_circle, azimuth):
    ax_circle.clear()
    circle = plt.Circle((0, 0), 1, color='b', fill=False)
    ax_circle.add_artist(circle)

    azimuth_rad = np.deg2rad(azimuth)
    ax_circle.plot([0, np.cos(azimuth_rad)], [0, np.sin(azimuth_rad)], color='r', marker='o')
    ax_circle.set_xlim([-1.5, 1.5])
    ax_circle.set_ylim([-1.5, 1.5])

    ax_circle.set_xlabel('X axis (meters)')
    ax_circle.set_ylabel('Y axis (meters)')
    ax_circle.set_title(f'Direction of Arrival (Azimuth): {azimuth:.2f}°')
    ax_circle.set_aspect('equal', 'box')
    plt.draw()


# Function to plot the microphones and prepare for 3D click interactions
def plot_3d_simulation():
    fig = plt.figure(num='Simulator', figsize=(12, 6))  # Set window title to 'Simulator'

    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(mic_positions[:, 0], mic_positions[:, 1], mic_positions[:, 2], c='b', marker='o', label="Microphones")
    ax.quiver(0, 0, 0, GUNSHOT_AREA_SIZE, 0, 0, color='r', arrow_length_ratio=0.1, label='X-axis')
    ax.quiver(0, 0, 0, 0, GUNSHOT_AREA_SIZE, 0, color='g', arrow_length_ratio=0.1, label='Y-axis')
    ax.quiver(0, 0, 0, 0, 0, GUNSHOT_AREA_SIZE, color='b', arrow_length_ratio=0.1, label='Z-axis')

    ax.set_xlim([-GUNSHOT_AREA_SIZE, GUNSHOT_AREA_SIZE])
    ax.set_ylim([-GUNSHOT_AREA_SIZE, GUNSHOT_AREA_SIZE])
    ax.set_zlim([-GUNSHOT_AREA_SIZE, GUNSHOT_AREA_SIZE])
    ax.set_xlabel('X axis (meters)')
    ax.set_ylabel('Y axis (meters)')
    ax.set_zlabel('Z axis (meters)')
    ax.set_title('3D Microphone Array Signal Simulation')
    ax.legend()

    ax_circle = fig.add_subplot(122)
    update_doa_circle(ax_circle, 0)

    plt.subplots_adjust(wspace=0.4)  # Increased space between subplots

    fig.canvas.mpl_connect('button_press_event', lambda event: on_click(event, fig, ax, ax_circle))

    plt.show()


# Run the simulation
plot_3d_simulation()