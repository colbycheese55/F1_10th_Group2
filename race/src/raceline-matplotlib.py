import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import csv

# Load the map image
map_image_path = 'base_map.pgm'
map_image = Image.open(map_image_path)
map_array = np.array(map_image)


# Map metadata from the YAML file
origin_x, origin_y = -3.983246, -1.321383  # Origin of the map
resolution = 0.050000  # Resolution of the map

# Function to convert image coordinates to map coordinates
def image_to_map_coordinates(x_img, y_img, origin_x, origin_y, resolution):
    x_map = x_img * resolution + origin_x
    y_map = (map_array.shape[0] - y_img) * resolution + origin_y
    return x_map, y_map

# Initialize lists to store clicked points
x_clicked = []
y_clicked = []

# Function to handle mouse click events
def onclick(event):
    if event.xdata is not None and event.ydata is not None:
        x_clicked.append(event.xdata)
        y_clicked.append(event.ydata)
        ax.plot(event.xdata, event.ydata, 'ro')  # Mark the clicked point with a red dot
        fig.canvas.draw()  # Update the figure to show the new point

# Function to save the clicked points to a CSV file, adjusted for map coordinates
def save_race_line(filename='race_line.csv'):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        for x_img, y_img in zip(x_clicked, y_clicked):
            x_map, y_map = image_to_map_coordinates(x_img, y_img, origin_x, origin_y, resolution)
            writer.writerow([x_map, y_map, 0.0, 1.0])  # Assigning a constant speed of 1.0 for demonstration
    print(f"Race line saved to {filename}.")

# Set up the plot
fig, ax = plt.subplots()
ax.imshow(map_array, cmap='gray')
ax.set_title('Click to draw the race line')
fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()

# Save the race line after the plot window is closed
save_race_line('demoline.csv')