#!/usr/bin/env python
"""
Raceline Generator from PGM Map
Generates an optimal racing line from a PGM occupancy grid map.

The algorithm:
1. Loads PGM map and YAML metadata
2. Extracts track boundaries (inner and outer walls)
3. Computes a skeleton/centerline of the track
4. Optimizes the racing line by minimizing curvature and maximizing distance from walls
5. Smooths the line using spline interpolation
6. Exports waypoints as CSV in the format: x, y, z, w (compatible with pure pursuit)

Usage:
    python generate_raceline.py <map_file.pgm> [--yaml <map_file.yaml>] [--output <raceline.csv>]
"""

import numpy as np
import cv2
import yaml
import argparse
import sys
from scipy import ndimage
from scipy.interpolate import splprep, splev
from scipy.spatial import distance
from skimage.morphology import skeletonize, medial_axis
import matplotlib.pyplot as plt


class RacelineGenerator:
    def __init__(self, pgm_file, yaml_file=None, output_file='raceline.csv'):
        """
        Initialize the raceline generator.
        
        Args:
            pgm_file: Path to the PGM map file
            yaml_file: Path to the YAML metadata file (optional)
            output_file: Output CSV filename
        """
        self.pgm_file = pgm_file
        self.yaml_file = yaml_file or pgm_file.replace('.pgm', '.yaml')
        self.output_file = output_file
        
        self.map_image = None
        self.resolution = 0.05  # Default resolution (meters per pixel)
        self.origin = [0.0, 0.0, 0.0]  # Default origin
        
        self.track_mask = None
        self.centerline = None
        self.raceline = None
        self.waypoints = None
        
    def load_map(self):
        """Load the PGM map and YAML metadata."""
        print(f"Loading map from {self.pgm_file}...")
        
        # Load PGM image
        self.map_image = cv2.imread(self.pgm_file, cv2.IMREAD_GRAYSCALE)
        if self.map_image is None:
            raise ValueError(f"Failed to load map image: {self.pgm_file}")
        
        print(f"Map size: {self.map_image.shape}")
        
        # Load YAML metadata if available
        try:
            with open(self.yaml_file, 'r') as f:
                map_metadata = yaml.safe_load(f)
                self.resolution = map_metadata.get('resolution', 0.05)
                self.origin = map_metadata.get('origin', [0.0, 0.0, 0.0])
                print(f"Map resolution: {self.resolution} m/pixel")
                print(f"Map origin: {self.origin}")
        except Exception as e:
            print(f"Warning: Could not load YAML metadata: {e}")
            print("Using default resolution and origin")
    
    def extract_track(self):
        """Extract the drivable track area from the map."""
        print("Extracting track boundaries...")
        
        # In occupancy grids: 255=free, 0=occupied, 205=unknown
        # Create binary mask: free space = 1, obstacles/unknown = 0
        free_threshold = 250
        self.track_mask = (self.map_image >= free_threshold).astype(np.uint8)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((3, 3), np.uint8)
        self.track_mask = cv2.morphologyEx(self.track_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        self.track_mask = cv2.morphologyEx(self.track_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Erode slightly to create safety margin from walls
        safety_margin_pixels = int(0.15 / self.resolution)  # 15cm safety margin
        kernel_erode = np.ones((safety_margin_pixels, safety_margin_pixels), np.uint8)
        self.track_mask = cv2.erode(self.track_mask, kernel_erode, iterations=1)
        
        print(f"Track area: {np.sum(self.track_mask)} pixels")
        
    def compute_centerline(self):
        """Compute the centerline/skeleton of the track using medial axis transform."""
        print("Computing track centerline...")
        
        # Use medial axis transform to find the centerline
        # This gives us points equidistant from track boundaries
        skeleton, distance_map = medial_axis(self.track_mask, return_distance=True)
        
        # Thin the skeleton further
        skeleton = skeletonize(skeleton)
        
        # Extract skeleton points
        skeleton_points = np.argwhere(skeleton)
        
        if len(skeleton_points) == 0:
            raise ValueError("No centerline found. Check if the track mask is valid.")
        
        print(f"Found {len(skeleton_points)} centerline points")
        
        # Convert from image coordinates (row, col) to world coordinates (x, y)
        # Row = y-axis (flipped), Col = x-axis
        centerline_world = []
        for point in skeleton_points:
            row, col = point
            x = col * self.resolution + self.origin[0]
            y = (self.map_image.shape[0] - row) * self.resolution + self.origin[1]
            centerline_world.append([x, y])
        
        self.centerline = np.array(centerline_world)
        
        # Store distance map for later optimization
        self.distance_map = distance_map
        self.skeleton = skeleton
        
    def order_centerline_points(self):
        """Order centerline points to form a continuous path."""
        print("Ordering centerline points...")
        
        if len(self.centerline) == 0:
            return
        
        # Use nearest neighbor approach to order points
        ordered_points = [self.centerline[0]]
        remaining_points = list(self.centerline[1:])
        
        while remaining_points:
            last_point = ordered_points[-1]
            # Find nearest remaining point
            distances = [np.linalg.norm(last_point - p) for p in remaining_points]
            nearest_idx = np.argmin(distances)
            ordered_points.append(remaining_points[nearest_idx])
            remaining_points.pop(nearest_idx)
        
        self.centerline = np.array(ordered_points)
        
        # Check if it's a closed loop (distance from end to start is small)
        loop_distance = np.linalg.norm(self.centerline[-1] - self.centerline[0])
        if loop_distance < 0.5:  # Within 0.5 meters
            print("Detected closed loop track")
            # Add first point at the end to close the loop
            self.centerline = np.vstack([self.centerline, self.centerline[0]])
        
    def optimize_raceline(self):
        """Optimize the racing line using smoothing and curvature minimization."""
        print("Optimizing racing line...")
        
        # Start with centerline
        raceline = self.centerline.copy()
        
        # Apply smoothing using moving average
        window_size = 5
        raceline_smooth = np.copy(raceline)
        for i in range(len(raceline)):
            if i < window_size // 2 or i >= len(raceline) - window_size // 2:
                continue
            start_idx = i - window_size // 2
            end_idx = i + window_size // 2 + 1
            raceline_smooth[i] = np.mean(raceline[start_idx:end_idx], axis=0)
        
        self.raceline = raceline_smooth
        
    def smooth_with_spline(self, num_points=200):
        """Apply spline smoothing to create a smooth raceline."""
        print(f"Applying spline smoothing with {num_points} points...")
        
        if len(self.raceline) < 4:
            print("Warning: Not enough points for spline fitting")
            self.waypoints = self.raceline
            return
        
        # Prepare data for spline fitting
        x = self.raceline[:, 0]
        y = self.raceline[:, 1]
        
        # Check if it's a closed loop
        is_closed = np.linalg.norm(self.raceline[-1] - self.raceline[0]) < 0.5
        
        # Fit spline
        # Use periodic spline for closed tracks
        try:
            if is_closed:
                # For closed tracks, use periodic boundary conditions
                tck, u = splprep([x, y], s=0.1, per=True, k=3)
            else:
                tck, u = splprep([x, y], s=0.1, k=3)
            
            # Evaluate spline at evenly spaced points
            u_new = np.linspace(0, 1, num_points)
            x_new, y_new = splev(u_new, tck)
            
            self.waypoints = np.column_stack([x_new, y_new])
            print(f"Generated {len(self.waypoints)} smooth waypoints")
            
        except Exception as e:
            print(f"Warning: Spline fitting failed: {e}")
            print("Using original raceline points")
            self.waypoints = self.raceline
    
    def save_waypoints(self):
        """Save waypoints to CSV file in the format: x, y, z, w"""
        print(f"Saving waypoints to {self.output_file}...")
        
        if self.waypoints is None or len(self.waypoints) == 0:
            print("Error: No waypoints to save")
            return
        
        # Format: x, y, z, w (z=0.0, w=1.0 for compatibility)
        with open(self.output_file, 'w') as f:
            for point in self.waypoints:
                f.write(f"{point[0]:.2f},{point[1]:.2f},0.0,1.0\n")
        
        print(f"Saved {len(self.waypoints)} waypoints")
    
    def visualize(self, save_plot=True):
        """Visualize the map, centerline, and raceline."""
        print("Generating visualization...")
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot 1: Map and track mask
        axes[0].imshow(self.map_image, cmap='gray', origin='lower')
        axes[0].set_title('Original Map')
        axes[0].axis('equal')
        
        # Plot 2: Raceline on map
        axes[1].imshow(self.track_mask, cmap='gray', origin='lower')
        
        # Convert world coordinates back to image coordinates for plotting
        if self.centerline is not None and len(self.centerline) > 0:
            centerline_img = self.world_to_image(self.centerline)
            axes[1].plot(centerline_img[:, 1], centerline_img[:, 0], 'b-', 
                        linewidth=1, alpha=0.5, label='Centerline')
        
        if self.waypoints is not None and len(self.waypoints) > 0:
            waypoints_img = self.world_to_image(self.waypoints)
            axes[1].plot(waypoints_img[:, 1], waypoints_img[:, 0], 'r-', 
                        linewidth=2, label='Raceline')
            axes[1].scatter(waypoints_img[0, 1], waypoints_img[0, 0], 
                           c='green', s=100, zorder=5, label='Start')
        
        axes[1].set_title('Generated Raceline')
        axes[1].legend()
        axes[1].axis('equal')
        
        plt.tight_layout()
        
        if save_plot:
            plot_file = self.output_file.replace('.csv', '_visualization.png')
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {plot_file}")
        
        plt.show()
    
    def world_to_image(self, points):
        """Convert world coordinates to image coordinates."""
        points_img = np.zeros_like(points)
        points_img[:, 1] = (points[:, 0] - self.origin[0]) / self.resolution  # x -> col
        points_img[:, 0] = self.map_image.shape[0] - (points[:, 1] - self.origin[1]) / self.resolution  # y -> row (flipped)
        return points_img
    
    def generate(self, num_waypoints=200, visualize=True):
        """
        Main method to generate the raceline.
        
        Args:
            num_waypoints: Number of waypoints to generate
            visualize: Whether to show visualization
        """
        self.load_map()
        self.extract_track()
        self.compute_centerline()
        self.order_centerline_points()
        self.optimize_raceline()
        self.smooth_with_spline(num_points=num_waypoints)
        self.save_waypoints()
        
        if visualize:
            self.visualize()
        
        print("\n=== Raceline Generation Complete ===")
        print(f"Output file: {self.output_file}")
        print(f"Number of waypoints: {len(self.waypoints)}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate optimal raceline from PGM map file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_raceline.py base_map.pgm
  python generate_raceline.py base_map.pgm --yaml base_map.yaml --output my_raceline.csv
  python generate_raceline.py base_map.pgm --waypoints 300 --no-viz
        """
    )
    
    parser.add_argument('pgm_file', type=str, help='Path to PGM map file')
    parser.add_argument('--yaml', type=str, help='Path to YAML metadata file (default: same as PGM with .yaml extension)')
    parser.add_argument('--output', '-o', type=str, default='raceline_generated.csv',
                       help='Output CSV filename (default: raceline_generated.csv)')
    parser.add_argument('--waypoints', '-w', type=int, default=200,
                       help='Number of waypoints to generate (default: 200)')
    parser.add_argument('--no-viz', action='store_true',
                       help='Disable visualization')
    
    args = parser.parse_args()
    
    try:
        generator = RacelineGenerator(
            pgm_file=args.pgm_file,
            yaml_file=args.yaml,
            output_file=args.output
        )
        
        generator.generate(
            num_waypoints=args.waypoints,
            visualize=not args.no_viz
        )
        
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
