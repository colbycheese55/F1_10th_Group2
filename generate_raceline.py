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
        
        # In occupancy grids: 255=free space (white), anything else is obstacle/wall
        # Only pixels very close to white (254-255) are considered free space
        free_threshold = 254
        self.track_mask = (self.map_image >= free_threshold).astype(np.uint8)
        
        print(f"Initial free space: {np.sum(self.track_mask)} pixels")
        
        # Find the largest connected component (the main track area)
        # This removes small isolated white pixels that aren't part of the track
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(self.track_mask, connectivity=8)
        
        if num_labels <= 1:
            raise ValueError("No track found in the map")
        
        # Find the largest component (excluding background label 0)
        largest_component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        self.track_mask = (labels == largest_component).astype(np.uint8)
        
        print(f"Largest track component: {np.sum(self.track_mask)} pixels")
        
        # Apply light morphological closing to fill small holes
        kernel = np.ones((3, 3), np.uint8)
        self.track_mask = cv2.morphologyEx(self.track_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Erode to create safety margin from walls
        safety_margin_pixels = int(0.20 / self.resolution)  # 20cm safety margin
        kernel_erode = np.ones((safety_margin_pixels, safety_margin_pixels), np.uint8)
        self.track_mask = cv2.erode(self.track_mask, kernel_erode, iterations=1)
        
        print(f"Track area after safety margin: {np.sum(self.track_mask)} pixels")
        
    def compute_centerline(self):
        """Compute the centerline/skeleton of the track using distance transform thresholding."""
        print("Computing track centerline...")
        
        # Calculate Euclidean Distance Transform
        # This tells us the distance from each free pixel to the nearest wall
        from scipy.ndimage import distance_transform_edt
        dist_transform = distance_transform_edt(self.track_mask)
        
        print(f"Distance transform max: {dist_transform.max():.2f} pixels")
        
        # Threshold the distance transform to keep only points far from walls
        # This naturally creates a clean centerline without branches
        # Threshold: keep points that are at least X% of the maximum distance from walls
        THRESHOLD = 0.45  # 35% of max distance - adjust if needed
        centers = dist_transform > THRESHOLD * dist_transform.max()
        
        print(f"Points after distance threshold: {np.sum(centers)}")
        
        # Apply skeletonization to thin the thresholded region to a 1-pixel line
        skeleton = skeletonize(centers)
        
        print(f"Skeleton points after thinning: {np.sum(skeleton)}")
        
        # Extract skeleton points
        skeleton_points = np.argwhere(skeleton)
        
        if len(skeleton_points) == 0:
            raise ValueError("No centerline found. Check if the track mask is valid.")
        
        print(f"Found {len(skeleton_points)} centerline points for raceline")
        
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
        self.distance_map = dist_transform
        self.skeleton = skeleton
    
    def remove_branches(self, skeleton):
        """Remove branch points from skeleton to get a single continuous loop."""
        # Create a copy to work with
        skel = skeleton.copy().astype(np.uint8)
        
        # Define 8-connectivity kernel (don't count center pixel)
        kernel = np.array([[1, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]], dtype=np.uint8)
        
        # Iteratively remove branch points (points with more than 2 neighbors)
        for iteration in range(15):
            # Count neighbors for each point (8-connectivity, excluding center)
            neighbor_count = cv2.filter2D(skel, -1, kernel)
            
            # Branch points have more than 2 neighbors
            # Normal path points have exactly 2 neighbors (like beads on a string)
            # Endpoints have exactly 1 neighbor
            branch_points = (neighbor_count > 2) & (skel > 0)
            
            num_branches = np.sum(branch_points)
            if num_branches == 0:
                print(f"Branch removal complete after {iteration} iterations")
                break
            
            # Remove branch points
            skel[branch_points] = 0
        
        # Remove endpoints (dead ends) iteratively to clean up dangling segments
        for iteration in range(15):
            neighbor_count = cv2.filter2D(skel, -1, kernel)
            
            # Endpoints have exactly 1 neighbor
            endpoints = (neighbor_count == 1) & (skel > 0)
            
            num_endpoints = np.sum(endpoints)
            if num_endpoints == 0:
                print(f"Endpoint removal complete after {iteration} iterations")
                break
                
            skel[endpoints] = 0
        
        return skel
        
    def order_centerline_points(self):
        """Order centerline points using greedy nearest-neighbor to follow a single path."""
        print("Ordering centerline points...")
        
        if len(self.centerline) == 0:
            return
        
        # Start from the leftmost point
        start_idx = np.argmin(self.centerline[:, 0])
        
        # Greedy nearest-neighbor traversal
        visited = np.zeros(len(self.centerline), dtype=bool)
        ordered_points = []
        
        current_idx = start_idx
        max_step = 0.2  # Maximum distance to next point (meters)
        
        while not visited[current_idx]:
            visited[current_idx] = True
            ordered_points.append(self.centerline[current_idx])
            
            # Find nearest unvisited neighbor
            min_dist = float('inf')
            next_idx = None
            
            for i in range(len(self.centerline)):
                if not visited[i]:
                    dist = np.linalg.norm(self.centerline[current_idx] - self.centerline[i])
                    if dist < min_dist and dist <= max_step:
                        min_dist = dist
                        next_idx = i
            
            # If no valid next point, we're done
            if next_idx is None:
                break
            
            current_idx = next_idx
        
        self.centerline = np.array(ordered_points)
        print(f"Ordered path has {len(self.centerline)} points")
        
        # Check if it's a closed loop
        if len(self.centerline) > 10:
            loop_distance = np.linalg.norm(self.centerline[-1] - self.centerline[0])
            print(f"Loop closure distance: {loop_distance:.2f}m")
            
            if loop_distance < 0.5:
                print("Detected closed loop track (already complete)")
            elif loop_distance < 1.0:
                print("Detected closed loop track - adding closure point")
                self.centerline = np.vstack([self.centerline, self.centerline[0]])
        
    def optimize_raceline(self):
        """Optimize the racing line by removing outliers and smoothing."""
        print("Optimizing racing line...")
        
        if len(self.centerline) < 10:
            self.raceline = self.centerline
            return
        
        # Remove outliers based on local curvature
        raceline = self.remove_outliers(self.centerline.copy())
        
        # Apply light smoothing using moving average
        window_size = 3
        raceline_smooth = np.copy(raceline)
        for i in range(len(raceline)):
            if i < window_size // 2 or i >= len(raceline) - window_size // 2:
                continue
            start_idx = i - window_size // 2
            end_idx = i + window_size // 2 + 1
            raceline_smooth[i] = np.mean(raceline[start_idx:end_idx], axis=0)
        
        self.raceline = raceline_smooth
        print(f"Optimized raceline has {len(self.raceline)} points")
    
    def remove_outliers(self, points):
        """Remove outlier points that create sharp angles."""
        if len(points) < 3:
            return points
        
        clean_points = [points[0]]
        
        for i in range(1, len(points) - 1):
            # Calculate angle at this point
            v1 = points[i] - points[i-1]
            v2 = points[i+1] - points[i]
            
            # Skip if vectors are too short
            if np.linalg.norm(v1) < 0.01 or np.linalg.norm(v2) < 0.01:
                continue
            
            # Calculate angle between vectors
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            
            # Only keep points where angle is reasonable (not too sharp)
            # Threshold: 45 degrees (pi/4 radians)
            if angle < 2.5:  # Less than ~143 degrees is OK
                clean_points.append(points[i])
        
        clean_points.append(points[-1])
        
        removed = len(points) - len(clean_points)
        if removed > 0:
            print(f"Removed {removed} outlier points")
        
        return np.array(clean_points)
        
    def smooth_with_spline(self, num_points=200):
        """Downsample to evenly spaced waypoints."""
        print(f"Downsampling to {num_points} waypoints...")
        
        if len(self.raceline) < 4:
            print("Warning: Not enough points")
            self.waypoints = self.raceline
            return
        
        # Simple downsampling: take every Nth point
        step = max(1, len(self.raceline) // num_points)
        self.waypoints = self.raceline[::step]
        
        # Ensure we have the last point if it's a closed loop
        is_closed = np.linalg.norm(self.raceline[-1] - self.raceline[0]) < 0.5
        if is_closed and len(self.waypoints) > 0:
            # Make sure last point connects back to first
            if np.linalg.norm(self.waypoints[-1] - self.waypoints[0]) > 0.3:
                self.waypoints = np.vstack([self.waypoints, self.waypoints[0]])
        
        print(f"Generated {len(self.waypoints)} waypoints")
    
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
