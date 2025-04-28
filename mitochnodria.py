import numpy as np
from typing import Optional, Tuple, List, Union
from abc import ABC, abstractmethod

class GeometryElement(ABC):
    """Abstract base class for geometric elements in the simulation."""
    
    @abstractmethod
    def get_position_at_distance(self, distance: float) -> np.ndarray:
        """Get position at a specified distance along the element."""
        pass
    
    @abstractmethod
    def distance_to_end(self, current_position: np.ndarray) -> float:
        """Calculate distance from current position to the end."""
        pass
    
    @property
    @abstractmethod
    def length(self) -> float:
        """Get the total length of the geometric element."""
        pass
    
    @property
    @abstractmethod
    def start_point(self) -> np.ndarray:
        """Get the starting point of the element."""
        pass
    
    @property
    @abstractmethod
    def end_point(self) -> np.ndarray:
        """Get the ending point of the element."""
        pass


class Edge(GeometryElement):
    def __init__(self, start_point: Tuple[float, float], end_point: Tuple[float, float], 
                 predecessor: Optional['GeometryElement'] = None, 
                 ancestor: Optional['GeometryElement'] = None,
                 color: str = 'blue'):
        """
        Initialize an Edge object representing a line segment in the simulation.
        
        Args:
            start_point: (x, y) coordinates of the starting point
            end_point: (x, y) coordinates of the ending point
            predecessor: The element that connects to this edge at its start point (optional)
            ancestor: The element that connects to this edge at its end point (optional)
            color: Color to use when drawing this edge (default: 'blue')
        """
        self._start_point = np.array(start_point, dtype=float)
        self._end_point = np.array(end_point, dtype=float)
        self.predecessor = predecessor
        self.ancestor = ancestor
        self.color = color
        
        # Compute derived properties
        self._compute_properties()
    
    def _compute_properties(self):
        """Compute length and angle from endpoints."""
        # Vector from start to end
        self.vector = self._end_point - self._start_point
        
        # Length of the edge
        self._length = np.linalg.norm(self.vector)
        
        # Angle in radians (measured counterclockwise from positive x-axis)
        self.angle = np.arctan2(self.vector[1], self.vector[0])
    
    @property
    def length(self) -> float:
        """Get the total length of the edge."""
        return self._length
    
    @property
    def start_point(self) -> np.ndarray:
        """Get the starting point of the edge."""
        return self._start_point
    
    @property
    def end_point(self) -> np.ndarray:
        """Get the ending point of the edge."""
        return self._end_point
    
    def get_position_at_distance(self, distance: float) -> np.ndarray:
        """
        Get the position coordinates at a specific distance along the edge.
        
        Args:
            distance: Distance from start point along the edge
            
        Returns:
            Array of (x, y) coordinates at the specified distance
        """
        # Ensure distance is within bounds
        distance = max(0, min(distance, self.length))
        
        # Calculate position
        position = self.start_point + (distance / self.length) * self.vector
        return position
    
    def distance_to_end(self, current_position: np.ndarray) -> float:
        """
        Calculate the distance from the current position to the end of the edge.
        
        Args:
            current_position: Current (x, y) position on the edge
            
        Returns:
            Distance to the end point
        """
        # Project the vector from current position to endpoint onto the edge direction
        to_end = self.end_point - current_position
        projection = np.dot(to_end, self.vector/self.length)
        return max(0, projection)  # Ensure non-negative
    
    def __repr__(self):
        """String representation of the Edge."""
        return f"Edge(start={tuple(self.start_point)}, end={tuple(self.end_point)}, length={self.length:.2f}, angle={np.degrees(self.angle):.2f}°)"
    



import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple

class Geometry:
    """
    A class representing the complete geometry of the mitochondria,
    consisting of connected edges.
    """
    
    def __init__(self, edges: List[Edge]):
        """
        Initialize the geometry with a list of edges.
        
        Args:
            edges: List of Edge objects that form the geometry
            
        Raises:
            ValueError: If the edges don't connect properly (end point of one edge
                        doesn't match the start point of the next edge)
        """
        self.edges = edges
        self._validate_and_connect_edges()
        self._compute_bounding_box()
        
    def _validate_and_connect_edges(self):
        """
        Validate that edges connect properly and assign predecessors and ancestors.
        """
        if not self.edges:
            return
        
        total_length = self.edges[0].length
        # Check connections and assign predecessor/ancestor relationships
        for i in range(len(self.edges) - 1):
            current_edge = self.edges[i]
            next_edge = self.edges[i + 1]
            
            # Check if the end point of current edge matches the start point of next edge
            if not np.allclose(current_edge.end_point, next_edge.start_point):
                raise ValueError(
                    f"Edges don't connect properly at index {i}. "
                    f"End point of edge {i}: {tuple(current_edge.end_point)}, "
                    f"Start point of edge {i+1}: {tuple(next_edge.start_point)}"
                )
            total_length += next_edge.length
            # Set predecessor and ancestor relationships
            next_edge.predecessor = current_edge
            current_edge.ancestor = next_edge

        self.total_length = total_length

    def _compute_bounding_box(self):
        """
        Compute the bounding box of the entire geometry.
        Sets min_x, max_x, min_y, max_y properties.
        """
        if not self.edges:
            self.min_x = self.max_x = self.min_y = self.max_y = 0
            return
            
        # Start with the first point
        all_points = []
        for edge in self.edges:
            all_points.append(edge.start_point)
            all_points.append(edge.end_point)
        
        all_points = np.array(all_points)
        self.min_x = np.min(all_points[:, 0])
        self.max_x = np.max(all_points[:, 0])
        self.min_y = np.min(all_points[:, 1])
        self.max_y = np.max(all_points[:, 1])
    
    def get_edge_at_position(self, position: np.ndarray) -> Optional[Edge]:
        """
        Find which edge contains the given position.
        
        Args:
            position: (x, y) coordinates to check
            
        Returns:
            The Edge containing the position, or None if position is not on any edge
        """
        for edge in self.edges:
            # Calculate distance from position to the line defined by edge
            start_to_pos = position - edge.start_point
            
            # Project this vector onto the edge direction
            edge_direction = edge.vector / edge.length
            projection_length = np.dot(start_to_pos, edge_direction)
            
            # Check if projection is within edge bounds (0 to edge.length)
            if 0 <= projection_length <= edge.length:
                # Calculate perpendicular distance to line
                perpendicular = start_to_pos - projection_length * edge_direction
                perp_distance = np.linalg.norm(perpendicular)
                
                # Check if position is close enough to the edge (using a small tolerance)
                if perp_distance < 1e-10:  # Adjust tolerance as needed
                    return edge
        
        return None
    

    
    def get_edge_at_length(self, distance: float) -> Tuple[Optional[Edge], float]:
        """
        Find which edge is at a specific distance from the start of the geometry,
        and the remaining distance along that edge.
        
        Args:
            distance: Distance from start of the geometry
            
        Returns:
            Tuple of (Edge at that distance, remaining distance along that edge).
            Returns (None, 0) if distance is negative or exceeds total length.
        """
        # Handle invalid distances
        if distance < 0 or not self.edges:
            return None, 0
            
        remaining_distance = distance
        cumulative_length = 0
        
        # Iterate through edges until we find the one containing our target distance
        for edge in self.edges:
            if remaining_distance <= edge.length:
                # We found the edge that contains our target distance
                return edge, remaining_distance
            
            # Move to the next edge
            remaining_distance -= edge.length
            cumulative_length += edge.length
        
        # If we get here, the distance exceeds the total length
        if distance >= cumulative_length:
            return None, 0
            
        return None, 0
    

    def draw(self, ax=None, edge_color='blue', vertex_color='red', 
             edge_width=1.5, vertex_size=20, show_vertices=False,
             show_labels=False):
        """
        Draw the geometry using matplotlib.
        
        Args:
            ax: Matplotlib axis object (creates a new one if None)
            edge_color: Color to draw the edges
            vertex_color: Color to draw the vertices
            edge_width: Line width for the edges
            vertex_size: Size of vertex markers
            show_vertices: Whether to show vertices
            show_direction: Whether to show directional arrows on edges
            direction_scale: Scale factor for direction arrows
            show_labels: Whether to show edge labels
            
        Returns:
            The matplotlib axis object with the plot
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot each edge
        for i, edge in enumerate(self.edges):
            # Draw the edge as a line

            # Use edge's specific color if available, otherwise use default
            edge_color = edge.color

            ax.plot([edge.start_point[0], edge.end_point[0]],
                    [edge.start_point[1], edge.end_point[1]],
                    color=edge_color, linewidth=edge_width)
            
            
            # Add edge label
            if show_labels:
                midpoint = edge.get_position_at_distance(edge.length / 2)
                ax.text(midpoint[0], midpoint[1], f"{i}", 
                        ha='center', va='center', backgroundcolor='white')
        
        # Plot vertices if requested
        if show_vertices:
            vertices = []
            for edge in self.edges:
                vertices.append(edge.start_point)
            # Add the last endpoint
            if self.edges:
                vertices.append(self.edges[-1].end_point)
            
            vertices = np.array(vertices)
            ax.scatter(vertices[:, 0], vertices[:, 1], 
                       color=vertex_color, s=vertex_size, zorder=10)
        
        # Set equal aspect ratio and reasonable limits
        margin = 0.1 * max(self.max_x - self.min_x, self.max_y - self.min_y)
        ax.set_xlim(self.min_x - margin, self.max_x + margin)
        ax.set_ylim(self.min_y - margin, self.max_y + margin)
        ax.set_aspect('equal')
        
        # Add labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Mitochondria Geometry')
        
        return ax
    
    def __repr__(self):
        """String representation of the Geometry."""
        return f"Geometry(edges={len(self.edges)}, total_length={self.get_total_length():.2f})"


    def map_displacements(self, displacements: np.ndarray, initial_distance: float = 0.0) -> np.ndarray:
        """
        Map 1D displacements to positions in the 2D geometry.
        
        Args:
            displacements: Array of 1D displacements along the geometry
            initial_distance: Starting distance along the geometry (default: 0)
            
        Returns:
            Array of 2D positions corresponding to each displacement
        """
        # Initialize arrays to store results
        total_length = self.total_length
        positions = np.zeros((len(displacements), 2))
        
        # Start from initial position
        current_distance = max(0, min(initial_distance, total_length))
        
        # Process each displacement
        for i, displacement in enumerate(displacements):
            # Update total distance with the displacement
            current_distance += displacement
            
            # Ensure we stay within geometry bounds
            current_distance = max(0, min(current_distance, total_length))
            
            # Find which edge we're on and how far along it
            current_edge, distance_on_edge = self.get_edge_at_length(current_distance)
            
            # Get the 2D position
            if current_edge:
                positions[i] = current_edge.get_position_at_distance(distance_on_edge)
            else:
                # If we somehow ended up with no edge (should not happen with proper bounds checking)
                if current_distance <= 0:
                    positions[i] = self.edges[0].start_point
                else:
                    positions[i] = self.edges[-1].end_point
        
        return positions
    


def draw_trajectory(positions: np.ndarray, ax=None, marker_size=10, 
                    connect_points=False, line_width=1, colormap='autumn', alpha=0.8,show_label=False):
    """
    Draw a trajectory of positions with a color gradient.
    
    Args:
        positions: Array of 2D positions to draw
        ax: Matplotlib axis object (creates a new one if None)
        marker_size: Size of position markers
        connect_points: Whether to connect points with lines
        line_width: Width of connecting lines
        colormap: Name of the colormap to use
        alpha: Transparency of the markers and lines
        
    Returns:
        The matplotlib axis object with the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get colormap
    cmap = plt.get_cmap(colormap)
    
    # Normalize positions for colormap (0 to 1)
    norm = plt.Normalize(0, len(positions) - 1)
    
    # Get colormap
    cmap = plt.get_cmap(colormap)
    
    # Create color array based on point indices
    colors = np.linspace(0, 1, len(positions))
    
    # Plot all points at once with color gradient
    scatter = ax.scatter(positions[:, 0], positions[:, 1], 
                        c=colors, cmap=colormap, s=marker_size, 
                        alpha=alpha, zorder=10)
    
    # Plot the last point
    if len(positions) > 0:
        ax.scatter(positions[-1, 0], positions[-1, 1], 
                  color=cmap(norm(len(positions) - 1)), s=marker_size, alpha=alpha, zorder=10)
    
    # Add a colorbar to show progression
    if len(positions) > 1 and show_label:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Time progression')
    
    return ax

from fbm import fgn


def disp_fbm(alpha : float,
                D : float,
                T: int, 
                deltaT : int = 1):
    ''' Generates normalized Fractional Gaussian noise. This means that, in 
    general:
    $$
    <x^2(t) > = 2Dt^{alpha}
    $$
                        
    and in particular:
    $$
    <x^2(t = 1)> = 2D 
    $$
    
    Parameters
    ----------
    alpha : float in [0,2]
        Anomalous exponent, for alpha = 1 -> Brownian motion
    D : float
        Diffusion coefficient
    T : int
        Number of displacements to generate
    deltaT : int, optional
        Sampling time
        
    Returns
    -------
    numpy.array
        Array containing T displacements of given parameters
    
    '''
    
    # Generate displacements
    disp = fgn(n=T, hurst=alpha/2, length=T, method="daviesharte")
    # Scale by D / deltaT
    # D is in px^2/s or nm^2/s, so multiply by deltaT and get a value in px^2 or nm^2, then take the square root to obtain a value in px or nm
    # MSD (<r^2 (dT)> scales as 2 * dim * D * dT, since we are simulating the two dimensions independently we set dim = 1 and take the sqrt
    disp *= np.sqrt(2*D*deltaT)        
    
    return disp