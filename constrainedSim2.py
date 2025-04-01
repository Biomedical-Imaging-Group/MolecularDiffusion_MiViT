import numpy as np
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import matplotlib.patches as pltpatches

class Shape(ABC):
    """
    Abstract base class for all shapes in the simulation.
    
    This class defines the interface that all shapes must implement,
    allowing for a consistent way to interact with different types of shapes.
    """
    
    @abstractmethod
    def is_inside(self, point):
        """
        Determines if a point is inside the shape.
        
        Parameters
        ----------
        point : tuple of (float, float)
            The (x, y) coordinates of the point to check.
            
        Returns
        -------
        bool
            True if the point is inside the shape, False otherwise.
        """
        pass
    
    @abstractmethod
    def reflect(self, point, velocity):
        """
        Reflects a particle that has hit the boundary of the shape.
        
        Parameters
        ----------
        point : tuple of (float, float)
            The (x, y) coordinates of the point to reflect.
        velocity : tuple of (float, float)
            The (vx, vy) velocity components of the particle.
            
        Returns
        -------
        tuple
            A tuple containing the new position and velocity after reflection.
        """
        pass
    
    @abstractmethod
    def draw(self, ax=None):
        """
        Draws the shape on a matplotlib axis.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axes on which to draw the shape. If None, a new figure and axes will be created.
            
        Returns
        -------
        matplotlib.axes.Axes
            The matplotlib axes with the drawn shape.
        """
        pass
    
    @staticmethod
    def draw_shapes(shapes, ax, max_dims = None):
        """
        Draws multiple shapes on a matplotlib axis.
        
        Parameters
        ----------
        shapes : list of Shape
            List of Shape objects to draw.
        ax : matplotlib.axes.Axes, optional
            The axes on which to draw the shapes. If None, a new figure and axes will be created.
        figsize : tuple of (float, float), optional
            Figure size in inches if a new figure is created. Defaults to (8, 8).
            
        Returns
        -------
        matplotlib.axes.Axes
            The matplotlib axes with the drawn shapes.
        """
        
        # Get the maximum dimension across all shapes
        max_dim = 0
        
        # Draw each shape
        for shape in shapes:
            shape.draw(ax)
            # Each shape should update max_dim in its draw method
            # but we'll also calculate it here for safety
            if hasattr(shape, 'get_max_dimension'):
                max_dim = max(max_dim, shape.get_max_dimension())
        
        # Add some padding to the max dimension
        max_dim += 1
        
        # Set equal axis scaling and limits

        if(max_dims == None):
            ax.set_xlim([-max_dim, max_dim])
            ax.set_ylim([-max_dim, max_dim])
        else:
            ax.set_xlim([max_dims[0], max_dims[1]])
            ax.set_ylim([max_dims[2], max_dims[3]]) 

        ax.set_aspect('equal')

        
        """
        label_dim = 5 * ( max_dim//5 )
        ax.set_xticks(np.arange(-int(label_dim), int(label_dim)+1,5))
        ax.set_yticks(np.arange(-int(label_dim), int(label_dim)+1,5))
        """

        return ax


class Rectangle(Shape):
    """
    A rectangle shape defined by its length, width, orientation, and center point.
    
    When angle = 0:
    - Width extends along the x-axis
    - Length extends along the y-axis
    """
    
    def __init__(self, length, width, angle, center_x=0, center_y=0):
        """
        Initialize a rectangle with the given parameters.
        
        Parameters
        ----------
        length : float
            The length of the rectangle (extends along the y-axis when angle = 0).
        width : float
            The width of the rectangle (extends along the x-axis when angle = 0).
        angle : float
            The orientation of the rectangle in radians. Positive angles represent
            counter-clockwise rotation (trigonometric convention).
        center_x : float, optional
            The x-coordinate of the rectangle's center. Defaults to 0.
        center_y : float, optional
            The y-coordinate of the rectangle's center. Defaults to 0.
        """
        self.length = length
        self.width = width
        self.angle = angle
        self.center_x = center_x
        self.center_y = center_y
        
        # Precompute sine and cosine for efficiency
        # Note: We use -angle because we're rotating the point in the opposite direction
        # of the rectangle's rotation to align with its local coordinate system
        self.cos_angle = np.cos(-angle)
        self.sin_angle = np.sin(-angle)


    def __eq__(self, other):
        """
        Check if this rectangle is equal to another rectangle.
        
        Two rectangles are considered equal if they have the same dimensions 
        (possibly swapped with corresponding angle adjustment) and center position.
        
        Parameters
        ----------
        other : Rectangle
            The rectangle to compare with
            
        Returns
        -------
        bool
            True if rectangles are equal, False otherwise
        """
        if not isinstance(other, Rectangle):
            return False
        
        # Tolerance for floating-point comparisons
        tolerance = 1e-6
        
        # Check if rectangles are the same with regular orientation
        same_regular = (
            abs(self.length - other.length) < tolerance and
            abs(self.width - other.width) < tolerance and
            abs(self.center_x - other.center_x) < tolerance and
            abs(self.center_y - other.center_y) < tolerance
        )
        
        # We need to account for the fact that angles are periodic
        angle_diff = abs((self.angle % (2 * np.pi)) - (other.angle % (2 * np.pi)))
        same_angle = angle_diff < tolerance or abs(angle_diff - 2 * np.pi) < tolerance
        
        # Check if rectangles are the same with swapped dimensions (rotated by 90°)
        same_swapped = (
            abs(self.length - other.width) < tolerance and
            abs(self.width - other.length) < tolerance and
            abs(self.center_x - other.center_x) < tolerance and
            abs(self.center_y - other.center_y) < tolerance
        )
        
        # If dimensions are swapped, the angles should differ by approximately ±90°
        swapped_angle_diff = abs((self.angle % (2 * np.pi)) - ((other.angle + np.pi/2) % (2 * np.pi)))
        swapped_same_angle = swapped_angle_diff < tolerance or abs(swapped_angle_diff - 2 * np.pi) < tolerance
        
        return (same_regular and same_angle) or (same_swapped and swapped_same_angle)
    

    def is_inside(self, point):
        """
        Determines if a point is inside the rectangle.
        
        Parameters
        ----------
        point : tuple of (float, float)
            The (x, y) coordinates of the point to check.
            
        Returns
        -------
        bool
            True if the point is inside the rectangle, False otherwise.
        """
        px, py = point
        
        # Translate point to origin relative to rectangle center
        px_translated = px - self.center_x
        py_translated = py - self.center_y
        
        # Rotate point to align with rectangle's local coordinate system
        rotated_x = px_translated * self.cos_angle - py_translated * self.sin_angle
        rotated_y = px_translated * self.sin_angle + py_translated * self.cos_angle
        
        # Check if the rotated point is within the rectangle boundaries
        half_length = self.length / 2
        half_width = self.width / 2
        
        return (
            -half_width <= rotated_x <= half_width and
            -half_length <= rotated_y <= half_length
        )
    
    def reflect(self, point, velocity):
        """
        Reflects a particle that has hit the boundary of the rectangle.
        
        Parameters
        ----------
        point : tuple of (float, float)
            The (x, y) coordinates of the point to reflect.
        velocity : tuple of (float, float)
            The (vx, vy) velocity components of the particle.
            
        Returns
        -------
        tuple
            A tuple containing the new position and velocity after reflection.
        """
        px, py = point
        vx, vy = velocity
        
        # Translate point to origin relative to rectangle center
        px_translated = px - self.center_x
        py_translated = py - self.center_y
        
        # Rotate point and velocity to align with rectangle's local coordinate system
        rotated_x = px_translated * self.cos_angle - py_translated * self.sin_angle
        rotated_y = px_translated * self.sin_angle + py_translated * self.cos_angle
        
        rotated_vx = vx * self.cos_angle - vy * self.sin_angle
        rotated_vy = vx * self.sin_angle + vy * self.cos_angle
        
        # Half dimensions
        half_length = self.length / 2
        half_width = self.width / 2
        
        # Check and reflect off each boundary
        new_x, new_y = rotated_x, rotated_y
        new_vx, new_vy = rotated_vx, rotated_vy
        
        # Reflect off vertical boundaries
        if rotated_x < -half_width:
            new_x = -half_width + (-half_width - rotated_x)
            new_vx = -rotated_vx
        elif rotated_x > half_width:
            new_x = half_width - (rotated_x - half_width)
            new_vx = -rotated_vx
            
        # Reflect off horizontal boundaries
        if rotated_y < -half_length:
            new_y = -half_length + (-half_length - rotated_y)
            new_vy = -rotated_vy
        elif rotated_y > half_length:
            new_y = half_length - (rotated_y - half_length)
            new_vy = -rotated_vy
        
        # Rotate back to global coordinates
        px_new = new_x * self.cos_angle + new_y * self.sin_angle + self.center_x
        py_new = -new_x * self.sin_angle + new_y * self.cos_angle + self.center_y
        
        vx_new = new_vx * self.cos_angle + new_vy * self.sin_angle
        vy_new = -new_vx * self.sin_angle + new_vy * self.cos_angle
        
        return (px_new, py_new), (vx_new, vy_new)
    
    def draw(self, ax=None):
        """
        Draws this rectangle on a matplotlib axis.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axes on which to draw the rectangle. If None, a new figure and axes will be created.
            
        Returns
        -------
        matplotlib.axes.Axes
            The matplotlib axes with the drawn rectangle.
        """
        # Create new figure and axes if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        
        # Calculate bottom-left corner coordinates (in local rectangle coordinates)
        local_bl_x = -self.width / 2
        local_bl_y = -self.length / 2
        
        # Rotate these coordinates (counter-clockwise) and translate to center
        cos_angle = np.cos(self.angle)
        sin_angle = np.sin(self.angle)
        
        bl_x = local_bl_x * cos_angle - local_bl_y * sin_angle + self.center_x
        bl_y = local_bl_x * sin_angle + local_bl_y * cos_angle + self.center_y
        
        # Create and add the matplotlib rectangle patch
        rect_patch = pltpatches.Rectangle(
            (bl_x, bl_y),
            self.width,
            self.length,
            angle=self.angle * 180 / np.pi,  # Convert to degrees
            edgecolor='red',
            facecolor='none',
            linewidth=2
        )
        ax.add_patch(rect_patch)
        
        return ax
    
    def get_max_dimension(self):
        """
        Returns the maximum distance from the center to any point on the rectangle.
        Useful for setting axis limits.
        
        Returns
        -------
        float
            The maximum dimension of the rectangle.
        """
        # Calculate the maximum radius from the center to any corner
        diagonal = np.sqrt(self.width**2 + self.length**2) / 2
        return max(abs(self.center_x) + diagonal, abs(self.center_y) + diagonal)


def animate_trajectory_with_shapes(trajectory, shapes, nframes, points_per_frame=None, fps=5, save_path=None, max_dims = None, figsize=(8,8)):
    """
    Creates an animation of a particle's trajectory inside multiple shapes, showing one subtrajectory per frame.
    
    Parameters
    ----------
    trajectory : np.ndarray of shape (N, 2)
        Each row represents the (x, y) coordinates of the particle.
    shapes : list of Shape
        List of Shape objects defining the boundaries.
    nframes : int
        Number of frames to divide the trajectory into.
    points_per_frame : int, optional
        Number of points to show in each frame. If None, calculated from trajectory length and nframes.
    fps : int, optional
        Frames per second for the animation. Defaults to 5.
    save_path : str, optional
        Path to save the animation (supports .mp4 and .gif).
        
    Returns
    -------
    IPython.display.HTML
        HTML animation that can be displayed in a Jupyter notebook.
    """
    from matplotlib import animation
    from IPython.display import HTML
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Draw all shapes
    Shape.draw_shapes(shapes, ax=ax, max_dims=max_dims)
    
    if points_per_frame is None:
        points_per_frame = len(trajectory) // nframes
    
    
    line, = ax.plot([], [], lw=2, color='blue')
    frame_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12)
    x_data, y_data = [], []
    
    def init():
        line.set_data([], [])
        frame_text.set_text('')
        return line, frame_text
    
    def update(frame):
        start = frame * points_per_frame
        end = min(start + points_per_frame, len(trajectory))
        current_segment = trajectory[start:end]
        
        if frame == 0:
            x_data.clear()
            y_data.clear()
        
        x_data.extend(current_segment[:, 0])
        y_data.extend(current_segment[:, 1])
        
        line.set_data(x_data, y_data)
        frame_text.set_text(f'Frame {frame + 1}/{nframes}')
        
        return line, frame_text
    
    ani = animation.FuncAnimation(fig, update, frames=nframes,
                                 init_func=init, blit=True, interval=1000/fps)
    
    if save_path:
        if save_path.endswith('.mp4'):
            writer = animation.FFMpegWriter(fps=fps, metadata=dict(artist='Me'), bitrate=1800)
            ani.save(save_path, writer=writer)
        elif save_path.endswith('.gif'):
            writer = animation.PillowWriter(fps=fps)
            ani.save(save_path, writer=writer)
    
    plt.close()
    display(HTML(ani.to_jshtml()))
    return 
def animate_trajectory_with_shapes2(trajectory, shapes, nframes, points_per_frame=None, fps=5, save_path=None, max_dims=None, figsize=(8,8)):
    """
    Creates an animation of a particle's trajectory inside multiple shapes, showing only the current positions in each frame.
    
    Parameters
    ----------
    trajectory : np.ndarray of shape (N, 2)
        Each row represents the (x, y) coordinates of the particle.
    shapes : list of Shape
        List of Shape objects defining the boundaries.
    nframes : int
        Number of frames to divide the trajectory into.
    points_per_frame : int, optional
        Number of points to show in each frame. If None, calculated from trajectory length and nframes.
    fps : int, optional
        Frames per second for the animation. Defaults to 5.
    save_path : str, optional
        Path to save the animation (supports .mp4 and .gif).
    max_dims : tuple, optional
        Maximum dimensions for the plot axes.
    figsize : tuple, optional
        Figure size in inches. Defaults to (8, 8).
        
    Returns
    -------
    IPython.display.HTML
        HTML animation that can be displayed in a Jupyter notebook.
    """
    from matplotlib import animation
    from IPython.display import HTML
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Draw all shapes
    Shape.draw_shapes(shapes, ax=ax, max_dims=max_dims)
    
    if points_per_frame is None:
        points_per_frame = len(trajectory) // nframes
    
    # We'll use a scatter plot for current positions
    scatter = ax.scatter([], [], s=5, color='blue')
    frame_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12)
    
    def init():
        scatter.set_offsets(np.empty((0, 2)))
        frame_text.set_text('')
        return scatter, frame_text
    
    def update(frame):
        start = frame * points_per_frame
        end = min(start + points_per_frame, len(trajectory))
        current_segment = trajectory[start:end]
        
        # Update scatter plot with only the current positions
        scatter.set_offsets(current_segment)
        frame_text.set_text(f'Frame {frame + 1}/{nframes}')
        
        return scatter, frame_text
    
    ani = animation.FuncAnimation(fig, update, frames=nframes,
                                 init_func=init, blit=True, interval=1000/fps)
    
    if save_path:
        if save_path.endswith('.mp4'):
            writer = animation.FFMpegWriter(fps=fps, metadata=dict(artist='Me'), bitrate=1800)
            ani.save(save_path, writer=writer)
        elif save_path.endswith('.gif'):
            writer = animation.PillowWriter(fps=fps)
            ani.save(save_path, writer=writer)
    
    plt.close()
    display(HTML(ani.to_jshtml()))
    return ani






from stochastic.processes.noise import FractionalGaussianNoise as FGN


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
    disp = FGN(hurst = alpha/2).sample(n = T)
    # Normalization factor
    disp *= np.sqrt(T)**(alpha)
    # Add D
    disp *= np.sqrt(2*D*deltaT)        
    
    return disp





def traj_in_rectangles(T, D, alpha, deltaT, rectangles, startPos=(0, 0)):
    """
    Simulates the trajectory of a single particle undergoing fractional Brownian motion 
    within a constrained rectangular region with an arbitrary angle.

    Parameters
    ----------
    T : int
        Number of time steps for the simulation.
    D : float
        Diffusion coefficient controlling the magnitude of displacements.
    alpha : float in [0,2]
        Anomalous exponent defining the nature of the motion 
        (alpha = 1 corresponds to standard Brownian motion).
    deltaT : int
        Sampling time interval between steps.
    rect : list of Rectangles

    startPos : tuple of (float, float), optional
        Initial position of the particle within the rectangle. Defaults to (0,0).

    Returns
    -------
    numpy.ndarray
        Array of shape (T, 2) containing the (x, y) positions of the particle at each step.
    """

    current_rect = None

    for rect in rectangles:
        if(rect.is_inside(startPos)):
            current_rect = rect
    print("Current rect:", current_rect.angle)

    if(current_rect == None):
        raise ValueError("startPos is inside no rectangle")

    # Generate displacements
    disp_x = disp_fbm(alpha, D, T, deltaT)
    disp_y = disp_fbm(alpha, D, T, deltaT)


    
    # Initialize position array in rotated frame
    positions = np.zeros((T, 2))
    positions[0] = startPos
    
    # Simulate trajectory with reflective boundaries
    for t in range(1, T):
        new_x = positions[t - 1, 0] + disp_x[t]
        new_y = positions[t - 1, 1] + disp_y[t]
        
        new_pos = (new_x, new_y)

        if(not(current_rect.is_inside(new_pos))):
            # search for other rectangle the particle could be in
            new_rect = None

            for rect in rectangles:
                if(rect.is_inside(new_pos) and not(rect == current_rect)):
                    new_rect = rect
            
            if(new_rect != None):
                current_rect = new_rect
            else:
                # reflect position on old shape
                # added 0,0 for velocity (not used yet)
                new_pos, _ = current_rect.reflect(new_pos, (0,0))

        positions[t] = new_pos
    
    return positions