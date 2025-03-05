import numpy as np
import matplotlib as plt


def plot1ParticleTrajectory(trajectory, nframes, D):
    """
    Plots the trajectory of a particle, coloring each frame differently 
    and labeling each frame with its number.
    
    Parameters:
    - trajectory: np.ndarray of shape (N, 2), where N is the total number of points.
                  Each row represents the (x, y) coordinates of the particle.
    - nframes: int, number of frames to divide the trajectory into.
    - D: float, diffusion coefficient for annotation.
    """
    plt.figure(figsize=(6, 6))
    
    # Calculate points per frame
    points_per_frame = len(trajectory) // nframes
    
    # Plot trajectory segments with frame labels
    for f in range(nframes):
        start = f * points_per_frame
        end = (f + 1) * points_per_frame + (1 if f != nframes - 1 else 0)
        
        # Plot each frame's trajectory in a different color
        plt.plot(
            trajectory[start:end, 0], 
            -trajectory[start:end, 1], 
            lw=1, 
            label=f'Frame {f + 1}'  # Frames start from 1
        )
    
    # Add legend and axis labels
    plt.legend(loc="best", fontsize=8)
    plt.title(f'Brownian Motion of 1 Particle with $D={D}$ (nm)$^2$/s on 4 Frames')
    plt.xlabel('X Position (nm)', fontsize=14)  # Increased font size
    plt.ylabel('Y Position (nm)', fontsize=14)  # Increased font size
    plt.grid(True)
    plt.axis('equal')  # Equal scaling for x and y axes
        # Increase the tick label size
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.tick_params(axis='both', which='minor', labelsize=12)
    # Show the plot
    plt.tight_layout()
    plt.show()


def show_plt(plt, title, xlabel='', ylabel='',legend=False):
    """
    A helper function to display plots with a uniform style and labeling.

    Parameters:
    - plt (matplotlib.pyplot): The matplotlib.pyplot module, used for plotting.
    - title (str): Title of the plot.
    - xlabel (str, optional): Label for the x-axis.
    - ylabel (str, optional): Label for the y-axis.

    Displays:
    - A styled plot with grid, labels, and title.
    """
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    if(legend):
        plt.legend()  # Uncomment if there are multiple series to label
    plt.show()