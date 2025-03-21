import numpy as np
import matplotlib as plt
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from IPython.display import HTML
import numpy as np

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


def play_video(video, figsize=(5, 5), fps=5, vmin=None, vmax=None, save_path=None):
    """
    Displays a stack of images as a video inside jupyter notebooks with consistent intensity scaling.

    Parameters
    ----------
    video : ndarray
        Stack of images.
    figsize : tuple, optional
        Canvas size of the video.
    fps : int, optional
        Video frame rate.
    vmin : float, optional
        Minimum intensity value for all frames. If None, will be automatically determined.
    vmax : float, optional
        Maximum intensity value for all frames. If None, will be automatically determined.

    Returns
    -------
    Video object
        Returns a video player with input stack of images.
    """
    fig = plt.figure(figsize=figsize)
    images = []

    if(len(video.shape) == 3):
        video = np.expand_dims(video,axis=-1)

    plt.axis("off")
    
    # If vmin/vmax not provided, compute global min/max across all frames
    if vmin is None:
        vmin = np.min([frame[:, :, 0].min() for frame in video])
    if vmax is None:
        vmax = np.max([frame[:, :, 0].max() for frame in video])
    mean = np.mean(video)

    print(f"vmin: {vmin} vmax: {vmax} mean: {mean:.2f}")

    for image in video:
        images.append([plt.imshow(image[:, :, 0], cmap="gray", vmin=vmin, vmax=vmax)])

    anim = animation.ArtistAnimation(
        fig, images, interval=1e3 / fps, blit=True, repeat_delay=0
    )

    html = HTML(anim.to_jshtml())
    display(html)

    # Save the animation if a save path is provided
    if save_path:
        if save_path.endswith('.mp4'):
            # Use FFMpegWriter for MP4 files (requires FFmpeg installed)
            writer = animation.FFMpegWriter(fps=fps, metadata=dict(artist='Me'), bitrate=1800)
            anim.save(save_path, writer=writer)
            print(f"Animation saved to {save_path}")
        elif save_path.endswith('.gif'):
            # Use PillowWriter for GIF files
            writer = animation.PillowWriter(fps=fps)
            anim.save(save_path, writer=writer)
            print(f"Animation saved to {save_path}")
        else:
            print("Unsupported file format. Use .mp4 or .gif extension.")
    plt.close()