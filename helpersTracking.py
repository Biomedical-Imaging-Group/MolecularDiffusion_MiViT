import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from IPython.display import HTML, display


def detect_particles(image, sigma1=1.0, sigma2=2.0, threshold_percentage=0.1, min_distance=3):
    """
    Detect particles in a microscopy image using Difference of Gaussians and local maxima.
    
    Parameters:
    -----------
    image : 2D numpy array
        The input microscopy image.
    sigma1 : float
        Standard deviation for the first Gaussian filter.
    sigma2 : float
        Standard deviation for the second Gaussian filter (should be larger than sigma1).
    threshold_abs : float or None
        Minimum intensity threshold for detected peaks. If None, it will be
        automatically estimated as a fraction of the image's maximum intensity.
    min_distance : int
        Minimum number of pixels separating peaks.
    
    Returns:
    --------
    coordinates : numpy array
        Array of shape (n_particles, 2) containing the y, x coordinates of detected particles.
    filtered_image : 2D numpy array
        The DoG filtered image.
    """
    # Apply Gaussian filters
    gaussian1 = ndimage.gaussian_filter(image, sigma=sigma1)
    gaussian2 = ndimage.gaussian_filter(image, sigma=sigma2)
    
    # Compute the Difference of Gaussians
    dog_image = gaussian1 - gaussian2
    

    threshold_abs = threshold_percentage * np.max(dog_image)
    
    # Find local maxima
    coordinates = peak_local_max(
        dog_image, 
        min_distance=min_distance,
        threshold_abs=threshold_abs,
        exclude_border=False
    )
    
    return coordinates, dog_image



import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def visualize_dog_detection(original_image, dog_image, coordinates, figsize=(16, 5)):
    """
    Visualize the original image, DoG filtered image, and detected particles.
    
    Parameters:
    -----------
    original_image : 2D numpy array
        The original microscopy image.
    dog_image : 2D numpy array
        The DoG filtered image.
    coordinates : numpy array
        Array of shape (n_particles, 2) containing the y, x coordinates of detected particles.
    figsize : tuple
        Figure size for the visualization.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Original image
    axes[0].imshow(original_image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # DoG filtered image
    im_dog = axes[1].imshow(dog_image, cmap='viridis')
    axes[1].set_title('DoG Filtered Image')
    axes[1].axis('off')
    plt.colorbar(im_dog, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Original image with detected particles overlaid
    axes[2].imshow(original_image, cmap='gray')
    axes[2].set_title(f'Detected Particles ({len(coordinates)})')
    
    # Add circles for each detected particle
    for y, x in coordinates:
        circle = Circle((x, y), radius=3, color='red', fill=False, linewidth=1.5)
        axes[2].add_patch(circle)
    
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Also plot the histogram of the DoG image to help with threshold selection
    plt.figure(figsize=(8, 4))
    plt.hist(dog_image.ravel(), bins=100)
    plt.title('Histogram of DoG Image Values')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()


    import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import pandas as pd



def link_particles(coords_t0, coords_t1, max_distance=15):
    """
    Link particles between two consecutive frames based on proximity.
    
    Parameters:
    -----------
    coords_t0 : numpy array
        Array of shape (n_particles, 2) with y, x coordinates at time t.
    coords_t1 : numpy array
        Array of shape (n_particles, 2) with y, x coordinates at time t+1.
    max_distance : float
        Maximum distance allowed for linking particles between frames.
        
    Returns:
    --------
    links : list of tuples
        Each tuple contains (idx_t0, idx_t1) for linked particles.
    unlinked_t0 : list
        Indices of particles in coords_t0 that weren't linked.
    unlinked_t1 : list
        Indices of particles in coords_t1 that weren't linked.
    """
    # If either frame has no particles, return empty links
    if len(coords_t0) == 0 or len(coords_t1) == 0:
        unlinked_t0 = list(range(len(coords_t0)))
        unlinked_t1 = list(range(len(coords_t1)))
        return [], unlinked_t0, unlinked_t1
    
    # Calculate pairwise distances between all particles
    cost_matrix = np.zeros((len(coords_t0), len(coords_t1)))
    
    for i, pos0 in enumerate(coords_t0):
        for j, pos1 in enumerate(coords_t1):
            distance = np.sqrt(((pos0 - pos1)**2).sum())
            cost_matrix[i, j] = distance
    
    # Create a mask for distances exceeding max_distance
    mask = cost_matrix > max_distance
    
    # Solve the linear assignment problem
    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=False)
    
    # Filter out assignments that exceed max_distance
    links = []
    unlinked_t0 = list(range(len(coords_t0)))
    unlinked_t1 = list(range(len(coords_t1)))
    
    for i, j in zip(row_ind, col_ind):
        if cost_matrix[i, j] <= max_distance:
            links.append((i, j))
            if i in unlinked_t0:
                unlinked_t0.remove(i)
            if j in unlinked_t1:
                unlinked_t1.remove(j)
    
    return links, unlinked_t0, unlinked_t1

def track_particles(image_sequence, sigma1=1.0, sigma2=2.0, threshold_percentage=0.1, 
                   min_distance=3, max_linking_distance=15, min_track_length=3, verbose=False):
    """
    Track particles across a sequence of microscopy images.
    
    Parameters:
    -----------
    image_sequence : list or array
        List of 2D arrays representing the image sequence.
    sigma1, sigma2 : float
        Standard deviations for DoG filtering.
    threshold_percentage : float
        Fraction of maximum intensity for thresholding.
    min_distance : int
        Minimum separation between particles.
    max_linking_distance : float
        Maximum distance for linking particles between frames.
    min_track_length : int
        Minimum number of frames a particle must be present to be considered a valid track.
        
    Returns:
    --------
    tracks : dict
        Dictionary where keys are track IDs and values are lists of (frame, y, x) coordinates.
    all_detections : pandas DataFrame
        DataFrame containing all detections with frame, y, x, and track_id columns.
    """
    # Detect particles in each frame
    all_coordinates = []
    all_filtered_images = []
    if(verbose):
        print(f"Detecting particles in {len(image_sequence)} frames...")
    for frame_idx, image in enumerate(image_sequence):
        coords, filtered = detect_particles(
            image, 
            sigma1=sigma1, 
            sigma2=sigma2, 
            threshold_percentage=threshold_percentage,
            min_distance=min_distance
        )
        all_coordinates.append(coords)
        all_filtered_images.append(filtered)
        if(verbose):
            print(f"Frame {frame_idx}: {len(coords)} particles detected")

    # Initialize tracks
    tracks = {}
    next_track_id = 0
    active_tracks = {}  # track_id -> last position and frame
    
    # Initialize detections DataFrame
    detections = []
    
    # Process first frame: create a new track for each detected particle
    for i, pos in enumerate(all_coordinates[0]):
        track_id = next_track_id
        tracks[track_id] = [(0, pos[0], pos[1])]
        active_tracks[track_id] = (pos, 0)
        detections.append({'frame': 0, 'y': pos[0], 'x': pos[1], 'track_id': track_id})
        next_track_id += 1
    if(verbose):
        print(f"Frame 0: Created {len(all_coordinates[0])} new tracks")
    
    # Link subsequent frames
    for frame_idx in range(1, len(image_sequence)):
        coords_prev = np.array([active_tracks[track_id][0] for track_id in active_tracks])
        track_ids = list(active_tracks.keys())
        coords_current = all_coordinates[frame_idx]
        
        if len(coords_prev) > 0 and len(coords_current) > 0:
            # Link particles between frames
            links, unlinked_prev, unlinked_current = link_particles(
                coords_prev, coords_current, max_distance=max_linking_distance
            )
            
            # Update existing tracks
            for prev_idx, current_idx in links:
                track_id = track_ids[prev_idx]
                pos = coords_current[current_idx]
                tracks[track_id].append((frame_idx, pos[0], pos[1]))
                active_tracks[track_id] = (pos, frame_idx)
                detections.append({
                    'frame': frame_idx, 
                    'y': pos[0], 
                    'x': pos[1], 
                    'track_id': track_id
                })
            
            # Create new tracks for unlinked particles in current frame
            for idx in unlinked_current:
                pos = coords_current[idx]
                track_id = next_track_id
                tracks[track_id] = [(frame_idx, pos[0], pos[1])]
                active_tracks[track_id] = (pos, frame_idx)
                detections.append({
                    'frame': frame_idx, 
                    'y': pos[0], 
                    'x': pos[1], 
                    'track_id': track_id
                })
                next_track_id += 1
        
        elif len(coords_current) > 0:
            # If no active tracks but particles detected in current frame,
            # create new tracks for all current particles
            for idx, pos in enumerate(coords_current):
                track_id = next_track_id
                tracks[track_id] = [(frame_idx, pos[0], pos[1])]
                active_tracks[track_id] = (pos, frame_idx)
                detections.append({
                    'frame': frame_idx, 
                    'y': pos[0], 
                    'x': pos[1], 
                    'track_id': track_id
                })
                next_track_id += 1
        
        # Remove tracks that weren't updated in this frame
        tracks_to_remove = []
        for track_id, (_, last_frame) in active_tracks.items():
            if last_frame < frame_idx:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del active_tracks[track_id]

        if(verbose):
            print(f"Frame {frame_idx}: {len(links)} links, {len(unlinked_current)} new tracks")
    
    # Convert detections to DataFrame
    all_detections = pd.DataFrame(detections)
    
    # Filter tracks by length
    long_tracks = {k: v for k, v in tracks.items() if len(v) >= min_track_length}
    
    print(f"Tracking complete: {len(tracks)} total tracks, {len(long_tracks)} tracks with ≥{min_track_length} frames")
    
    return long_tracks, all_detections, all_filtered_images

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML, display


def visualize_tracks(image_sequence, tracks, fps=5, figsize=(12, 12), save_path=None):
    """
    Visualize particle tracks as an animation in a Jupyter notebook.

    Parameters:
    -----------
    image_sequence : list or array
        List of 2D arrays representing the image sequence.
    tracks : dict
        Dictionary where keys are track IDs and values are lists of (frame, y, x) positions.
    fps : int
        Frames per second for the animation.
    figsize : tuple
        Figure size (width, height) in inches.
    save_path : str or None
        If provided, save the animation to this path (.mp4 or .gif).
    """
    num_frames = len(image_sequence)

    track_ids = list(tracks.keys())
    num_tracks = len(track_ids)
    cmap = plt.cm.get_cmap('hsv', num_tracks)
    track_colors = {track_id: cmap(i) for i, track_id in enumerate(track_ids)}

    # Set up figure
    fig, ax = plt.subplots(figsize=figsize)

    plt.axis("off")
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

    im = ax.imshow(image_sequence[0], cmap='gray', animated=True)
    scat = ax.scatter([], [], s=30)
    text_annotations = []

    def init():
        im.set_data(image_sequence[0])
        scat.set_offsets(np.empty((0, 2)))
        scat.set_color([])
        for txt in text_annotations:
            txt.remove()
        text_annotations.clear()
        return [im, scat]

    def update(frame):
        im.set_data(image_sequence[frame])
        dots = []
        colors = []

        # Clear old text annotations
        for txt in text_annotations:
            txt.remove()
        text_annotations.clear()

        for track_id, positions in tracks.items():
            for f, y, x in positions:
                if f == frame:
                    dots.append([x, y])
                    colors.append(track_colors[track_id])
                    # Add track ID label next to dot
                    txt = ax.text(x + 2, y + 2, str(track_id), color=track_colors[track_id],
                                  fontsize=10, weight='bold', animated=True)
                    text_annotations.append(txt)

        if dots:
            scat.set_offsets(np.array(dots))
            scat.set_color(colors)
        else:
            scat.set_offsets(np.empty((0, 2)))
            scat.set_color([])

        return [im, scat] + text_annotations

    ani = animation.FuncAnimation(fig, update, frames=num_frames, init_func=init,
                                  blit=True, interval=1000 / fps)

    if save_path:
        if save_path.endswith('.gif'):
            ani.save(save_path, writer='pillow', fps=fps)
        elif save_path.endswith('.mp4'):
            ani.save(save_path, writer='ffmpeg', fps=fps)
        else:
            raise ValueError("Unsupported file format. Use '.gif' or '.mp4'.")
        plt.close(fig)
    else:
        plt.close(fig)

        html = HTML(ani.to_jshtml())
        display(html)
        return 



# Example usage function for the entire workflow
def analyze_microscopy_sequence(image_sequence, 
                               sigma1=1.0, 
                               sigma2=2.0, 
                               threshold_percentage=0.1,
                               min_distance=3,
                               max_linking_distance=15,
                               min_track_length=3,
                               visualize=True,
                               verbose=False,
                               output_prefix=None):
    """
    Complete workflow for analyzing a microscopy image sequence.
    
    Parameters:
    -----------
    image_sequence : list or array
        List of 2D arrays representing the image sequence.
    sigma1, sigma2 : float
        Standard deviations for DoG filtering.
    threshold_percentage : float
        Fraction of maximum intensity for thresholding.
    min_distance : int
        Minimum separation between particles.
    max_linking_distance : float
        Maximum distance for linking particles between frames.
    min_track_length : int
        Minimum number of frames a particle must be present to be considered a valid track.
    visualize : bool
        Whether to visualize the tracks.
    output_prefix : str or None
        Prefix for output files. If None, no files are saved.
        
    Returns:
    --------
    tracks : dict
        Dictionary of particle tracks.
    all_detections : pandas DataFrame
        DataFrame of all particle detections.
    filtered_images : list
        List of DoG-filtered images.
    """
    # Track particles across the sequence
    tracks, all_detections, filtered_images = track_particles(
        image_sequence,
        sigma1=sigma1,
        sigma2=sigma2,
        threshold_percentage=threshold_percentage,
        min_distance=min_distance,
        max_linking_distance=max_linking_distance,
        min_track_length=min_track_length,
        verbose=verbose
    )
    
    # Visualize tracks if requested
    if visualize:
        visualize_tracks(
            image_sequence,
            tracks,
            save_path=output_prefix + "_tracks" if output_prefix else None,
            figsize=(15,5)
        )
    
    # Save results if output_prefix is provided
    if output_prefix:
        # Save detections CSV
        all_detections.to_csv(f"{output_prefix}_detections.csv", index=False)
        
        # Save tracks as a pickle file for later analysis
        import pickle
        with open(f"{output_prefix}_tracks.pkl", 'wb') as f:
            pickle.dump(tracks, f)
        
        print(f"Results saved with prefix: {output_prefix}")
    
    return tracks, all_detections, filtered_images