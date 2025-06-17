import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from IPython.display import HTML, display
import seaborn as sns




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
    
    # Re-index tracks to have sequential IDs
    reindexed_tracks = {}
    all_detections_copy = all_detections.copy()  # Create a copy to avoid modifying the original

    # Create a mapping from old to new track IDs
    track_id_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted(long_tracks.keys()))}

    # Re-index the long_tracks dictionary
    for old_id, new_id in track_id_mapping.items():
        reindexed_tracks[new_id] = long_tracks[old_id]

    # Update track IDs in the all_detections DataFrame
    all_detections_copy.loc[all_detections_copy['track_id'].isin(long_tracks.keys()), 'track_id'] = \
        all_detections_copy.loc[all_detections_copy['track_id'].isin(long_tracks.keys()), 'track_id'].map(track_id_mapping)

    # Replace the original variables
    long_tracks = reindexed_tracks
    all_detections = all_detections_copy

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


def extract_particle_patches(image_3d, tracks, patch_size=7):
    """
    Extract square patches around each particle's position in each frame.

    Parameters:
    -----------
    image_3d : np.ndarray
        3D array of shape (num_frames, height, width).
    tracks : dict
        Dictionary where keys are track IDs and values are lists of (frame, y, x) positions.
    patch_size : int
        Size of the square patch (must be odd).

    Returns:
    --------
    patches : dict
        Dictionary where each key is a track ID and value is a list of 2D numpy arrays
        of shape (patch_size, patch_size), one per position in the track.
    """
    assert patch_size % 2 == 1, "patch_size must be an odd number"
    half = patch_size // 2
    num_frames, height, width = image_3d.shape

    patches = {}

    for track_id, positions in tracks.items():
        track_patches = []
        for frame, y, x in positions:
            
            y, x = int(round(y)), int(round(x))
            # Pad the image if the patch would go out of bounds
            padded = np.pad(image_3d[frame], pad_width=half, mode='constant')
            y_p, x_p = y + half, x + half  # shift coords due to padding
            patch = padded[y_p - half:y_p + half + 1, x_p - half:x_p + half + 1]
            track_patches.append(patch)
        patches[track_id] = np.array(track_patches)

    return patches
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

def add_refined_localization_to_dataframe(df_tracks, tracks, patches, patch_size):
    """
    Add sub-pixel localization, PSF size (sigma), and max intensity per frame to DataFrame.
    """
    def symmetrical_gaussian(coords, amplitude, xo, yo, sigma, offset):
        x, y = coords
        g = offset + amplitude * np.exp(
            -(((x - xo)**2 + (y - yo)**2) / (2 * sigma**2))
        )
        return g.ravel()

    half = patch_size // 2

    x_refined_list = []
    y_refined_list = []
    sigma_list = []
    intensity_list = []

    for track_id, original_positions in tracks.items():
        track_patches = patches[track_id]
        for i, (frame, y_int, x_int) in enumerate(original_positions):
            patch = track_patches[i]
            x = np.arange(patch.shape[1])
            y = np.arange(patch.shape[0])
            x, y = np.meshgrid(x, y)

            initial_guess = (patch.max(), half, half, 1.0, patch.min())

            try:
                popt, _ = curve_fit(symmetrical_gaussian, (x, y), patch.ravel(), p0=initial_guess)
                _, x0, y0, sigma, _ = popt
                refined_x = x_int - half + x0
                refined_y = y_int - half + y0
            except RuntimeError:
                print(f"particle {track_id} could not be located on frame {frame}")
                refined_x = x_int
                refined_y = y_int
                sigma = 10

            key = (track_id, frame)
            x_refined_list.append((key, refined_x))
            y_refined_list.append((key, refined_y))
            sigma_list.append((key, sigma))
            intensity_list.append((key, patch.max()))

    df_tracks['x_refined'] = pd.Series(dict(x_refined_list))
    df_tracks['y_refined'] = pd.Series(dict(y_refined_list))
    df_tracks['psf_size'] = pd.Series(dict(sigma_list))
    df_tracks['max_intensity'] = pd.Series(dict(intensity_list))

    return df_tracks


def compute_displacement(df_tracks):
    """
    Compute displacement, mean displacement, mean PSF size, and max intensity over track.
    """
    displacements = []
    mean_displacements = {}
    mean_psf_sizes = {}
    max_intensities = {}
    mean_intensities = {}
    std_intensities = {}

    df_tracks_reset = df_tracks.reset_index()

    for track_id, group in df_tracks_reset.groupby('track_id'):
        group = group.sort_values('frame')
        track_displacements = []

        for i in range(1, len(group)):
            x1, y1 = group.iloc[i-1][['x_refined', 'y_refined']]
            x2, y2 = group.iloc[i][['x_refined', 'y_refined']]
            displacement = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            track_displacements.append(displacement)

        track_displacements.insert(0, 0)
        df_tracks_reset.loc[group.index, 'displacement'] = track_displacements

        mean_displacements[track_id] = np.mean(track_displacements)
        mean_psf_sizes[track_id] = group['psf_size'].mean()
        max_intensities[track_id] = group['max_intensity'].max()
        mean_intensities[track_id] = group['max_intensity'].mean()
        std_intensities[track_id] = group['max_intensity'].std()

    df_tracks_reset.set_index(['track_id', 'frame'], inplace=True)
    df_tracks_reset['mean_displacement'] = df_tracks_reset.index.get_level_values('track_id').map(mean_displacements)
    df_tracks_reset['mean_psf_size'] = df_tracks_reset.index.get_level_values('track_id').map(mean_psf_sizes)
    df_tracks_reset['max_intensity_over_track'] = df_tracks_reset.index.get_level_values('track_id').map(max_intensities)
    df_tracks_reset['mean_max_intensity_over_track'] = df_tracks_reset.index.get_level_values('track_id').map(mean_intensities)
    df_tracks_reset['std_max_intensity_over_track'] = df_tracks_reset.index.get_level_values('track_id').map(std_intensities)

    return df_tracks_reset





def tracks_to_dataframe(tracks, patches, patch_size):
    """
    Convert tracks dictionary into a pandas DataFrame indexed by (track_id, frame)
    and containing x and y positions.

    Parameters:
    -----------
    tracks : dict
        Dictionary where keys are track IDs and values are lists of (frame, y, x) positions.

    Returns:
    --------
    df : pandas.DataFrame
        DataFrame indexed by (track_id, frame) with columns 'x' and 'y'.
    """
    data = []
    for track_id, positions in tracks.items():
        nbr_frames = len(positions)
        for frame, y, x in positions:
            data.append((track_id, frame,nbr_frames, x, y))
    
    df = pd.DataFrame(data, columns=['track_id', 'frame', 'nbr_frames','x', 'y'])
    df.set_index(['track_id', 'frame'], inplace=True)
    df.sort_index(inplace=True)  # Optional: to have it ordered by track and frame

    df = add_refined_localization_to_dataframe(df, tracks, patches, patch_size=patch_size)
    df = compute_displacement(df)

    return df


import matplotlib.pyplot as plt
import pandas as pd

def plot_comparison_with_std(df_under, df_over, columns):
    """
    Plot mean ± std for selected columns from two DataFrames (under and over threshold).

    Parameters:
    -----------
    df_under : pd.DataFrame
        DataFrame for tracks under the threshold (multi-indexed).
    df_over : pd.DataFrame
        DataFrame for tracks over the threshold (multi-indexed).
    columns : list of str
        List of column names to include in the plot.
    scale : bool
        If True, values are scaled using StandardScaler before plotting.
    """
    # Aggregate one row per track
    df_under_grouped = df_under.groupby(level=0).first()[columns]
    df_over_grouped = df_over.groupby(level=0).first()[columns]

    df_under_scaled = df_under_grouped
    df_over_scaled = df_over_grouped

    # Compute mean and std
    means = pd.DataFrame({
        'Under 3': df_under_scaled.mean(),
        'Over 3': df_over_scaled.mean()
    })

    stds = pd.DataFrame({
        'Under 3': df_under_scaled.std(),
        'Over 3': df_over_scaled.std()
    })

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    width = 0.35
    x = range(len(columns))

    ax.bar([i - width/2 for i in x], means['Under 3'], width, yerr=stds['Under 3'], label='Under 3', capsize=4)
    ax.bar([i + width/2 for i in x], means['Over 3'], width, yerr=stds['Over 3'], label='Over 3', capsize=4)

    ax.set_xticks(x)
    ax.set_xticklabels(columns, rotation=45, ha='right')
    ax.set_ylabel('Mean ± STD')
    ax.set_title('Comparison of Selected Metrics')
    ax.legend()
    plt.tight_layout()
    plt.show()


def plotPointsVsFeature(df, features, x_feat, chosen_models):
    # Extract per-track values (as done before)
    track_level_df = df.groupby(level=0).first()[
        features
    ]

    # Plot
    plt.figure(figsize=(8, 6))

    # Loop over each model prediction column
    for column in chosen_models:
        plt.scatter(track_level_df[x_feat], track_level_df[column], label=column, alpha=0.7)

    plt.xlabel('Mean Displacement')
    plt.ylabel('Predicted Diffusion Coefficient')
    plt.title('Model Predictions vs Mean Displacement')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def computeCorrforFeaturesPlotCorr(df, features, index='track_id'):

    # Step 1: Reset index to access 'track_id' as a column
    df_per_frame = df.reset_index()

    # Step 2: Drop duplicates to get one row per track (since these values are constant within each track)
    df_per_track = df_per_frame.drop_duplicates(subset='track_id')[
        features
    ].set_index('track_id')

    # Step 3: Compute Pearson correlation matrix
    correlation_matrix = df_per_track.corr()

    # Step 4: Extract correlations with 'mean_displacement'
    correlation_with_disp = correlation_matrix['mean_max_intensity_over_track'][['D_resnet', 'D_cnn1', 'D_cnn2']]

    print("Correlation between mean_displacement and each model prediction:")
    print(correlation_with_disp)
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title("Correlation Matrix (per track)")
    plt.tight_layout()
    plt.show()
    return correlation_matrix



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML, display

def play_video(video, figsize=(5, 5), fps=5, vmin=None, vmax=None, save_path=None, no_borders=False):
    """
    Displays a stack of images as a video inside Jupyter notebooks with consistent intensity scaling.

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
    save_path : str, optional
        Path to save the video. Supports .mp4 or .gif.
    no_borders : bool, optional
        If True, removes axis, ticks, and borders for clean display.

    Returns
    -------
    Video object
        Returns a video player with input stack of images.
    """
    fig, ax = plt.subplots(figsize=figsize)
    images = []

    if len(video.shape) == 3:
        video = np.expand_dims(video, axis=-1)

    if no_borders:
        ax.axis('off')
        for spine in ax.spines.values():
            spine.set_visible(False)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    else:
        ax.axis("off")

    # If vmin/vmax not provided, compute global min/max across all frames
    if vmin is None:
        vmin = np.min([frame[:, :, 0].min() for frame in video])
    if vmax is None:
        vmax = np.max([frame[:, :, 0].max() for frame in video])
    mean = np.mean(video)

    print(f"vmin: {vmin} vmax: {vmax} mean: {mean:.2f}")

    for image in video:
        images.append([ax.imshow(image[:, :, 0], cmap="gray", vmin=vmin, vmax=vmax)])

    anim = animation.ArtistAnimation(
        fig, images, interval=1e3 / fps, blit=True, repeat_delay=0
    )

    html = HTML(anim.to_jshtml())
    display(html)

    if save_path:
        if save_path.endswith('.mp4'):
            writer = animation.FFMpegWriter(fps=fps, metadata=dict(artist='Me'), bitrate=1800)
            anim.save(save_path, writer=writer)
            print(f"Animation saved to {save_path}")
        elif save_path.endswith('.gif'):
            writer = animation.PillowWriter(fps=fps)
            anim.save(save_path, writer=writer)
            print(f"Animation saved to {save_path}")
        else:
            print("Unsupported file format. Use .mp4 or .gif extension.")

    plt.close()
