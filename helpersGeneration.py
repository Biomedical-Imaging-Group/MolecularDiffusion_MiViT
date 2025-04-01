import numpy as np
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from andi_datasets.models_phenom import models_phenom



def brownian_motion(nparticles, nframes, nposframe, D, dt, startAtZero=False):

    """
    Simulates the Brownian motion of particles over a specified number of frames 
    and interframe positions.

    Parameters:
    - nparticles (int): Number of particles to simulate.
    - nframes (int): Number of frames in the simulation.
    - nposframe (int): Number of interframe positions to calculate per frame.
    - D (float): Diffusion coefficient, influencing the spread of particle movement.
    - dt (float): Time interval between frames, affects particle displacement.
    - startAtZero (bool): If True, initializes the starting position at (0, 0).

    Returns:
    - trajectory (ndarray): Array of shape (nparticles, num_steps, 2) containing 
                            the x, y coordinates of each particle at each time step.
                            `num_steps` is calculated as `nframes * nposframe`.
    """
    num_steps = nframes * nposframe
    positions = np.zeros(2)
    trajectory = np.zeros((nparticles, num_steps, 2))
    
    
    sigma = np.sqrt(2 * D * dt / nposframe)

    for p in range(nparticles):
        # Generate random steps in x and y directions based on normal distribution
        dxy = np.random.randn(num_steps, 2) * sigma
        
        if startAtZero:
            dxy[0, :] = [0, 0]  # Set starting position at origin for the first step
        # Calculate cumulative sum to get positions from step displacements
        positions = np.cumsum(dxy, axis=0)
        trajectory[p] = positions

    return trajectory


def average_trajectory_frames(trajectories, nPosFrame):
    """
    Average multiple position frames together to create a reduced trajectory.
    
    Parameters:
    - trajectories (ndarray): Array of shape (Nparticles, nsteps, 2) 
                              representing the x, y positions of each particle over time.
    - nPosFrame (int): Number of frames to average together.
    
    Returns:
    - averaged_trajectories (ndarray): Array of shape (Nparticles, nsteps/nPosFrame, 2)
                                      with positions averaged over nPosFrame frames.
    """
    Nparticles, nsteps, dims = trajectories.shape
    
    # Calculate how many full frames we can make
    nFullFrames = nsteps // nPosFrame
    
    # Reshape to group frames together
    # This creates an array of shape (Nparticles, nFullFrames, nPosFrame, 2)
    reshaped = trajectories[:, :nFullFrames*nPosFrame].reshape(Nparticles, nFullFrames, nPosFrame, dims)
    
    # Average over the nPosFrame axis (axis=2)
    # Result will be shape (Nparticles, nFullFrames, 2)
    averaged_trajectories = np.mean(reshaped, axis=2)
    
    return averaged_trajectories


def gaussian_2d(xc, yc, sigma, grid_size, amplitude):
    """
    Generates a 2D Gaussian point spread function (PSF) centered at a specified position.

    Parameters:
    - xc, yc (float): The center coordinates (x, y) of the Gaussian within the grid.
    - sigma (float): Standard deviation of the Gaussian, controlling the spread (related to FWHM).
    - grid_size (int): Size of the output grid (grid will be grid_size x grid_size).
    - amplitude (float): Peak amplitude of the Gaussian function.

    Returns:
    - gauss (ndarray): A 2D array representing the Gaussian function centered at (xc, yc).
    """
    limit = (grid_size - 1) // 2  # Defines the range for x and y axes
    x = np.linspace(-limit, limit, grid_size)
    y = np.linspace(-limit, limit, grid_size)
    x, y = np.meshgrid(x, y)
    
    # Calculate the Gaussian function centered at (xc, yc)
    gauss = amplitude * np.exp(-(((x - xc) ** 2) / (2 * sigma ** 2) + ((y - yc) ** 2) / (2 * sigma ** 2)))
    return gauss


from skimage.measure import block_reduce

def generateImages(trajectory, nframes, npixel, factor_hr, nposframe, fwhm_psf, pixelsize, flux, background, gaussian_noise):
    """
    Old function used for generating the IABM Images, see transform_to_video for real version
    """
    frame_hr = np.zeros((nframes, npixel*factor_hr, npixel*factor_hr))
    frame_noisy = np.zeros((nframes, npixel, npixel))
    frame_lr = np.zeros((nframes, npixel, npixel))

    for k in range(nframes):
        start = k*nposframe
        end = (k+1)*nposframe
        trajectory_segment = trajectory[start:end,:]
        xtraj = trajectory_segment[:,0]
        ytraj = trajectory_segment[:,1]
        # Generate frame, convolution, resampling, noise
        for p in range(nposframe):
            frame_spot = gaussian_2d(xtraj[p], ytraj[p], 2.35*fwhm_psf/pixelsize, npixel*factor_hr, flux) 
            frame_hr[k] += frame_spot
        frame_lr[k] = block_reduce(frame_hr[k], block_size=factor_hr, func=np.mean)
        # Add Gaussian noise to background intensity across the image
        frame_noisy[k] = frame_lr[k] + np.clip(np.random.normal(background, gaussian_noise, frame_lr[k].shape), 
                                       0, background + 3 * gaussian_noise)
        
    return frame_hr, frame_lr, frame_noisy



def trajectories_to_video(
    trajectories,
    nPosPerFrame,
    center = False,
    image_props={},
):
    """
    Transforms trajectory data into microscopy imagery data.

    Trajectories generated through phenomenological models in andi-datasets are imaged under a Fluorescence microscope to generate 2D timelapse videos.

    Parameters
    ----------
    trajectory_data : ndarray
        Array of the shape (N, T, 2): (Number of particles, Trajectories Length,2) containing the trajectories of all particles

    nPosPerFrame : int
        Number of subpositions used to generate 1 Framee.

    image_props : dict
        Dictionary containing the properties needed for the image creation simulated as keyword arguments. Valid keys are:

            '`particle_intensity`' : array_like[int, int]
                Intensity distribution of particles within a frame given as mean and standard deviations.

            '`NA`': float
                Numerical aperture of the microscope.

            '`wavelength`' : float
                Wavelength of light in meters.

            '`resolution`' : float
                Effective pixel size of the camera in meters.

            '`output_size`': int
                Image size in pixels, output images will be of size (output_size, output_size)

            "upsampling_factor": int
                Upsampling factor used when generating the image, before reducing to final size

            '`background_intensity`' : array_like[int, int]
                Intensity of background given as mean and standard deviation.
            
            '`poisson_noise`' : float
                If poisson noise should be added after all the other operations. if amount of noise specified is -1, no noise is added
                Default value is 1, normal amount of poisson noise
            
            '`trajectory_unit`' : int
                Unit of the trajectory, -1: pixels , which means the trajectory will not be divided by the resolution, or positive int value (standard value 100nm) which means the trajectory will be divided by resolution. 
                The trajectory is supposed to be given in nm unit, meaning x=1 is a displacement of 1nm
                For example if the unit is 100nm and the resolution is 200nm, the trajectory will be divided by 2, resulting in smaller displacement on screen. 
                100 is the standard value standing for 100nm, to use pixels use -1.

    center : Boolean
        If each subframe should be centered around 0,0: making the particle always be in center of the image

    Returns
    -------

    out_images: ndarray
        Array of the shape (N, T/nPosPerFrame, output_size, output_size) containing nFrames images for each particle

    """
    
    N,T,_ = trajectories.shape

# Invert the y axis, for video creation purposes where y-axis is inverted
    trajectories[:, :, 1] *= -1


    if(T % nPosPerFrame != 0):
        raise Exception("T is not divisble by posPerFrame")

    nFrames = T // nPosPerFrame

    _image_dict = {
        "particle_intensity": [
            500,
            20,
        ],  # Mean and standard deviation of the particle intensity
        "NA": 1.46,  # Numerical aperture
        "wavelength": 500e-9,  # Wavelength
        "resolution": 100e-9,  # Camera resolution or effective resolution, aka pixelsize
        "output_size": 32,
        "upsampling_factor": 5,
        "background_intensity": [
            100,
            10,
        ],  # Standard deviation of background intensity within a video
        "poisson_noise": 1,
        "trajectory_unit" : 100
    }

    # Update the dictionaries with the user-defined values
    _image_dict.update(image_props)
    resolution =_image_dict["resolution"]
    traj_unit = _image_dict["trajectory_unit"]
    
    if(traj_unit !=-1 ):
        trajectories = trajectories * traj_unit* 1e-9 / resolution

    output_size = _image_dict["output_size"]
    upsampling_factor = _image_dict["upsampling_factor"]
    
    # Psf is computed as wavelenght/2NA according to:
    #https://www.sciencedirect.com/science/article/pii/S0005272819301380?via%3Dihub
    fwhm_psf = _image_dict["wavelength"] / 2 * _image_dict["NA"]


    gaussian_sigma = upsampling_factor/resolution * fwhm_psf/2.355
    poisson_noise = _image_dict["poisson_noise"]
    
    out_videos = np.zeros((N,nFrames,output_size,output_size),np.float32)
    particle_mean, particle_std = _image_dict["particle_intensity"][0],_image_dict["particle_intensity"][1]
    background_mean, background_std = _image_dict["background_intensity"][0],_image_dict["background_intensity"][1]

    # n is for indexing the particle
    for n in range(N):
        trajectory_to_video(out_videos[n,:],trajectories[n,:],nFrames,output_size,upsampling_factor,nPosPerFrame,
                                              gaussian_sigma,particle_mean,particle_std,background_mean,background_std, poisson_noise,center)
    return out_videos


 

def trajectory_to_video(out_video,trajectory,nFrames, output_size, upsampling_factor, nPosPerFrame,gaussian_sigma,particle_mean,particle_std,background_mean,background_std, poisson_noise, center):
    """Helper function of function above, all arguments documented above"""
    for f in range(nFrames):
        frame_hr = np.zeros(( output_size*upsampling_factor, output_size*upsampling_factor),np.float32)
        frame_lr = np.zeros((output_size, output_size),np.float32)

        start = f*nPosPerFrame
        end = (f+1)*nPosPerFrame
        trajectory_segment = (trajectory[start:end,:] - np.mean(trajectory[start:end,:],axis=0) if center else trajectory[start:end,:]) 
        xtraj = trajectory_segment[:,0]  * upsampling_factor
        ytraj = trajectory_segment[:,1] * upsampling_factor

        

        # Generate frame, convolution, resampling, noise
        for p in range(nPosPerFrame):
            spot_intensity = np.random.normal(particle_mean/nPosPerFrame,particle_std/nPosPerFrame)
            frame_spot = gaussian_2d(xtraj[p], ytraj[p], gaussian_sigma, output_size*upsampling_factor, spot_intensity)

            # gaussian_2d maximum is not always the wanted one because of some misplaced pixels. 
            # We can force the peak of the gaussian to have the right intensity
            spot_max = np.max(frame_spot)
            if(spot_max < 0.00001):
                print("Particle Left the image")
            frame_hr += spot_intensity/spot_max * frame_spot
        
        frame_lr = block_reduce(frame_hr, block_size=upsampling_factor, func=np.mean)
        # Add Gaussian noise to background intensity across the image
        frame_lr += np.clip(np.random.normal(background_mean, background_std, frame_lr.shape), 
                                    0, background_mean + 3 * background_std)
        
        # Add poisson noise if specified
        if poisson_noise != -1:
            frame_lr = np.random.poisson(frame_lr * poisson_noise) / poisson_noise

        out_video[f,:] = frame_lr


# Define this function at the module level, not nested inside another function
def process_one_video(args):
    """
    Process a single video.
    
    Args:
        args: Tuple containing (n, trajectories_n, nFrames, output_size, upsampling_factor, 
              nPosPerFrame, gaussian_sigma, particle_mean, particle_std, 
              background_mean, background_std)
    
    Returns:
        Tuple of (n, video)
    """
    # Unpack arguments
    (n, trajectory, nFrames, output_size, upsampling_factor, 
     nPosPerFrame, gaussian_sigma, particle_mean, particle_std, 
     background_mean, background_std) = args
    
    # Create a local array for this video
    video = np.zeros((nFrames, output_size, output_size))
    
    # Call your existing function
    trajectory_to_video(
        video, trajectory, nFrames, output_size, upsampling_factor, 
        nPosPerFrame, gaussian_sigma, particle_mean, particle_std, 
        background_mean, background_std
    )
    
    return n, video

def generate_videos_parallel(trajectories, N, nFrames, output_size, upsampling_factor, 
                            nPosPerFrame, gaussian_sigma, particle_mean, particle_std, 
                            background_mean, background_std, max_workers=None):
    """
    Generate videos in parallel using ProcessPoolExecutor.
    
    Parameters:
    - trajectories: Array of trajectories
    - N: Number of videos to generate
    - Other parameters as needed by trajectory_to_video
    - max_workers: Maximum number of worker processes (defaults to number of CPU cores)
    
    Returns:
    - out_videos: Array of generated videos
    """
    # Initialize output array
    out_videos = np.zeros((N, nFrames, output_size, output_size))
    
    # If max_workers is None, use number of CPU cores
    if max_workers is None:
        max_workers = multiprocessing.cpu_count()
    
    # Prepare arguments for each video
    args_list = [
        (n, trajectories[n,:], nFrames, output_size, upsampling_factor, 
         nPosPerFrame, gaussian_sigma, particle_mean, particle_std, 
         background_mean, background_std) 
        for n in range(N)
    ]
    
    # Process videos in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks and collect results
        results = list(executor.map(process_one_video, args_list))
        
        # Organize results
        for n, video in results:
            out_videos[n] = video
    
    return out_videos


def trajectories_to_video2(
    trajectories,
    nPosPerFrame,
    image_props={},
):

    
    N,T,_ = trajectories.shape

    if(T % nPosPerFrame != 0):
        raise Exception("T is not divisble by posPerFrame")

    nFrames = T // nPosPerFrame

    _image_dict = {
        "particle_intensity": [
            500,
            20,
        ],  # Mean and standard deviation of the particle intensity
        "NA": 1.46,  # Numerical aperture
        "wavelength": 500e-9,  # Wavelength
        "resolution": 100e-9,  # Camera resolution or effective resolution, aka pixelsize
        "output_size": 32,
        "upsampling_factor": 5,
        "background_intensity": [
            100,
            10,
        ],  # Standard deviation of background intensity within a video
        "add_poisson_noise": True
    }

    # Update the dictionaries with the user-defined values
    _image_dict.update(image_props)

    output_size = _image_dict["output_size"]
    upsampling_factor = _image_dict["upsampling_factor"]
    # Psf is computed as 0.51 * wavelenght/NA according to:
    fwhm_psf = 0.51 * _image_dict["wavelength"] / _image_dict["NA"]
    gaussian_sigma = upsampling_factor* fwhm_psf/2.355/_image_dict["resolution"]

    trajectories = trajectories 
    
    out_videos = np.zeros((N,nFrames,output_size,output_size))

    # https://www.leica-microsystems.com/science-lab/life-science/microscope-resolution-concepts-factors-and-calculation/
    particle_mean, particle_std = _image_dict["particle_intensity"][0],_image_dict["particle_intensity"][1]
    background_mean, background_std = _image_dict["background_intensity"][0],_image_dict["background_intensity"][1]

    out_videos = generate_videos_parallel(
        trajectories, N, nFrames, output_size, upsampling_factor, nPosPerFrame,
        gaussian_sigma, particle_mean, particle_std, background_mean, background_std
    )

    return out_videos




def normalize_images(images, background_mean=None, background_sigma=None, theoretical_max=None):
    """
    Normalize images according to the formula:
    im_norm = (im - (background_mean - background_sigma)) / (theoretical_max - (background_mean - background_sigma))
    
    Parameters:
    ----------
    images : numpy.ndarray
        The images to normalize. Can be single image or batch with shape (..., height, width)
    background_mean : float, optional
        Mean of the background. If None, computed as np.mean(images)
    background_sigma : float, optional
        Standard deviation of the background. If None, computed as np.std(images)
    theoretical_max : float, optional
        Maximum theoretical value of particle that would not move. If None, computed as np.max(images)
        
    Returns:
    -------
    numpy.ndarray
        Normalized images with same shape as input
    """
    
    # Compute statistics if not provided
    if background_mean is None:
        background_mean = np.mean(images)
    
    if background_sigma is None:
        background_sigma = np.std(images)
    
    if theoretical_max is None:
        theoretical_max = np.max(images)
    
    # Apply normalization
    denominator = theoretical_max - (background_mean - background_sigma)
    
    # Avoid division by zero
    if denominator == 0:
        raise ValueError("Denominator in normalization is zero. Check your inputs.")
    
    normalized = (images - (background_mean - background_sigma)) / denominator
    
    return normalized, (background_mean, background_sigma, theoretical_max)

def generateTrajAndVideosBrownian(Ds,nPart,nImages,nPosPerFrame,optics_props):
    trajs, labels = models_phenom().single_state(nPart, 
                                    L = 0,
                                    T = nImages * nPosPerFrame,
                                    Ds = Ds, # Mean and variance
                                    alphas = 1)
    
    trajs = trajs.transpose(1,0,2)
    labels = labels.transpose(1,0,2)

    videos = trajectories_to_video(trajs, nPosPerFrame,True,optics_props)   

    return videos, labels[:,0,1]















# Helpers for plotting: 


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