import numpy as np
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from andi_datasets.models_phenom import models_phenom
from skimage.measure import block_reduce
import skimage as ski


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
    use_multiprocessing = False,
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
                Default value is 100, normal amount of poisson noise. Lower values generate more noise and higher values less noise
                Adding Poisson noise replicated the microscope's error rate, where observed noise is a right tail skewed gaussian
            
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
        "psf_division_factor": 1, 
        "resolution": 100e-9,  # Camera resolution or effective resolution, aka pixelsize
        "output_size": 32,
        "upsampling_factor": 5,
        "background_intensity": [
            100,
            10,
        ],  # Standard deviation of background intensity within a video
        "poisson_noise": 100,
        "trajectory_unit" : 100
    }

    # Update the dictionaries with the user-defined values
    _image_dict.update(image_props)
    resolution =_image_dict["resolution"]
    traj_unit = _image_dict["trajectory_unit"]
    
    if(traj_unit !=-1 ):
        # put the trajectory in pixels
        trajectories = trajectories * traj_unit / (resolution* 1e9)

    output_size = _image_dict["output_size"]
    upsampling_factor = _image_dict["upsampling_factor"]
    psf_div_factor = _image_dict["psf_division_factor"]
    
    # Psf is computed as wavelenght/2NA according to:
    #https://www.sciencedirect.com/science/article/pii/S0005272819301380?via%3Dihub
    fwhm_psf = _image_dict["wavelength"] / 2 * _image_dict["NA"] / psf_div_factor

    
    gaussian_sigma = upsampling_factor/ resolution * fwhm_psf/2.355
    poisson_noise = _image_dict["poisson_noise"]
    
    out_videos = np.zeros((N,nFrames,output_size,output_size),np.float32)
    particle_mean, particle_std = _image_dict["particle_intensity"][0],_image_dict["particle_intensity"][1]
    background_mean, background_std = _image_dict["background_intensity"][0],_image_dict["background_intensity"][1]


    if (use_multiprocessing):
        # Use multiprocessing to generate videos in parallel        

        # Get the number of available CPU cores
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
    
    else:
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
            if(particle_mean >0.0001 and particle_std > 0.0001):
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
            frame_lr = frame_lr * np.random.poisson(poisson_noise, size=(frame_lr.shape)) / poisson_noise

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





def normalize_images(images, background_mean=None, background_sigma=None, theoretical_max=None, clip_image=False):
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

    if clip_image:
        normalized = np.clip(normalized,0, 1.5)
    
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







def trajectories_to_video_multiple_settings(
    trajectories,
    nPosPerFrame,
    center = False,
    image_props={},
):
    
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
        "psf_division_factor": 1, 
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
    psf_div_factor = _image_dict["psf_division_factor"]
    
    # Psf is computed as wavelenght/2NA according to:
    #https://www.sciencedirect.com/science/article/pii/S0005272819301380?via%3Dihub
    fwhm_psf = _image_dict["wavelength"] / 2 * _image_dict["NA"] / psf_div_factor

    
    gaussian_sigma = upsampling_factor/ resolution * fwhm_psf/2.355
    poisson_noise = _image_dict["poisson_noise"]
    
    out_videos_no_noise = np.zeros((N,nFrames,output_size,output_size),np.float32)
    out_videos_gauss_noise = np.zeros((N,nFrames,output_size,output_size),np.float32)
    out_videos_Poisson = np.zeros((N,nFrames,output_size,output_size),np.float32)
    out_videos_gauss_filter = np.zeros((N,nFrames,output_size,output_size),np.float32)

    particle_mean, particle_std = _image_dict["particle_intensity"][0],_image_dict["particle_intensity"][1]
    background_mean, background_std = _image_dict["background_intensity"][0],_image_dict["background_intensity"][1]
    
    for n in range(N):
        trajectory_to_videoTest(out_videos_no_noise[n,:],out_videos_gauss_noise[n,:],out_videos_Poisson[n,:],out_videos_gauss_filter[n,:],trajectories[n,:],nFrames,output_size,upsampling_factor,nPosPerFrame,
                                                gaussian_sigma,particle_mean,particle_std,background_mean,background_std, poisson_noise,center)
        
    return out_videos_no_noise,out_videos_gauss_noise, out_videos_Poisson, out_videos_gauss_filter


def trajectory_to_videoTest(out_video_no_noise,out_video_gauss_noise, out_video_poiss_noise, out_video_gauss_filter,trajectory,nFrames, output_size, upsampling_factor, nPosPerFrame,gaussian_sigma,particle_mean,particle_std,background_mean,background_std, poisson_noise, center):
    """Helper function of function above, all arguments documented above"""
    for f in range(nFrames):
        frame_hr = np.zeros(( output_size*upsampling_factor, output_size*upsampling_factor),np.float32)
        frame_lr = np.zeros((output_size, output_size),np.float32)

        start = f*nPosPerFrame
        end = (f+1)*nPosPerFrame
        trajectory_segment = (trajectory[start:end,:] - np.mean(trajectory[start:end,:],axis=0) if center else trajectory[start:end,:]) 
        xtraj = trajectory_segment[:,0]  * upsampling_factor
        ytraj = trajectory_segment[:,1] * upsampling_factor

        frame_intensity = np.random.normal(particle_mean,particle_std)
        

        # Generate frame, convolution, resampling, noise
        for p in range(nPosPerFrame):
            if(particle_mean >0.0001 and particle_std > 0.0001):
                #spot_intensity = np.random.normal(particle_mean/nPosPerFrame,particle_std/nPosPerFrame)
                spot_intensity = frame_intensity/ nPosPerFrame
                frame_spot = gaussian_2d(xtraj[p], ytraj[p], gaussian_sigma, output_size*upsampling_factor, spot_intensity)

                # gaussian_2d maximum is not always the wanted one because of some misplaced pixels. 
                # We can force the peak of the gaussian to have the right intensity
                spot_max = np.max(frame_spot)
                if(spot_max < 0.00001):
                    print("Particle Left the image")
                frame_hr += spot_intensity/spot_max * frame_spot
        
        frame_lr = block_reduce(frame_hr, block_size=upsampling_factor, func=np.mean)
        # Add Gaussian noise to background intensity across the image
        frame_no_noise = frame_lr.copy()
        frame_lr += np.clip(np.random.normal(background_mean, background_std, frame_lr.shape), 
                                    0, background_mean + 3 * background_std)

        frame_lr_poisson = np.random.poisson(frame_lr * poisson_noise) / poisson_noise
        frame_gaussian_filter = ski.filters.gaussian(frame_lr_poisson, sigma=0.5)  # mild smoothing

        out_video_no_noise[f,:] = frame_no_noise
        out_video_gauss_noise[f,:] = frame_lr
        out_video_poiss_noise[f,:] = frame_lr_poisson
        out_video_gauss_filter[f,:] = frame_gaussian_filter

    return 

from scipy.signal import fftconvolve


def tv_gradient(image):
    """Compute the gradient of total variation."""
    grad = np.zeros_like(image)
    dx = np.diff(image, axis=1, append=image[:, -1:])
    dy = np.diff(image, axis=0, append=image[-1:, :])
    eps = 1e-8
    mag = np.sqrt(dx**2 + dy**2 + eps)
    dx_norm = dx / mag
    dy_norm = dy / mag
    grad[:, :-1] -= dx_norm[:, :-1]
    grad[:, 1:] += dx_norm[:, :-1]
    grad[:-1, :] -= dy_norm[:-1, :]
    grad[1:, :] += dy_norm[:-1, :]
    return grad

def richardson_lucy_tv(image, psf, iterations=20, tv_weight=0.01):
    image = np.clip(image, 1e-6, None)
    psf_mirror = psf[::-1, ::-1]
    estimate = np.full(image.shape, 0.5, dtype=np.float32)
    for i in range(iterations):
        relative_blur = image / (fftconvolve(estimate, psf, mode='same') + 1e-6)
        correction = fftconvolve(relative_blur, psf_mirror, mode='same')
        estimate *= correction
        tv_grad = tv_gradient(estimate)
        estimate -= tv_weight * tv_grad
        estimate = np.clip(estimate, 0, 1)

    return estimate

def richardson_lucy_tv_iter_list(image, psf, iterations_list, out_array, tv_weight=0.01):
    image = np.clip(image, 1e-6, None)
    psf_mirror = psf[::-1, ::-1]
    estimate = np.full(image.shape, 0.5, dtype=np.float32)
    max_iterations = iterations_list[-1] +1
    for i in range(max_iterations):
        relative_blur = image / (fftconvolve(estimate, psf, mode='same') + 1e-6)
        correction = fftconvolve(relative_blur, psf_mirror, mode='same')
        estimate *= correction
        tv_grad = tv_gradient(estimate)
        estimate -= tv_weight * tv_grad
        estimate = np.clip(estimate, 0, 1)

        if(i in iterations_list):
            index = iterations_list.index(i)
            out_array[index] = estimate.copy()
    return estimate



def create_gaussian_psf(size=9, sigma=1.3):
    if size % 2 == 0:
        size += 1  # ensure odd size for symmetry
    ax = np.arange(-size // 2 + 1, size // 2 + 1)
    x, y = np.meshgrid(ax, ax)
    psf = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    psf /= psf.sum()
    return psf

import torch
import numpy as np

def apply_rl_tv_tensor(tensor, psf, n_iters=10, tv_weight=0.01):
    B, seq, H, W = tensor.shape
    assert H == 9 and W == 9, "Only images of shape 9x9 are supported"
    
    tensor_np = tensor.detach().cpu().numpy()  # Convert to NumPy
    result_np = np.empty_like(tensor_np)

    for b in range(B):
        for t in range(seq):
            result_np[b, t] = richardson_lucy_tv(tensor_np[b, t], psf, iterations=n_iters, tv_weight=tv_weight)

    return torch.tensor(result_np, dtype=tensor.dtype, device=tensor.device)

def apply_rl_tv_tensor_iter_list(tensor, psf, iterations_list=[2,5,10], tv_weight=0.01):
    B, seq, H, W = tensor.shape
    assert H == 9 and W == 9, "Only images of shape 9x9 are supported"
    
   # Convert to NumPy if it's a tensor, otherwise use as-is if it's already a numpy array
    tensor_np = tensor.detach().cpu().numpy() if torch.is_tensor(tensor) else tensor

    n_iters = len(iterations_list)

    result_np = np.empty((B, n_iters, seq, H, W), dtype=tensor_np.dtype)
    for b in range(B):
        for t in range(seq):
            richardson_lucy_tv_iter_list(tensor_np[b, t], psf, iterations_list=iterations_list, out_array=result_np[b,:,t], tv_weight=tv_weight)

    return result_np


psf = create_gaussian_psf(sigma=1)


def trajs_to_vid_norm_rl(trajectories,
    nPosPerFrame,
    center,
    image_props, 
    rl_iterations,
    poisson_index = 2):
    
    bg_mean, bg_sigma = image_props["background_intensity"]
    part_mean, part_sigma = image_props["particle_intensity"]

    videos = trajectories_to_video_multiple_settings(trajectories, nPosPerFrame, center=center, image_props=image_props)
    # videos is a tuple of 5 arrays of shape (N, length, P, P)
    # Stack along a new axis to get shape (N, 4, length, P, P)
    videos = np.stack(videos, axis=1)
    # Normalize videos (handle batch of shape (N, 4, length, P, P))
    videos, _ = normalize_images(videos, bg_mean, bg_sigma, part_mean + bg_mean)
    # Take poisson noisy sequences
    vids_to_rl = videos[:,poisson_index]
    vids_with_rl = apply_rl_tv_tensor_iter_list(vids_to_rl,psf,rl_iterations)

    combined_videos = np.concatenate([videos, vids_with_rl], axis=1)
    return combined_videos

from helpersFeatures import *

def create_video_and_feature_pairs(trajectories, 
                                   nPosPerFrame, 
                                   center,
                                   image_props,
                                   dt=1.0):
    """
    Converts a list of trajectories into normalized videos and diffusion features.

    Args:
        trajectories: List or array of particle trajectories.
        nPosPerFrame: Number of positions per video frame.
        image_props: Dictionary of image properties for rendering.
        background_mean, background_sigma, part_mean: Parameters for normalization.
        dt: Time step for computing diffusion features.

    Returns:
        videos: List of normalized video arrays (one per trajectory), shape (num_traj, num_frames, H, W)
        features: List of feature vectors (one per trajectory), shape (num_traj, num_features)
    """
    features = []

    bg_mean, bg_sigma = image_props["background_intensity"]
    part_mean, part_sigma = image_props["particle_intensity"]

    # Generate and normalize video
    videos = trajectories_to_video(trajectories, nPosPerFrame, center=center, image_props=image_props)
    videos, _ = normalize_images(videos, bg_mean, bg_sigma, part_mean + bg_mean)

    for traj in trajectories:
        # === Average positions every nPosPerFrame ===
        num_frames = len(traj) // nPosPerFrame
        reshaped = traj[:num_frames * nPosPerFrame].reshape(num_frames, nPosPerFrame, -1)  # shape: [num_frames, nPosPerFrame, 2]
        avg_traj = reshaped.mean(axis=1)  # shape: [num_frames, 2]

        # Compute features on the averaged trajectory
        feat = compute_diffusion_features(avg_traj, dt=dt)
        features.append(feat)


    # Stack into arrays
    features = np.stack(features)

    return videos, features