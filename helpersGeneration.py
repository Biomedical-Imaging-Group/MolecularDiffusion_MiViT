import numpy as np
import matplotlib as plt

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



def transform_to_video(
    trajectories,
    nPosPerFrame,
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
            
            '`add_poisson_noise`' : Boolean
                If poisson noise should be added after all the other operations.

    Returns
    -------

    out_images: ndarray
        Array of the shape (N, T/nPosPerFrame, output_size, output_size) containing nFrames images for each particle

    """
    
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

    trajectories = trajectories 

    out_images = np.zeros((N,nFrames,output_size,output_size))

    # Psf is computed as 0.51 * wavelenght/NA according to:
    # https://www.leica-microsystems.com/science-lab/life-science/microscope-resolution-concepts-factors-and-calculation/
    fwhm_psf = 0.51 * _image_dict["wavelength"] / _image_dict["NA"]
    particle_mean, particle_std = _image_dict["particle_intensity"][0],_image_dict["particle_intensity"][1]
    background_mean, background_std = _image_dict["background_intensity"][0],_image_dict["background_intensity"][1]

    # n is for indexing the particle, f the frame, p the subposition
    for n in range(N):
        for f in range(nFrames):
            frame_hr = np.zeros(( output_size*upsampling_factor, output_size*upsampling_factor))
            frame_noisy = np.zeros((output_size, output_size))
            frame_lr = np.zeros((output_size, output_size))

            start = f*nPosPerFrame
            end = (f+1)*nPosPerFrame
            trajectory_segment = trajectories[n,start:end,:]
            xtraj = trajectory_segment[:,0] * upsampling_factor
            ytraj = trajectory_segment[:,1] * upsampling_factor

            # Generate frame, convolution, resampling, noise
            for p in range(nPosPerFrame):
                frame_spot = gaussian_2d(xtraj[p], ytraj[p], upsampling_factor* fwhm_psf/2.355/_image_dict["resolution"], output_size*upsampling_factor, 
                                         np.random.normal(particle_mean/nPosPerFrame,particle_std/nPosPerFrame))
                 
                frame_hr += frame_spot
            frame_lr = block_reduce(frame_hr, block_size=upsampling_factor, func=np.mean)
            # Add Gaussian noise to background intensity across the image
            frame_noisy = frame_lr + np.clip(np.random.normal(background_mean, background_std, frame_lr.shape), 
                                        0, background_mean + 3 * background_std)
            out_images[n,f,:] = frame_noisy

    return out_images