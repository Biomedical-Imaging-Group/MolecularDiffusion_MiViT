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


def gaussian_2d(xc, yc, sigma, grid_size, amplitude=1.0):
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
