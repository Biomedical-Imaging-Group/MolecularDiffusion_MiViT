import numpy as np


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