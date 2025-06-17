
import numpy as np
import matplotlib.pyplot as plt



def mean_square_displacement(traj):
    """
    Computes the Mean Square Displacement (MSD) for a particle trajectory, 
    which represents the average squared distance moved over time, useful 
    for analyzing diffusion characteristics.

    Parameters:
    - traj (ndarray): Array of shape (num_steps, 2) representing the x, y positions 
                      of a particle over time.

    Returns:
    - msd (ndarray): Array of MSD values computed for each time lag.
    """
    len = traj.shape[0]
    msd = np.zeros(len)
    for tao in range(1,len):
        # Calculate the square of displacements for each tao time t
        displacements = np.sum((traj[tao:] - traj[:len-tao])**2, axis=1)
        msd[tao] = np.mean(displacements)  # Average displacement for the given lag
    return msd


def mean_square_displacements(trajectories):
    """
    Computes the Mean Square Displacement (MSD) for multiple particle trajectories.
    The MSD represents the average squared distance moved over time, useful for
    analyzing diffusion characteristics for each particle.

    Parameters:
    - trajectories (ndarray): Array of shape (nparticles, num_steps, 2) representing
                              the x, y positions of each particle over time.

    Returns:
    - msd (ndarray): Array of MSD values with shape (nparticles, num_steps),
                     where each row corresponds to the MSD values of a particle.
    """
    nparticles, num_steps, _ = trajectories.shape
    msd = np.zeros((nparticles, num_steps))

    # Vectorized MSD calculation
    for tao in range(1, num_steps):
        # Compute squared displacements for all particles at this time lag
        displacements = np.sum((trajectories[:, tao:] - trajectories[:, :num_steps-tao])**2, axis=2)
        
        # Store MSD for each particle at this time lag
        msd[:, tao] = np.mean(displacements, axis=1)

    return msd



def computeAndPlotMeanMSD(msds, nparticles, nframes, nposframe, dt):
    # Set up plot for Mean Square Displacement and diffusion coefficient estimation
    plt.figure(figsize=(4, 4))
    time_range = np.arange(nframes * nposframe) * dt / nposframe    # Time points for MSD plot
    #print(time_range)
    D_estimated = estimateDfromMSDs(msds,time_range) # Array to store estimated diffusion coefficients

    # Loop over each particle to calculate and plot its MSD
    for p in range(nparticles):
        plt.plot(time_range, msds[p], lw=0.25, label=f'Particle {p}')
        

    mean_estimated_D =np.mean(D_estimated)
    plt.plot(time_range, mean_estimated_D *4* time_range , 'k--', lw=0.5, label=f'Slope for Particle {p}')

    # Display estimated diffusion coefficients for each particle
    print("Estimated Diffusion Coefficient:", mean_estimated_D)

    # Set plot details
    plt.title("Mean Square Displacement (MSD) and Estimated Diffusion Coefficient")
    plt.xlabel("Time (s)")
    plt.ylabel("MSD (pixels^2)")
    #plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    return mean_estimated_D




def estimateDfromMSD(msd, time_range):
    """
    Estimates the diffusion coefficient (D) from the mean squared displacement (MSD) 
    using linear regression without an intercept.
    
    Parameters:
    - msd (numpy array): 1D array of MSD values.
    - time_range (numpy array): 1D array of time values corresponding to MSD.
    
    Returns:
    - D_estimated (float): Estimated diffusion coefficient.
    """
    # Solve the least squares problem to fit a line through the origin (y = slope * x)
    slope, = np.linalg.lstsq(time_range.reshape(-1, 1), msd, rcond=None)[0]
    
    # Diffusion coefficient is the slope divided by 4
    D_estimated = slope / 4
    return D_estimated


def estimateDfromMSDs(msds, time_range):
    """
    Estimates the diffusion coefficient (D) for multiple particles from MSD values
    using linear regression without an intercept.
    
    Parameters:
    - msds (numpy array): 2D array of shape (num_particles, msd_len) containing MSD values
                           for multiple particles.
    - time_range (numpy array): 1D array of time values corresponding to MSD.
    
    Returns:
    - D_estimated (numpy array): 1D array of estimated diffusion coefficients for each particle.
    """
    # Solve the least squares problem for multiple MSDs at once
    # Transpose msds to shape (msd_len, num_particles) to fit all particles simultaneously
    slopes = np.linalg.lstsq(time_range.reshape(-1, 1), msds.T, rcond=None)[0][0]
    
    # Diffusion coefficient is the slope divided by 4
    D_estimated = slopes / 4
    return D_estimated  # Shape (num_particles,)

def estimateDfromMSDsWeighted(msds, time_range):
    """
    Estimates the diffusion coefficient (D) for multiple particles from MSD values
    by weighting the lower msd values (using smaller tau) higher than the higher ones.
    
    Parameters:
    - msds (numpy array): 2D array of shape (num_particles, msd_len) containing MSD values
                           for multiple particles.
    - time_range (numpy array): 1D array of time values corresponding to MSD.
    
    Returns:
    - D_estimated (numpy array): 1D array of estimated diffusion coefficients for each particle.
    """
    T = msds.shape[1]
    # weights to multiply each position with to weitgh smaller values of tau more
    weights = np.arange(T,0,-1)

    # weigths to divide the values of masds, since they are proportional to tao
    div_weights = np.arange(0,T)
    # Avoid division by 0, putting one does not change anything since msds[0,:] = 0
    div_weights[0] = 1
    # Element-wise multiplication - divide each MSD value by its corresponding tau
    # This turns MSD into an approximation of the diffusion coefficient at each tau
    normalized_msds = msds / div_weights[np.newaxis, :]

    r = normalized_msds @ weights
    return r / np.sum(weights) /4 # Shape (num_particles,)


def estimateDfromMSDs2(msds, time_range):
    """
    Estimates the diffusion coefficient (D) for multiple particles from MSD values
    using linear regression without an intercept.
    
    Parameters:
    - msds (numpy array): 2D array of shape (num_particles, msd_len) containing MSD values
                           for multiple particles.
    - time_range (numpy array): 1D array of time values corresponding to MSD.
    
    Returns:
    - D_estimated (numpy array): 1D array of estimated diffusion coefficients for each particle.
    """
    # Solve the least squares problem for multiple MSDs at once
    # Transpose msds to shape (msd_len, num_particles) to fit all particles simultaneously

    slopes = np.polyfit(time_range, msds.transpose(1,0), deg = 1)[0, :]
    
    # Diffusion coefficient is the slope divided by 4
    D_estimated = slopes / 4
    return D_estimated  # Shape (num_particles,)








