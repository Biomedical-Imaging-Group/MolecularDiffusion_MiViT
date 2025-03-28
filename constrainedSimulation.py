import numpy as np
from stochastic.processes.noise import FractionalGaussianNoise as FGN


def disp_fbm(alpha : float,
                D : float,
                T: int, 
                deltaT : int = 1):
    ''' Generates normalized Fractional Gaussian noise. This means that, in 
    general:
    $$
    <x^2(t) > = 2Dt^{alpha}
    $$
                        
    and in particular:
    $$
    <x^2(t = 1)> = 2D 
    $$
    
    Parameters
    ----------
    alpha : float in [0,2]
        Anomalous exponent, for alpha = 1 -> Brownian motion
    D : float
        Diffusion coefficient
    T : int
        Number of displacements to generate
    deltaT : int, optional
        Sampling time
        
    Returns
    -------
    numpy.array
        Array containing T displacements of given parameters
    
    '''
    
    # Generate displacements
    disp = FGN(hurst = alpha/2).sample(n = T)
    # Normalization factor
    disp *= np.sqrt(T)**(alpha)
    # Add D
    disp *= np.sqrt(2*D*deltaT)        
    
    return disp




import numpy as np

def traj_in_rectangle(T, D, alpha, deltaT, rect, startPos=(0, 0)):
    """
    Simulates the trajectory of a single particle undergoing fractional Brownian motion 
    within a constrained rectangular region with an arbitrary angle.

    Parameters
    ----------
    T : int
        Number of time steps for the simulation.
    D : float
        Diffusion coefficient controlling the magnitude of displacements.
    alpha : float in [0,2]
        Anomalous exponent defining the nature of the motion 
        (alpha = 1 corresponds to standard Brownian motion).
    deltaT : int
        Sampling time interval between steps.
    rect : tuple of (float, float, float)
        A tuple defining the constrained rectangular domain:
        - rect[0] (length): The length of the rectangle.
        - rect[1] (width): The width of the rectangle.
        - rect[2] (angle in radians): The orientation of the rectangle in 2D space, oriented to the right compared to the vertical line.
    startPos : tuple of (float, float), optional
        Initial position of the particle within the rectangle. Defaults to (0,0).

    Returns
    -------
    numpy.ndarray
        Array of shape (T, 2) containing the (x, y) positions of the particle at each step.
    """
    length, width, angle = rect

    # angle is given in trigonometry convention, convert it to our system (oriented to the right)
    angle = -angle
    
    # Rotation matrix
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    R_inv = np.linalg.inv(R)  # Inverse for converting back to original space
    
    # Transform start position to rotated frame
    start_pos_rot = R_inv @ np.array(startPos)
    x_min, x_max = -width / 2, width / 2
    y_min, y_max = -length / 2, length / 2
    
    # Check if start position is inside boundaries
    if not (x_min <= start_pos_rot[0] <= x_max and y_min <= start_pos_rot[1] <= y_max):
        raise ValueError("startPos is outside the defined rectangle boundaries.")

    # Generate displacements
    disp_x = disp_fbm(alpha, D, T, deltaT)
    disp_y = disp_fbm(alpha, D, T, deltaT)
    
    # Initialize position array in rotated frame
    positions_rot = np.zeros((T, 2))
    positions_rot[0] = start_pos_rot
    
    # Simulate trajectory with reflective boundaries
    for t in range(1, T):
        new_x = positions_rot[t - 1, 0] + disp_x[t]
        new_y = positions_rot[t - 1, 1] + disp_y[t]
        
        # Reflect off vertical walls
        if new_x < x_min:
            new_x = x_min + (x_min - new_x)
        elif new_x > x_max:
            new_x = x_max - (new_x - x_max)
        
        # Reflect off horizontal walls
        if new_y < y_min:
            new_y = y_min + (y_min - new_y)
        elif new_y > y_max:
            new_y = y_max - (new_y - y_max)
        
        positions_rot[t] = (new_x, new_y)
    
    # Transform trajectory back to original space
    positions = (R @ positions_rot.T).T
    
    return positions

