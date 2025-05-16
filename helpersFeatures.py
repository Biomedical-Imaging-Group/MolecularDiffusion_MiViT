import numpy as np
import scipy.stats as stats
from scipy.spatial import ConvexHull


# Define feature names in order they'll appear in output array
feature_names = [
        "alpha",                    # Anomalous diffusion exponent
        "diffusion_coefficient",    # Diffusion coefficient
        "r_squared",                # R² of the MSD fit
        "efficiency_log",           # Log of efficiency
        "efficiency",               # Efficiency (end-to-end distance / sum of step lengths)
        "fractal_dimension",        # Fractal dimension
        "gaussianity",              # Gaussianity
        "kurtosis",                 # Kurtosis
        "msd_ratio",                # MSD ratio
        "trappedness",              # Trappedness
        "trajectory_length",        # Number of points in trajectory
        "mean_step_length",         # Mean step length
        "mean_msd",                 # Mean MSD value
        "mean_dot_product",         # Mean dot product between consecutive steps
        "fraction_same_direction",  # Fraction of steps in same direction as previous
        "fraction_positive_direction", # Fraction of steps in positive direction
        "total_distance",           # Total distance traveled
        "min_step",                 # Minimum step length
        "max_step",                 # Maximum step length
        "step_range",               # Range of step lengths
        "avg_velocity",             # Average velocity
        "step_cv",                  # Coefficient of variation of step lengths
        "fraction_small_steps",     # Fraction of steps < 0.1
        "fraction_large_steps",     # Fraction of steps > 0.4
        "convex_hull_area"          # Area of convex hull
    ]
N_features = len(feature_names)

def squared_dist(x0, x1, y0, y1):
    """
    Computes the squared distance between the two points (x0,y0) and (x1,y1)
    
    Parameters
    ----------
    x0, x1, y0, y1 : floats
        Coordinates of the two points
        
    Returns
    -------
    float
        Squared distance between the two input points
    """
    return (x1 - x0) ** 2 + (y1 - y0) ** 2


def quad_dist(x0, x1, y0, y1):
    """
    Computes the four-norm (x1-x0)**4 + (y1-y0)**4.
    
    Parameters
    ----------
    x0, x1, y0, y1 : floats
        Coordinates of the two points
        
    Returns
    -------
    float
        Four-norm distance
    """
    return (x1 - x0) ** 4 + (y1 - y0) ** 4


def get_max_dist(x, y):
    """
    Computes the maximum squared distance between all points in the (x,y) set.
    
    Parameters
    ----------
    x : array-like
        x-coordinates
    y : array-like
        y-coordinates
        
    Returns
    -------
    float
        Largest squared distance between any two points in the set
    """
    from itertools import combinations
    
    A = np.array([x, y]).T
    
    def square_distance(p1, p2):
        return sum([(xi - yi) ** 2 for xi, yi in zip(p1, p2)])
    
    max_square_distance = 0
    for pair in combinations(A, 2):
        current_dist = square_distance(*pair)
        if current_dist > max_square_distance:
            max_square_distance = current_dist
    
    return max_square_distance


def msd(x, y, frac=0.5):
    """
    Computes the mean squared displacement (msd) for a trajectory (x,y) up to
    frac*len(x) of the trajectory.
    
    Parameters
    ----------
    x : array-like
        x-coordinates for the trajectory
    y : array-like
        y-coordinates for the trajectory
    frac : float in [0,1]
        Fraction of trajectory duration to compute msd up to
        
    Returns
    -------
    array
        msd for the trajectory
    """
    N = int(len(x) * frac) if len(x) > 20 else len(x)
    msd_values = []
    
    for lag in range(1, N):
        msd_values.append(
            np.mean([
                squared_dist(x[j], x[j + lag], y[j], y[j + lag])
                for j in range(len(x) - lag)
            ])
        )
    
    return np.array(msd_values)


def fit_diffusion_scaling(msds, dt, dim=2):
    """
    Fit mean squared displacements to determine diffusion parameters.
    
    Parameters
    ----------
    msds : array-like
        Mean squared displacements
    dt : float
        Time step
    dim : int
        Dimensionality (should be 2 for 2D trajectories)
    difftype : str
        Type of diffusion model to fit
        
    Returns
    -------
    tuple
        (parameters, pvalue) where parameters contains D (diffusion coefficient) 
        and alpha (anomalous exponent) for normal/subdiffusive,
        or additional parameters for other diffusion types
    """
    from scipy.optimize import curve_fit
    
    t_vals = np.arange(1, len(msds) + 1) * dt
    
    # MSD = 2*d*D*t^alpha + offset
    def power_law(t, D, alpha, offset):
        return 2 * dim * D * (t) ** alpha + offset
    
    p0 = [msds[0] / (4 * dt), 1, 0.001]
    bounds = ([0.00001, 0.00001, 0], [np.inf, 10, np.inf])
        

    try:
        # Try to fit with curve_fit
        params, pcov = curve_fit(
            power_law, t_vals, msds, 
            p0=p0, bounds=bounds, method='trf', 
            maxfev=10000
        )
        
        # Calculate residuals and chi-square
        residuals = msds - power_law(t_vals, *params)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((msds - np.mean(msds))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # Use R² as a simple measure of goodness of fit instead of p-value
        # since we're avoiding the complex Chi2Fit function
        pval = r_squared
        
    except (RuntimeError, ValueError):        # If fitting fails, return default values
        params = (0,0)
        pval = 0
    
    return params, pval


def efficiency(x, y):
    """
    Computes the efficiency of a trajectory, logarithm of the ratio of squared end-to-end distance
    and the sum of squared distances.
    
    Parameters
    ----------
    x : array-like
        x-coordinates for the trajectory
    y : array-like
        y-coordinates for the trajectory
        
    Returns
    -------
    tuple
        (log_efficiency, efficiency)
    """
    top = squared_dist(x[0], x[-1], y[0], y[-1])
    bottom = sum([squared_dist(x[i], x[i + 1], y[i], y[i + 1]) for i in range(len(x) - 1)])
    
    if bottom == 0:  # Avoid division by zero
        return -np.inf, 0
    
    eff = top / ((len(x) - 1) * bottom)
    return np.log(eff), eff


def fractal_dim(x, y, max_square_distance):
    """
    Computes the fractal dimension using the estimator suggested by Katz & George.
    
    Parameters
    ----------
    x : array-like
        x-coordinates for the trajectory
    y : array-like
        y-coordinates for the trajectory
    max_square_distance : float
        Maximum squared pair-wise distance for the points in the trajectory
        
    Returns
    -------
    float
        Estimated fractal dimension
    """
    total_length = sum([
        np.sqrt(squared_dist(x[i], x[i + 1], y[i], y[i + 1]))
        for i in range(len(x) - 1)
    ])
    
    if total_length == 0:  # Avoid division by zero
        return 1
    
    return np.log(len(x)) / (np.log(len(x)) + np.log(np.sqrt(max_square_distance) / total_length))


def gaussianity(x, y, r2):
    """
    Computes the Gaussianity parameter.
    
    Parameters
    ----------
    x : array-like
        x-coordinates for the trajectory
    y : array-like
        y-coordinates for the trajectory
    r2 : array-like
        Mean squared displacements for the trajectory
        
    Returns
    -------
    float
        Gaussianity parameter
    """
    gn = []
    for lag in range(1, len(r2) + 1):
        if lag >= len(x):
            break
            
        r4 = np.mean([
            quad_dist(x[j], x[j + lag], y[j], y[j + lag]) 
            for j in range(len(x) - lag)
        ])
        
        if r2[lag-1] > 0:  # Avoid division by zero
            gn.append(r4 / (2 * r2[lag-1] ** 2))
    
    if len(gn) == 0:
        return np.nan
    
    return np.mean(gn)


def kurtosis(x, y):
    """
    Computes the kurtosis for the trajectory.
    
    Parameters
    ----------
    x : array-like
        x-coordinates for the trajectory
    y : array-like
        y-coordinates for the trajectory
        
    Returns
    -------
    float
        Kurtosis
    """
    from scipy.stats import kurtosis as scipy_kurtosis
    
    try:
        # Get the covariance matrix
        cov_matrix = np.cov(x, y)
        
        # Get eigenvalues and eigenvectors
        val, vec = np.linalg.eig(cov_matrix)
        
        # Get the dominant direction (eigenvector with largest eigenvalue)
        dominant = vec[:, np.argsort(val)][:, -1]
        
        # Project points onto the dominant direction
        projection = np.array([np.dot(dominant, v) for v in np.array([x, y]).T])
        
        # Calculate kurtosis (using Fisher's definition = 0 for normal distribution)
        kurt = scipy_kurtosis(projection, fisher=False)
        
        return kurt
    
    except:
        return np.nan


def msd_ratio(msd_values):
    """
    Computes the MSD ratio parameter.
    
    Parameters
    ----------
    msd_values : array-like
        Mean squared displacements
        
    Returns
    -------
    float
        MSD ratio
    """
    if len(msd_values) < 2:
        return np.nan
    
    ratios = [msd_values[i] / msd_values[i + 1] - (i + 1) / (i + 2) 
              for i in range(len(msd_values) - 1)]
    
    return np.mean(ratios)


def trappedness(x, maxpair, msd_params):
    """
    Computes the trappedness parameter.
    
    Parameters
    ----------
    x : array-like
        x-coordinates for the trajectory
    maxpair : float
        Maximum squared pair-wise distance for the points in the trajectory
    msd_params : tuple
        Parameters from MSD fitting
        
    Returns
    -------
    float
        Trappedness parameter
    """
    if len(msd_params) < 2:
        return np.nan
    
    r0 = np.sqrt(maxpair) / 2
    D = msd_params[0]  # Diffusion coefficient
    
    if r0 == 0 or D == 0:  # Avoid division by zero or log of zero
        return 0
    
    # Formula from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1262511/
    return 1 - np.exp(0.2045 - 0.25117 * (D * len(x)) / r0 ** 2)


def convex_hull_area(x, y):
    """
    Computes the area of the convex hull of the trajectory.
    
    Parameters
    ----------
    x : array-like
        x-coordinates for the trajectory
    y : array-like
        y-coordinates for the trajectory
        
    Returns
    -------
    float
        Area of the convex hull
    """
    try:
        points = np.vstack([x, y]).T
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is actually the area
    except:
        return 0


def dot_product_traces(trace):
    """
    Computes dot products between consecutive vectors in a trajectory.
    
    Parameters
    ----------
    trace : array-like
        Trajectory points as an Nx2 array
        
    Returns
    -------
    array
        Dot products between consecutive vectors
    """
    vecs = trace[1:] - trace[:-1]
    if len(vecs) <= 1:
        return np.array([0])
    dots = np.dot(vecs[:-1], vecs[1:].T).diagonal()
    return dots


def step_lengths(x, y):
    """
    Compute step lengths for a trajectory.
    
    Parameters
    ----------
    x : array-like
        x-coordinates for the trajectory
    y : array-like
        y-coordinates for the trajectory
        
    Returns
    -------
    array
        Step lengths
    """
    return np.sqrt(np.diff(x)**2 + np.diff(y)**2)





def compute_diffusion_features(trajectory, dt=1.0):
    """
    Compute diffusion features for a 2D trajectory and return them as a numpy array.
    
    Parameters
    ----------
    trajectory : array-like
        2D trajectory as an Nx2 array where N is the number of positions
    dt : float
        Time step between positions
        
    Returns
    -------
    tuple
        (features_array, feature_names)
        - features_array: numpy array of computed features
        - feature_names: list of feature names in the same order as the array
    """

    # Extract x and y coordinates
    x = trajectory[:, 0]
    y = trajectory[:, 1]
    
    # Ensure we have enough points
    if len(x) < 3:
        return np.array([np.nan] * len(feature_names)), feature_names
    
    # Compute basic metrics
    msd_values = msd(x, y)
    max_dist = get_max_dist(x, y)
    sl = step_lengths(x, y)
    dots = dot_product_traces(np.array([x, y]).T)
    
    # Fit diffusion model (normal diffusion only)
    params, r_squared = fit_diffusion_scaling(msd_values, dt, dim=2)
    
    # Extract parameters
    D = params[0]  # Diffusion coefficient
    alpha = params[1]  # Anomalous exponent
    
    # Compute all the features
    eff_log, eff = efficiency(x, y)
    
    # Create features array in the same order as feature_names
    features = np.array([
        alpha,
        D,
        r_squared,
        eff_log,
        eff,
        fractal_dim(x, y, max_dist),
        gaussianity(x, y, msd_values),
        kurtosis(x, y),
        msd_ratio(msd_values),
        trappedness(x, max_dist, params),
        len(x),
        np.nanmean(sl),
        np.nanmean(msd_values),
        np.nanmean(dots) if len(dots) > 0 else np.nan,
        np.nanmean(np.sign(dots[1:]) == np.sign(dots[:-1])) if len(dots) > 1 else np.nan,
        np.nanmean(np.sign(dots) > 0) if len(dots) > 0 else np.nan,
        np.nansum(sl),
        np.nanmin(sl) if len(sl) > 0 else np.nan,
        np.nanmax(sl) if len(sl) > 0 else np.nan,
        np.nanmax(sl) - np.nanmin(sl) if len(sl) > 0 else np.nan,
        np.nansum(sl) / len(x) if len(x) > 0 else np.nan,
        np.nanstd(sl, ddof=1) / np.nanmean(sl) if np.nanmean(sl) > 0 and len(sl) > 1 else np.nan,
        np.nansum(sl < 0.1) / len(sl) if len(sl) > 0 else np.nan,
        np.nansum(sl > 0.4) / len(sl) if len(sl) > 0 else np.nan,
        convex_hull_area(x, y)
    ])
    
    return features
    


def compute_features_for_multiple_trajectories(trajectories, dt=1):
    """
    Compute features for multiple particle trajectories.
    
    Parameters:
    -----------
    trajectories : numpy.ndarray
        Array of particle trajectories with shape (P, N, 2), where:
        - P is the number of particles
        - N is the number of positions per trajectory
        - 2 represents x and y coordinates
    
    Returns:
    --------
    numpy.ndarray
        Array of features with shape (P, 24), where:
        - P is the number of particles
        - 24 is the number of features computed for each trajectory
    """
    # Get the number of particles
    num_particles = trajectories.shape[0]
    
    # Initialize the output features array
    features = np.zeros((num_particles, N_features))
    
    # Compute features for each particle's trajectory
    for i in range(num_particles):
        # Extract the current trajectory
        trajectory = trajectories[i]
        
        # Compute features for this trajectory using the existing function
        # Assuming compute_trajectory_features is the original function
        particle_features = compute_diffusion_features(trajectory, dt)
        # Replace any NaN values with 0
        particle_features = np.nan_to_num(particle_features, nan=0.0)
        
        # Store the features
        features[i] = particle_features
    
    return features