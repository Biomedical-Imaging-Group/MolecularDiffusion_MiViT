from models import *
from helpersGeneration import *
from helpersFeatures import *

# Define settings needed in the other files
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sequences = False
center = True
# adaptive batch size doubles the size of batch every adaptive_batch_size cycles
# set to -1 if no adaptive batch_size 
adaptive_batch_size = 20
lr = 1e-4
D_max_normalization = 10



if(sequences):
    # Models settings
    loss_function = nn.MSELoss()
    val_loss_function = nn.MSELoss(reduction='none')
    single_prediction = False
    use_regression_token = False
    use_pos_encoding = True
    tr_activation_fct = F.relu
else:
    # Models settings
    loss_function = nn.MSELoss()
    val_loss_function = nn.MSELoss(reduction='none')
    single_prediction = True
    use_regression_token = True
    use_pos_encoding = False
    tr_activation_fct = F.relu


# tr_activation_fct = F.LeakyReLU, F.GELU F.ReLU

# Define model hyperparameters
patch_size = 9
embed_dim = 64
num_heads = 4
hidden_dim = 128
num_layers = 6
dropout = 0.0


# Image generation parameters
traj_div_factor = 100 # need to divide trajectories because they are given in pixels/s but we want trajectories in ms domain

nPosPerFrame = 10 
nFrames = 30 # = Seuence length
T = nFrames * nPosPerFrame
# number of trajectories

PSF_Settings = [2, 1.75, 1.5, 1.25, 1]
Noise_Settings = [0, 1/50, 1/25, 1/20, 1/10, 1/5]

N_PSF, N_Noise = len(PSF_Settings), len(Noise_Settings)


# values from Real data
background_mean = 5000
part_mean, part_std = 5000 ,500

image_props = {
    "particle_intensity": [
        part_mean,
        part_std,
    ],  # Mean and standard deviation of the particle intensity
    "NA": 1.46,  # Numerical aperture
    "wavelength": 500e-9,  # Wavelength
    "psf_division_factor": 1.3,  
    "resolution": 100e-9,  # Camera resolution or effective resolution, aka pixelsize
    "output_size": patch_size,
    "upsampling_factor": 5,
    "background_intensity": [
        background_mean,
        0,
    ],  # Standard deviation of background intensity within a video
    "poisson_noise": 100,
    "trajectory_unit" : 1200
}



# Change this function to 
def getTrainingModels(lr=1e-4):
    # Get all transformer models, _s stands for small, _b for big models
    embed_kwargs = {"patch_size": patch_size, "embed_dim": embed_dim}
    # Define MLP heads
    twoLayerMLP = MLPHead

    models = {}
    # Define model instances
    for psf_index in range(N_PSF):
        for noise_index in range(N_Noise):
            tr = GeneralTransformer(
                embedding_cls=DeepResNetEmbedding,
                embed_kwargs=embed_kwargs,
                embed_dim=embed_dim,
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                mlp_head=twoLayerMLP,
                tr_activation_fct=tr_activation_fct,
                dropout=dropout,
                use_pos_encoding=use_pos_encoding,
                use_regression_token=use_regression_token,
                single_prediction=single_prediction
            )
            res = MultiImageResNet(patch_size, single_prediction=single_prediction, activation=nn.ReLU)

            models.update({f"tr_{psf_index}_{noise_index}": tr, f"res_{psf_index}_{noise_index}": res})
    
    # Create 1 optimizer and scheuler for each model
    optimizers = {name: optim.AdamW(model.parameters(), lr=lr) for name, model in models.items()}
    schedulers = {name: optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.9) for name, opt in optimizers.items()}



    
    return models, optimizers, schedulers


val_d_in_order = np.arange(0.1,10.01,0.1)
N_in_order = 10 # number of particles

def load_validation_data(length = 20, skip_inorder=False):

    length_values = [20,30]
    if( length not in length_values):
        ValueError(f"Invalid length value, select one in: {length_values}")

    trajs1 = np.load("./valTrajs"+str(length)+"/val1.npy") /traj_div_factor
    trajs3 = np.load("./valTrajs"+str(length)+"/val3.npy") /traj_div_factor
    trajs5 = np.load("./valTrajs"+str(length)+"/val5.npy") /traj_div_factor
    trajs7 = np.load("./valTrajs"+str(length)+"/val7.npy") /traj_div_factor
    trajs9 = np.load("./valTrajs"+str(length)+"/val9.npy") /traj_div_factor

    trajs_in_order = np.load("./valTrajsInOrderImFt.npy") /traj_div_factor


    vid1 = trajs_to_vid_psf_noise(trajs1,nPosPerFrame,center=center,image_props=image_props, PSF_Settings=PSF_Settings, Noise_Settings=Noise_Settings)
    vid3 = trajs_to_vid_psf_noise(trajs3,nPosPerFrame,center=center,image_props=image_props, PSF_Settings=PSF_Settings, Noise_Settings=Noise_Settings)
    vid5 = trajs_to_vid_psf_noise(trajs5,nPosPerFrame,center=center,image_props=image_props, PSF_Settings=PSF_Settings, Noise_Settings=Noise_Settings)
    vid7 = trajs_to_vid_psf_noise(trajs7,nPosPerFrame,center=center,image_props=image_props, PSF_Settings=PSF_Settings, Noise_Settings=Noise_Settings)
    vid9 = trajs_to_vid_psf_noise(trajs9,nPosPerFrame,center=center,image_props=image_props, PSF_Settings=PSF_Settings, Noise_Settings=Noise_Settings)

    if(skip_inorder):
        vid_inorder = np.zeros(1)

    else:
        trajs_in_order = trajs_in_order.reshape(-1,T,2)
        vid_inorder = trajs_to_vid_psf_noise(trajs_in_order,nPosPerFrame,center=center,image_props=image_props, PSF_Settings=PSF_Settings, Noise_Settings=Noise_Settings)
        vid_inorder = vid_inorder.reshape(len(val_d_in_order),N_in_order,N_PSF,N_Noise,nFrames,patch_size,patch_size)

    return (torch.Tensor(vid1)), (torch.Tensor(vid3)),(torch.Tensor(vid5)),(torch.Tensor(vid7)),(torch.Tensor(vid9)),(torch.Tensor(vid_inorder))



def make_prediction(model, name, images, eval=True):

    prefix, psf_index, noise_index = name.split("_")

    input = images[:, int(psf_index), int(noise_index)]
    predictions = model(input)

    
    return predictions


def select_models_from_psf(models, wanted_psf_index, wanted_prefix=None):
    selected_models = []
    for model_name in models.keys():
        prefix, model_psf_index, noise = model_name.split("_")
        if(int(model_psf_index) == wanted_psf_index):
            if(wanted_prefix == None or prefix==wanted_prefix):
                selected_models.append(model_name)
    return selected_models

def select_models_from_noise(models, wanted_noise_index, wanted_prefix=None):
    selected_models = []
    for model_name in models.keys():
        prefix, model_psf_index, model_noise_index = model_name.split("_")
        if(int(model_noise_index) == wanted_noise_index):
            if(wanted_prefix == None or prefix==wanted_prefix):
                selected_models.append(model_name)

    return selected_models



def trajs_to_vid_psf_noise(
    trajectories,
    nPosPerFrame,
    center = False,
    image_props={},
    PSF_Settings=[],
    Noise_Settings=[],
):
    
    N,T,_ = trajectories.shape


    if(T % nPosPerFrame != 0):
        raise Exception("T is not divisble by posPerFrame")
    if(PSF_Settings == [] or Noise_Settings == []):
        raise Exception("No settings given")

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

    # Psf is computed as wavelenght/2NA according to:
    #https://www.sciencedirect.com/science/article/pii/S0005272819301380?via%3Dihub
    fwhm_psf = _image_dict["wavelength"] / 2 * _image_dict["NA"]

    
    gaussian_sigma = upsampling_factor/ resolution * fwhm_psf/2.355
    poisson_noise = _image_dict["poisson_noise"]
    
    n_psf_settings = len(PSF_Settings)
    n_noise_settings = len(PSF_Settings)

    out_videos = np.zeros((N,N_PSF,N_Noise,nFrames,output_size,output_size),np.float32)


    particle_mean, particle_std = _image_dict["particle_intensity"][0],_image_dict["particle_intensity"][1]
    background_mean= _image_dict["background_intensity"][0]
    
    for n in range(N):
        traj_to_vid_psf_noise(out_videos[n],trajectories[n],nFrames,output_size,upsampling_factor,nPosPerFrame,
                                                gaussian_sigma,particle_mean,particle_std,background_mean, poisson_noise,center, PSF_Settings, Noise_Settings)
        
    return out_videos


def traj_to_vid_psf_noise(out_video,trajectory,nFrames, output_size, upsampling_factor, nPosPerFrame,gaussian_sigma,particle_mean,particle_std,background_mean, poisson_noise, center, PSF_Settings, Noise_Settings):
    """Helper function of function above, all arguments documented above"""


    for f in range(nFrames):
        frame_hr = np.zeros((N_PSF, output_size*upsampling_factor, output_size*upsampling_factor),np.float32)

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

                for psf_index in range(N_PSF):
                    
                    frame_spot = gaussian_2d(xtraj[p], ytraj[p], gaussian_sigma/PSF_Settings[psf_index], output_size*upsampling_factor, spot_intensity)

                    # gaussian_2d maximum is not always the wanted one because of some misplaced pixels. 
                    # We can force the peak of the gaussian to have the right intensity
                    spot_max = np.max(frame_spot)
                    if(spot_max < 0.00001):
                        print("Particle Left the image")
                    frame_hr[psf_index] += spot_intensity/spot_max * frame_spot
        for psf_index in range(N_PSF):
            out_video[psf_index,0,f] = block_reduce(frame_hr[psf_index], block_size=upsampling_factor, func=np.mean)

            for noise_index in range(N_Noise):
                bg_std = part_mean * Noise_Settings[noise_index]
                out_video[psf_index,noise_index,f] = out_video[psf_index,0,f] + np.clip(np.random.normal(background_mean, bg_std, out_video[psf_index,0,f].shape), 
                                            0, background_mean + 3 * bg_std)
                out_video[psf_index,noise_index,f]  = np.random.poisson(out_video[psf_index,noise_index,f] * poisson_noise) / poisson_noise



    return 