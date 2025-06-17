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

originalNposPerFrame = 10
nPosPerFrame = [5,10,15,20,30]
N_POSPERFRAME = len(nPosPerFrame)
nFrames = 30 # = Seuence length
T = nFrames * originalNposPerFrame
# number of trajectories



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
    for i,x in enumerate(nPosPerFrame):
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

            models.update({f"tr_{i}": tr, f"res_{i}": res})
    
    # Create 1 optimizer and scheuler for each model
    optimizers = {name: optim.AdamW(model.parameters(), lr=lr) for name, model in models.items()}
    schedulers = {name: optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.9) for name, opt in optimizers.items()}
    
    return models, optimizers, schedulers

class ImageDatasetFrameRate(Dataset):
    def __init__(self, images, labels):
        self.images = images  # Shape (N, C, H, W)
        self.labels = labels  # Shape (N, 1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return [self.images[i][idx] for i,x in enumerate(self.images)], self.labels[idx]
    

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


    vid1 = trajs_to_vid_framerates(trajs1,nPosPerFrame,center=center,image_props=image_props)
    vid3 = trajs_to_vid_framerates(trajs3,nPosPerFrame,center=center,image_props=image_props)
    vid5 = trajs_to_vid_framerates(trajs5,nPosPerFrame,center=center,image_props=image_props)
    vid7 = trajs_to_vid_framerates(trajs7,nPosPerFrame,center=center,image_props=image_props)
    vid9 = trajs_to_vid_framerates(trajs9,nPosPerFrame,center=center,image_props=image_props)

    if(skip_inorder):
        vid_inorder = np.zeros(1)

    else:
        trajs_in_order = trajs_in_order.reshape(-1,T,2)
        vid_inorder = trajs_to_vid_framerates(trajs_in_order,nPosPerFrame,center=center,image_props=image_props)

    return vid1, vid3,vid5,vid7,vid9,vid_inorder



def make_prediction(model, name, images, eval=True):

    prefix, frameR_index = name.split("_")

    input = images[int(frameR_index)]
    predictions = model(input)

    
    return predictions



def trajs_to_vid_framerates(
    trajectories,
    nPosPerFrame = [],
    center = False,
    image_props={},
    ):
    N,T,_ = trajectories.shape

    part_flux, part_std = image_props["particle_intensity"]

    N_POSPERFRAME = len(nPosPerFrame)

    allVideos = []

    for nSubPos in nPosPerFrame:
        if(T % nSubPos != 0):
            raise Exception("T is not divisble by posPerFrame")

        part_flux_FrameRate = part_flux * (nSubPos/originalNposPerFrame)
        image_props_framerate = image_props.copy()
        image_props_framerate["particle_intensity"]  = [part_flux_FrameRate, part_std]

        videos = trajectories_to_video(trajectories, nSubPos, center=center, image_props=image_props_framerate)
        videos = torch.Tensor(videos)

        allVideos.append(videos)

    return allVideos