from models import *
from helpersGeneration import *

import torch
import torch.nn as nn

class RelativeMSELoss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(RelativeMSELoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, pred, target):
        relative_error = (pred - target) / (target + self.epsilon)
        return torch.mean(relative_error ** 2)



# Define settings needed in the other files

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
    loss_function = RelativeMSELoss()
    val_loss_function = nn.MSELoss(reduction='none')
    single_prediction = True
    use_regression_token = True
    use_pos_encoding = True
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
# values from Real data
background_mean,background_sigma = 1420, 290
part_mean, part_std = 6000 - background_mean,500

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
        background_sigma,
    ],  # Standard deviation of background intensity within a video
    "poisson_noise": 100,
    "trajectory_unit" : 1200
}



RL_iterations = [2,5,10]
settings = ["no_noise", "gaussian_noise", "poisson_noise","gauss_filter"]

for rl in RL_iterations:
    settings.append("RL_" + str(rl))


def r_name(setting):
    return "resnet_" + setting

def t_name(setting):
    return "trans_" + setting

def images_idx_from_name(name):
    if "no_noise" in name:
        return 0
    elif "gaussian_noise" in name:
        return 1
    elif "poisson_noise" in name:
        return 2
    elif "gauss_filter" in name:
        return 3
    else:
        if("RL" in name):
            return 4 + RL_iterations.index(int(name.split("_")[-1]))

        else:
            return -1
    
val_d_in_order = np.arange(0.1,7.01,0.1)
N_in_order = 10 # number of particles


embed_kwargs = {"patch_size": patch_size, "embed_dim": embed_dim}
# Define MLP heads
twoLayerMLP = nn.Sequential(
    nn.Linear(embed_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, 1)  # Output a single scalar value
)
def getModels():
    models, optimizers, schedulers = {}, {}, {}

    for i, setting in enumerate(settings):
        trans = GeneralTransformer(
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
        
        t_opt = optim.AdamW(trans.parameters(), lr=lr)
        t_sced = optim.lr_scheduler.StepLR(t_opt, step_size=5, gamma=0.9)
        models[t_name(setting)] = trans
        optimizers[t_name(setting) ] = t_opt
        schedulers[t_name(setting) ] = t_sced

        resnet = MultiImageLightResNet(patch_size, single_prediction=single_prediction, activation=nn.ReLU)
        r_opt = optim.AdamW(resnet.parameters(), lr=lr)
        r_sced = optim.lr_scheduler.StepLR(r_opt, step_size=5, gamma=0.9)
        models[r_name(setting)] = resnet
        optimizers[r_name(setting) ] = r_opt
        schedulers[r_name(setting) ] = r_sced
    return models, optimizers, schedulers
