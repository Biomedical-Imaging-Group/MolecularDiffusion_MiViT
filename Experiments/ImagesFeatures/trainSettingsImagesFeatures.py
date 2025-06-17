import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from helpers.models import *
from helpers.helpersGeneration import *
from helpers.helpersFeatures import *

# Define settings needed in the other files

sequences = False
center = True
# adaptive batch size doubles the size of batch every adaptive_batch_size cycles
# set to -1 if no adaptive batch_size 
adaptive_batch_size = 20
lr = 1e-4
D_max_normalization = 10


msdPerfect, msdFrame, msdLocalized = ("MSD_Perfect", "MSD_Frame", "MSD_Localized")
MSDModels = [msdPerfect, msdFrame, msdLocalized]
MSD_mult_factor = 250
MSD_mult_factor_avg = 37.5

localization_uncertainty = [0.0, 0.01]

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


# feature generation parameters
dt = 1
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




im_resnet, im_ft_resnet = "im_resnet", "im_ft_resnet"
ft_mlp = "ft_mlp"
im_tr, im_ft_early_tr, im_ft_late_tr = "im_tr", "im_ft_early_tr", "im_ft_late_tr"


name_map = {
    msdLocalized: "MSD Localized",
    msdFrame: "MSD Frame",
    msdPerfect: "MSD Perfect",
    im_resnet: "CNN only",
    im_ft_resnet : "CNN + Feat",
    ft_mlp : "Feat only",
    im_tr: "Transf(CNN)",
    im_ft_early_tr: "Transf(CNN + Feat)",
    im_ft_late_tr: "Transfo(CNN) + Feat",
    "im_tr_rot" : "Transf(CNN) Aug",
    "im_res_rot": "CNN only Aug",
    "im_ft_res_rot": "CNN + Feat Aug",
    "im_ft_tr_rot" : "Transf(CNN + Feat) Aug"
}

# Change this function to 
def getTrainingModels(lr=1e-4, addMSDModels=False):
    # Get all transformer models, _s stands for small, _b for big models
    embed_kwargs = {"patch_size": patch_size, "embed_dim": embed_dim}
    # Define MLP heads
    twoLayerMLP = MLPHead

    # Define model instances
    models = {
        im_tr: GeneralTransformer(
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
        ),
        im_ft_late_tr: GeneralTransformer(
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
            single_prediction=single_prediction,
            use_global_features=True,
            fusion_type='late',
            global_feature_dim=N_features
        ),
        im_ft_early_tr: GeneralTransformer(
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
            single_prediction=single_prediction,
            use_global_features=True,
            fusion_type='early',
            global_feature_dim=N_features
        ),
    }
    
    resnet = MultiImageResNet(patch_size, single_prediction=single_prediction, activation=nn.ReLU)
    models.update({im_resnet: resnet})

    resnet_ft = MultiImageFeatureResNet(patch_size, N_features,feature_size=embed_dim, hidden_size=hidden_dim, activation=nn.ReLU)
    models.update({im_ft_resnet: resnet_ft})

    features_reg = MLPHead(input_dim=N_features)
    models.update({ft_mlp: features_reg})
    
    # Create 1 optimizer and scheuler for each model
    optimizers = {name: optim.AdamW(model.parameters(), lr=lr) for name, model in models.items()}
    schedulers = {name: optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.9) for name, opt in optimizers.items()}



    if(addMSDModels):
        models.update({msd_mod: None for msd_mod in MSDModels})

    return models, optimizers, schedulers


val_d_in_order = np.arange(0.1,10.01,0.1)
N_in_order = 10 # number of particles

def load_validation_data(length = 20, skip_inorder=False):

    length_values = [20,30]
    if( length not in length_values):
        ValueError(f"Invalid length value, select one in: {length_values}")

    trajs1 = np.load("../validation_trajectories/"+str(length)+"/val1.npy") /traj_div_factor
    trajs3 = np.load("../validation_trajectories/"+str(length)+"/val3.npy") /traj_div_factor
    trajs5 = np.load("../validation_trajectories/"+str(length)+"/val5.npy") /traj_div_factor
    trajs7 = np.load("../validation_trajectories/"+str(length)+"/val7.npy") /traj_div_factor
    trajs9 = np.load("../validation_trajectories/"+str(length)+"/val9.npy") /traj_div_factor

    trajs_in_order = np.load("../validation_trajectories/valTrajsInOrder.npy") /traj_div_factor


    vid1, ft1, trajs_1 = create_video_and_feature_pairs(trajs1,nPosPerFrame,center=center,image_props=image_props, localization_uncertainty=localization_uncertainty, dt=dt)
    vid3, ft3, trajs_3 = create_video_and_feature_pairs(trajs3,nPosPerFrame,center=center,image_props=image_props, localization_uncertainty=localization_uncertainty, dt=dt)
    vid5, ft5, trajs_5 = create_video_and_feature_pairs(trajs5,nPosPerFrame,center=center,image_props=image_props, localization_uncertainty=localization_uncertainty, dt=dt)
    vid7, ft7, trajs_7 = create_video_and_feature_pairs(trajs7,nPosPerFrame,center=center,image_props=image_props, localization_uncertainty=localization_uncertainty, dt=dt)
    vid9, ft9, trajs_9 = create_video_and_feature_pairs(trajs9,nPosPerFrame,center=center,image_props=image_props, localization_uncertainty=localization_uncertainty, dt=dt)

    if(skip_inorder):
        vid_inorder = np.zeros(1)
        ft_inorder = np.zeros(1)
        trajs_inorder = np.zeros(1)

    else:
        trajs_in_order = trajs_in_order.reshape(-1,T,2)
        vid_inorder, ft_inorder, trajs_inorder = create_video_and_feature_pairs(trajs_in_order,nPosPerFrame,center=center,image_props=image_props, localization_uncertainty=localization_uncertainty, dt=dt)
        vid_inorder = vid_inorder.reshape(len(val_d_in_order),10,nFrames,patch_size,patch_size)
        ft_inorder = ft_inorder.reshape(len(val_d_in_order),10,N_features)

    return (torch.Tensor(vid1),torch.Tensor(ft1), trajs_1), (torch.Tensor(vid3),torch.Tensor(ft3), trajs_3), (torch.Tensor(vid5),torch.Tensor(ft5), trajs_5), (torch.Tensor(vid7),torch.Tensor(ft7), trajs_7),(torch.Tensor(vid9),torch.Tensor(ft9), trajs_9), (torch.Tensor(vid_inorder),torch.Tensor(ft_inorder), trajs_inorder)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


import numpy as np

def d_fromMSDTau1(trajectories):
    """
    Computes the MSD at lag tau=1 for multiple particle trajectories.

    Parameters:
    - trajectories (ndarray): shape (nparticles, num_steps, 2), positions over time.

    Returns:
    - msd_tau1 (ndarray): shape (nparticles,), MSD at tau=1 for each particle.
    """
    # Difference between consecutive steps: shape (nparticles, num_steps - 1, 2)
    deltas = trajectories[:, 1:] - trajectories[:, :-1]

    # Squared displacements: shape (nparticles, num_steps - 1)
    squared_displacements = np.sum(deltas**2, axis=2)

    # Average over time steps for each particle
    msd_tau1 = np.mean(squared_displacements, axis=1)

    return msd_tau1


def generate_rotated_sequences(x):
    """
    Given a tensor of shape (B, T, H, W), return a tuple of 4 tensors:
    - Original
    - Rotated 90°
    - Rotated 180°
    - Rotated 270°
    """
    if x.ndim != 4:
        raise ValueError(f"Expected tensor of shape (B, T, H, W), got {x.shape}")
    
    original = x
    rot90   = torch.rot90(x, k=1, dims=(2, 3))
    rot180  = torch.rot90(x, k=2, dims=(2, 3))
    rot270  = torch.rot90(x, k=3, dims=(2, 3))

    return (original, rot90, rot180, rot270)


def predict_with_rotations(model, images, features=None):
    """
    Apply model to images and their 90°, 180°, and 270° rotations.
    Return the mean prediction over the 4 versions.
    
    Args:
        model: a model that takes input of shape (B, T, H, W)
        images: tensor of shape (B, T, H, W)
    
    Returns:
        Tensor of shape (B, 1): averaged predictions
    """
    # Generate all rotations
    rotations = generate_rotated_sequences(images)  # tuple of 4 tensors

    # Get predictions for each rotated sequence
    if(features != None):
        preds = [model(rot, features) for rot in rotations]  # each (B, 1)
    else:
        preds = [model(rot) for rot in rotations]  # each (B, 1)


    # Stack predictions along new dimension and average
    stacked = torch.stack(preds, dim=0)  # shape (4, B, 1)
    mean_pred = stacked.mean(dim=0)      # shape (B, 1)

    return mean_pred


def make_prediction(model, name, images, features, trajectories, msd_mult_fact = MSD_mult_factor, eval=True):
    images = images.to(device)
    features = features.to(device)

    predictions = None

    if("MSD" not in name and eval):
        model.eval()

    if(name == "im_resnet"):
        if("rot" in name):
            predictions = predict_with_rotations(model,images)
        else :
            predictions = model(images)
    elif(name == "im_ft_resnet"):
        if("rot" in name):
            predictions = predict_with_rotations(model,images, features)
        else :
            predictions = model(images, features)
    elif(name == "ft_mlp"): 
        predictions = model(features)
    elif(name == "MSD_Perfect"):
        predictions = d_fromMSDTau1(trajectories[0]) * msd_mult_fact
    elif(name == "MSD_Frame"):
        predictions = d_fromMSDTau1(trajectories[1]) * MSD_mult_factor_avg
    elif(name == "MSD_Localized"):
        predictions = d_fromMSDTau1(trajectories[2]) * MSD_mult_factor_avg
    else:
        with torch.no_grad():
            if("ft" not in name):
                features = None

            if("rot" in name):
                predictions = predict_with_rotations(model,images, features)
            else :
                predictions = model(images, features)

    
    return predictions





def make_prediction_tuple(model, name, tuple):
    images = tuple[0]
    features = tuple[1]
    trajectories = tuple[2]

    return make_prediction(model, name, images, features, trajectories)