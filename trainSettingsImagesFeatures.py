from models import *
from helpersGeneration import *
from helpersFeatures import *

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




# Change this function to 
def getTrainingModels(lr=1e-4):
    # Get all transformer models, _s stands for small, _b for big models
    embed_kwargs = {"patch_size": patch_size, "embed_dim": embed_dim}
    # Define MLP heads
    twoLayerMLP = MLPHead

    # Define model instances
    models = {
        "im_tr": GeneralTransformer(
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
        "im_ft_late_tr": GeneralTransformer(
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
        "im_ft_early_tr": GeneralTransformer(
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
    
    resnet = MultiImageLightResNet(patch_size, single_prediction=single_prediction, activation=nn.ReLU)
    models.update({"im_resnet": resnet})

    features_reg = MLPHead(input_dim=N_features)
    models.update({"ft_mlp": features_reg})
    
    # Create 1 optimizer and scheuler for each model
    optimizers = {name: optim.AdamW(model.parameters(), lr=lr) for name, model in models.items()}
    schedulers = {name: optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.9) for name, opt in optimizers.items()}

    return models, optimizers, schedulers


val_d_in_order = np.arange(0.1,7.01,0.1)
N_in_order = 10 # number of particles

def load_validation_data(length = 20):

    length_values = [20,30]
    if( length not in length_values):
        ValueError(f"Invalid length value, select one in: {length_values}")

    trajs1 = np.load("./valTrajs"+str(length)+"/val1.npy") /traj_div_factor
    trajs3 = np.load("./valTrajs"+str(length)+"/val3.npy") /traj_div_factor
    trajs5 = np.load("./valTrajs"+str(length)+"/val5.npy") /traj_div_factor
    trajs7 = np.load("./valTrajs"+str(length)+"/val7.npy") /traj_div_factor
    trajs_in_order = np.load("./valTrajsInOrder.npy") /traj_div_factor


    vid1, ft1 = create_video_and_feature_pairs(trajs1,nPosPerFrame,center=center,image_props=image_props, dt=dt)
    vid3, ft3 = create_video_and_feature_pairs(trajs3,nPosPerFrame,center=center,image_props=image_props, dt=dt)
    vid5, ft5 = create_video_and_feature_pairs(trajs5,nPosPerFrame,center=center,image_props=image_props, dt=dt)
    vid7, ft7 = create_video_and_feature_pairs(trajs7,nPosPerFrame,center=center,image_props=image_props, dt=dt)


    trajs_in_order = trajs_in_order.reshape(-1,T,2)
    vid_inorder, ft_inorder = create_video_and_feature_pairs(trajs_in_order,nPosPerFrame,center=center,image_props=image_props, dt=dt)
    vid_inorder = vid_inorder.reshape(len(val_d_in_order),10,nFrames,patch_size,patch_size)
    ft_inorder = ft_inorder.reshape(len(val_d_in_order),10,N_features)

    return (torch.Tensor(vid1),torch.Tensor(ft1)), (torch.Tensor(vid3),torch.Tensor(ft3)), (torch.Tensor(vid5),torch.Tensor(ft5)), (torch.Tensor(vid7),torch.Tensor(ft7)), (torch.Tensor(vid_inorder),torch.Tensor(ft_inorder))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_prediction(model, name, images, features):
    images = images.to(device)
    features = features.to(device)

    predictions = None

    if(name == "im_resnet"):
        predictions = model(images)
    elif(name == "ft_mlp"): 
        predictions = model(features)
    else:
        if("ft" not in name):
            features = None

        predictions = model(images, features)
    
    return predictions


def make_prediction_pair(model, name, pair):
    images = pair[0]
    features = pair[1]

    return make_prediction(model, name, images, features)