from models import *
from helpersGeneration import *


# Define settings needed in the other files

sequences = False
center = True

if(sequences):
    # Models settings
    loss_function = nn.MSELoss()
    single_prediction = False
    use_regression_token = False
    use_pos_encoding = True
    tr_activation_fct = F.relu
else:
    # Models settings
    loss_function = nn.L1Loss()
    single_prediction = True
    use_regression_token = True
    use_pos_encoding = True
    tr_activation_fct = F.leaky_relu


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
part_mean, part_std = 5500,500
background_mean,background_sigma = 1400, 290

image_props = {
    "particle_intensity": [
        part_mean - background_mean,
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
    "trajectory_unit" : 500
}




# Change this function to 
def getTrainingModels(lr=1e-4):

    # Get all transformer models, _s stands for small, _b for big models
    models = get_transformer_models(patch_size, embed_dim, num_heads, hidden_dim, num_layers, dropout, use_pos_encoding=use_pos_encoding,tr_activation_fct=tr_activation_fct,name_suffix='_s', use_regression_token= use_regression_token, single_prediction=single_prediction)

    #models_very_small = get_transformer_models(patch_size, embed_dim//2, num_heads//2, hidden_dim//2, num_layers//2, dropout, use_pos_encoding=use_pos_encoding,tr_activation_fct=tr_activation_fct,name_suffix='_vs', use_regression_token= use_regression_token, single_prediction=single_prediction)
    #models.update(models_very_small)
    """
    cnn_big =  GeneralTransformer(
        embedding_cls=CNNEmbedding,
        embed_kwargs=embed_kwargs,
        embed_dim=embed_dim,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        mlp_head=twoLayerMLP,
        dropout=dropout,
        use_pos_encoding=use_pos_encoding,
        tr_activation_fct=tr_activation_fct,
        use_regression_token=use_regression_token,
        single_prediction=single_prediction
    )

    deep_cnn_big = GeneralTransformer(
                embedding_cls=DeepResNetEmbedding,
                embed_kwargs=embed_kwargs,
                embed_dim=embed_dim,
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                mlp_head=twoLayerMLP,
                dropout=dropout,
                use_pos_encoding=use_pos_encoding,
                tr_activation_fct=tr_activation_fct,
                use_regression_token=use_regression_token,
                single_prediction=single_prediction
            )
    models.update({"cnn_b": cnn_big})
    models.update({"deep_cnn_b": deep_cnn_big})

    """
    resnet = MultiImageLightResNet(patch_size, single_prediction=single_prediction, activation=nn.LeakyReLU)
    models.update({"resnet": resnet})

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


    vid1 = trajectories_to_video(trajs1,nPosPerFrame,center=True,image_props=image_props)
    vid1,_ = normalize_images(vid1,background_mean,background_sigma,part_mean+background_mean)

    vid3 = trajectories_to_video(trajs3,nPosPerFrame,center=True,image_props=image_props)
    vid3,_ = normalize_images(vid3,background_mean,background_sigma,part_mean+background_mean)

    vid5 = trajectories_to_video(trajs5,nPosPerFrame,center=True,image_props=image_props)
    vid5,_ = normalize_images(vid5,background_mean,background_sigma,part_mean+background_mean)

    vid7 = trajectories_to_video(trajs7,nPosPerFrame,center=True,image_props=image_props)
    vid7,_ = normalize_images(vid7,background_mean,background_sigma,part_mean+background_mean)


    trajs_in_order = trajs_in_order.reshape(-1,T,2)
    vid_in_order =  trajectories_to_video(trajs_in_order,nPosPerFrame,center=True,image_props=image_props)
    vid_in_order = vid_in_order.reshape(len(val_d_in_order),10,nFrames,patch_size,patch_size)
    vid_in_order,_ = normalize_images(vid_in_order,background_mean,background_sigma,part_mean+background_mean)

    return torch.Tensor(vid1),torch.Tensor(vid3), torch.Tensor(vid5), torch.Tensor(vid7), torch.Tensor(vid_in_order)









# Define model instances

def get_transformer_models(patch_size = patch_size, embed_dim = embed_dim, num_heads = num_heads, hidden_dim = hidden_dim, num_layers = num_layers,
                            dropout= dropout, use_pos_encoding = False, tr_activation_fct='gelu', use_regression_token=True , single_prediction = True, name_suffix = ''):
    """
    Returns different variants of the GeneralTransformer model.
    """


    embed_kwargs = {"patch_size": patch_size, "embed_dim": embed_dim}
    # Define MLP heads
    twoLayerMLP = nn.Sequential(
        nn.Linear(embed_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1)  # Output a single scalar value
    )

    oneLayerMLP = nn.Sequential(
        nn.Linear(embed_dim, 1)  # Output a single scalar value
    )
    # Define model instances
    models = {
        "linear_2layer" + name_suffix: GeneralTransformer(
            embedding_cls=LinearProjectionEmbedding,
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
        "linear_1layer"+ name_suffix: GeneralTransformer(
            embedding_cls=LinearProjectionEmbedding,
            embed_kwargs=embed_kwargs,
            embed_dim=embed_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            mlp_head=oneLayerMLP,
            tr_activation_fct=tr_activation_fct,
            dropout=dropout,
            use_pos_encoding=use_pos_encoding,
            use_regression_token=use_regression_token,
            single_prediction=single_prediction
        ),
        
        "cnn_1layer"+ name_suffix: GeneralTransformer(
            embedding_cls=CNNEmbedding,
            embed_kwargs=embed_kwargs,
            embed_dim=embed_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            mlp_head=oneLayerMLP,
            tr_activation_fct=tr_activation_fct,
            dropout=dropout,
            use_pos_encoding=use_pos_encoding,
            use_regression_token=use_regression_token,
            single_prediction=single_prediction
        ),
        "deepcnn_1layer"+ name_suffix: GeneralTransformer(
            embedding_cls=DeepResNetEmbedding,
            embed_kwargs=embed_kwargs,
            embed_dim=embed_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            mlp_head=oneLayerMLP,
            tr_activation_fct=tr_activation_fct,
            dropout=dropout,
            use_pos_encoding=use_pos_encoding,
            use_regression_token=use_regression_token,
            single_prediction=single_prediction
        ),
        "cnn_2layer"+ name_suffix: GeneralTransformer(
            embedding_cls=CNNEmbedding,
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
        "deepcnn_2layer"+ name_suffix: GeneralTransformer(
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
    }
    
    return models
