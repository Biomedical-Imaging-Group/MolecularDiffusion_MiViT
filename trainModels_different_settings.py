from models import *
from andi_datasets.models_phenom import models_phenom
from helpersGeneration import *
from helpersPlot import *
from trainSettings import *
import datetime
from scipy.signal import fftconvolve

def tv_gradient(image):
    """Compute the gradient of total variation."""
    grad = np.zeros_like(image)
    dx = np.diff(image, axis=1, append=image[:, -1:])
    dy = np.diff(image, axis=0, append=image[-1:, :])
    eps = 1e-8
    mag = np.sqrt(dx**2 + dy**2 + eps)
    dx_norm = dx / mag
    dy_norm = dy / mag
    grad[:, :-1] -= dx_norm[:, :-1]
    grad[:, 1:] += dx_norm[:, :-1]
    grad[:-1, :] -= dy_norm[:-1, :]
    grad[1:, :] += dy_norm[:-1, :]
    return grad

def richardson_lucy_tv(image, psf, iterations=20, tv_weight=0.01):
    image = np.clip(image, 1e-6, None)
    psf_mirror = psf[::-1, ::-1]
    estimate = np.full(image.shape, 0.5, dtype=np.float32)
    for i in range(iterations):
        relative_blur = image / (fftconvolve(estimate, psf, mode='same') + 1e-6)
        correction = fftconvolve(relative_blur, psf_mirror, mode='same')
        estimate *= correction
        tv_grad = tv_gradient(estimate)
        estimate -= tv_weight * tv_grad
        estimate = np.clip(estimate, 0, 1)

    return estimate


def create_gaussian_psf(size=patch_size, sigma=1.3):
    if size % 2 == 0:
        size += 1  # ensure odd size for symmetry
    ax = np.arange(-size // 2 + 1, size // 2 + 1)
    x, y = np.meshgrid(ax, ax)
    psf = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    psf /= psf.sum()
    return psf

import torch
import numpy as np

def apply_rl_tv_tensor(tensor, psf, n_iters=10, tv_weight=0.01):
    B, seq, H, W = tensor.shape
    assert H == 9 and W == 9, "Only images of shape 9x9 are supported"
    
    tensor_np = tensor.detach().cpu().numpy()  # Convert to NumPy
    result_np = np.empty_like(tensor_np)

    for b in range(B):
        for t in range(seq):
            result_np[b, t] = richardson_lucy_tv(tensor_np[b, t], psf, iterations=n_iters, tv_weight=tv_weight)

    return torch.tensor(result_np, dtype=tensor.dtype, device=tensor.device)

psf = create_gaussian_psf(sigma=1)


def load_validation_dataMult(length = 30):

    length_values = [20,30]
    if( length not in length_values):
        ValueError(f"Invalid length value, select one in: {length_values}")

    trajs1 = np.load("./valTrajs"+str(length)+"/val1.npy") /traj_div_factor
    trajs3 = np.load("./valTrajs"+str(length)+"/val3.npy") /traj_div_factor
    trajs5 = np.load("./valTrajs"+str(length)+"/val5.npy") /traj_div_factor
    trajs7 = np.load("./valTrajs"+str(length)+"/val7.npy") /traj_div_factor
    trajs_in_order = np.load("./valTrajsInOrder.npy") /traj_div_factor


    vid1 = trajectories_to_video_multiple_settings(trajs1,nPosPerFrame,center=True,image_props=image_props)
    vid1 = np.array(vid1)
    vid1,_ = normalize_images(vid1,background_mean,background_sigma,part_mean+background_mean)

    vid3 = trajectories_to_video_multiple_settings(trajs3,nPosPerFrame,center=True,image_props=image_props)
    vid3 = np.array(vid3)
    vid3,_ = normalize_images(vid3,background_mean,background_sigma,part_mean+background_mean)

    vid5 = trajectories_to_video_multiple_settings(trajs5,nPosPerFrame,center=True,image_props=image_props)
    vid5 = np.array(vid5)
    vid5,_ = normalize_images(vid5,background_mean,background_sigma,part_mean+background_mean)

    vid7 = trajectories_to_video_multiple_settings(trajs7,nPosPerFrame,center=True,image_props=image_props)
    vid7 = np.array(vid7)
    vid7,_ = normalize_images(vid7,background_mean,background_sigma,part_mean+background_mean)


    trajs_in_order = trajs_in_order.reshape(-1,T,2)
    vid_in_order =  trajectories_to_video_multiple_settings(trajs_in_order,nPosPerFrame,center=True,image_props=image_props)
    vid_in_order = np.array(vid_in_order)
    vids = []
    for i in range(5):
        vids.append( vid_in_order[i].reshape(len(val_d_in_order),10,nFrames,patch_size,patch_size))
    vid_in_order = np.array(vids)
    vid_in_order,_ = normalize_images(vid_in_order,background_mean,background_sigma,part_mean+background_mean)

    return torch.Tensor(vid1),torch.Tensor(vid3), torch.Tensor(vid5), torch.Tensor(vid7), torch.Tensor(vid_in_order)

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
        return 4
    else:
        return 3
def rl_number(name):
    try:
        return int(name.split("_")[-1])
    except (IndexError, ValueError):
        return None  # Handle unexpected format like "RL_" or "RL_abc"
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

embed_kwargs = {"patch_size": patch_size, "embed_dim": embed_dim}
# Define MLP heads
twoLayerMLP = nn.Sequential(
    nn.Linear(embed_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, 1)  # Output a single scalar value
)

settings = ["no_noise", "gaussian_noise", "poisson_noise", "RL_2","gauss_filter","RL_5","RL_10", "RL_20"]
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

### Models Settings ###
# Most of models settings are in Train_settings for unity
# Type of training: mixTrajectories for learning trajectories with a switch in the middle
mix_trajectories = True and not single_prediction


for name in models:
    models[name] = models[name].to(device)

### Training Settings ###
num_cycles = 100  # Number of dataset refreshes
batch_size = 1 if adaptive_batch_size != -1 else 16 # Number of sequences in 1 batch
shuffle = True # if trajectories should be shuffled during training
N = 64 # Number of sequences in per value of D in Trainings_Ds
# Mean and variance of the trajectories of Ds
#TrainingDs_list = [[1, 1], [3, 1], [5, 1], [7, 1]]
TrainingDs_list = [[1, 1], [3, 1], [5, 1], [7, 1]]

printParams = True
verbose = False


# Load validation dataset (fixed, does not change across cycles)
val_videos = load_validation_dataMult(nFrames)  # Returns (vid1, vid3, vid5, vid7)
val_labels = torch.tensor([1, 3, 5, 7], dtype=torch.float32)  # Corresponding labels

# Divide Labels by D_max to have values between 0 and 1 -> better optimizers
val_labels = val_labels / D_max_normalization


# Dictionary to store validation losses per model and dataset type
validation_losses = {name: {f"val_{label.item()}": [] for label in val_labels} for name in models.keys()}
for models_name in validation_losses.keys():
    validation_losses[models_name].update({"val_avg":[]})
# create all_labels to be plotted later
all_gen_labels = np.array([])  # Empty array to start



if(printParams):
    print("Starting Training with parameters:")
    # Training Settings
    print("### Training Settings ###")
    print(f"Num Cycles: {num_cycles}")
    print(f"Batch Size: {batch_size}")
    print(f"Sequences per D: {N}")
    print(f"Training D List: {TrainingDs_list}")
    print(f"Verbose: {verbose}")
    print(f"Mix Trajectories: {mix_trajectories}")
    print()

    # Model Settings
    print("### Model Hyperparameters ###")
    print(f"Loss Function: {loss_function}")
    print(f"Single Prediction: {single_prediction}")
    print(f"Use Regression Token: {use_regression_token}")
    print(f"Use Positional Encoding: {use_pos_encoding}")
    print(f"Transformer Activation Function: {tr_activation_fct}")
    print()

    # Model Architecture
    print("### Model Architecture ###")
    print(f"Patch Size: {patch_size}")
    print(f"Embedding Dimension: {embed_dim}")
    print(f"Num Heads: {num_heads}")
    print(f"Hidden Dimension: {hidden_dim}")
    print(f"Num Layers: {num_layers}")
    print(f"Dropout: {dropout}")
    print()


    print("### Image Generation Parameters ###")
    print(f"Trajectory Division Factor: {traj_div_factor}")
    print(f"Positions Per Frame: {nPosPerFrame}")
    print(f"Number of Frames: {nFrames}")
    print(f"Total Time Steps: {T}")
    print(f"Image Properties: {image_props}")
    print()


    print("StartTime: ",datetime.datetime.now())


for cycle in range(num_cycles):
    
    # adaptive batch size doubles the size of batch every adaptive_batch_size cycles
    #https://arxiv.org/abs/1712.02029
    if(adaptive_batch_size != -1 and cycle != 0 and cycle % adaptive_batch_size == 0):
        batch_size = batch_size * 2
        print(f"Cycle: {cycle} new batch size: {batch_size}")

    print(f"Cycle {cycle+1} out of {num_cycles}: {(cycle+1)/num_cycles * 100:.2f}%")
    # Generate a new batch of images and labels
    all_videos = []
    all_labels = []

    for TrainingDs in TrainingDs_list:
        # Generate a new batch of 2000 images and labels
        trajs, labels = models_phenom().single_state(N, 
                                    L=0,
                                    T=T,
                                    Ds=TrainingDs,  # Mean and variance
                                    alphas=1)
        # Reshape trajectories
        trajs = trajs.transpose(1, 0, 2)
        labels = labels.transpose(1, 0, 2)

        # For reporting stats save generated labels 
        all_gen_labels = np.append(all_gen_labels,labels[:, 0, 1])



        if single_prediction:
            labels = labels[:, 0, 1]
        else:
            labels = labels[:, :, 1]
            labels = labels.reshape(labels.shape[0], -1, nPosPerFrame).mean(axis=2)

        # Store all labels
        all_labels.append(labels)


        # Convert trajectories of D (pixels/s) to D (micro_m/ms)
        trajs = trajs / traj_div_factor
        videos = trajectories_to_video_multiple_settings(trajs, nPosPerFrame, center=center, image_props=image_props)
        # videos is a tuple of 5 arrays of shape (N, length, P, P)
        # Stack along a new axis to get shape (N, 5, length, P, P)
        videos = np.stack(videos, axis=1)
        # Normalize videos (handle batch of shape (N, 5, length, P, P))
        videos, _ = normalize_images(videos, background_mean, background_sigma, part_mean + background_mean)
        # Append to list

        all_videos.append(videos)

    # Concatenate all generated data
    all_videos = np.concatenate(all_videos, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    # Divide Labels by D_max to have values between 0 and 1 -> better optimizers
    all_labels = all_labels / D_max_normalization



    # Convert to tensors
    all_videos = torch.Tensor(all_videos)
    all_labels = torch.Tensor(all_labels).unsqueeze(-1)  # Add an extra dimension for single_prediction

    # Create a dataset and shuffle
    dataset = ImageDataset(all_videos, all_labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle= mix_trajectories or shuffle)


    # Training Loop
    for name, model in models.items():
        model.train()
        optimizer = optimizers[name]
        scheduler = schedulers[name]

        for batch_images, batch_labels in dataloader:
            
            idx = images_idx_from_name(name)

            batch_images = batch_images[:,idx,:]

            if("RL" in name):
                rl_iters = rl_number(name)
                batch_images = apply_rl_tv_tensor(batch_images, psf, n_iters=rl_iters)

            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            
            predictions = model(batch_images)

            loss = loss_function(predictions, batch_labels)
            loss.backward()
            optimizer.step()

        # Step the learning rate scheduler after training
        scheduler.step()
        
        if(verbose):
            for p_name, param in model.named_parameters():
                if param.grad is not None:
                    print(f"{name}: {p_name}: Mean : {param.grad.abs().mean().item():.8f} Max:  {param.grad.abs().max().item():.8f}")
                    #print(model.reg_token)


    # Evaluation part
    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            label_losses = []
            for vid, label_value in zip(val_videos, val_labels):
                vid = vid.to(device)

                idx = images_idx_from_name(name)

                vid = vid[:,idx,:]

                if("RL" in name):
                    rl_iters = rl_number(name)
                    batch_images = apply_rl_tv_tensor(batch_images, psf, n_iters=rl_iters)
                

                # Adjust label shape based on single_prediction
                if not model.single_prediction:
                    batch_size, num_images, _, _ = vid.shape
                    label = torch.full((batch_size, num_images, 1), label_value, device=device)  # Shape: [batch_size, num_images, 1]
                else:
                    label = torch.full((vid.shape[0],), label_value, device=device).view(-1, 1)  # Shape: [batch_size, 1]
                
                val_predictions = model(vid)
                val_loss = loss_function(val_predictions, label)
                avg_val_loss = val_loss.item()
                validation_losses[name][f"val_{label_value.item()}"].append(avg_val_loss)
                if(verbose):
                    print(f"{name} on val_{label_value.item()}: Validation Loss = {avg_val_loss:.4f}")
                label_losses.append(avg_val_loss)
            # Compute average across all labels
            avg_val_avg = np.mean(label_losses)  # Use numpy for mean calculation
            validation_losses[name]["val_avg"].append(avg_val_avg)
            if(verbose):
                print(f"{name} on val_avg: Validation Loss = {avg_val_avg:.4f}")




print(f"Number of generated sequences: {all_gen_labels.shape}")
# --- Save everything for later analysis ---
save_path = "training_results.pth"
results = {
    "validation_losses": validation_losses,
    "all_labels": all_gen_labels,  # Convert to NumPy for easier histogram plotting
    "model_weights": {name: model.state_dict() for name, model in models.items()}
}
torch.save(results, save_path)
print(f"\nTraining results saved to {save_path}")
print(datetime.datetime.now())
