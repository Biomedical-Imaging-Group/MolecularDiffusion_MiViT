from models import *
from andi_datasets.models_phenom import models_phenom
from helpersGeneration import *
from helpersPlot import *
from models import get_transformer_models
import datetime
print(datetime.datetime.now())

# need to divide trajectories because they are given in pixels/s but we want trajectories in ms domain
traj_div_factor = 100

def load_validation_data():

    trajs1 = np.load("./valTrajs/val1.npy") /traj_div_factor
    trajs3 = np.load("./valTrajs/val3.npy") /traj_div_factor
    trajs5 = np.load("./valTrajs/val5.npy") /traj_div_factor
    trajs7 = np.load("./valTrajs/val7.npy") /traj_div_factor


    vid1 = trajectories_to_video(trajs1,nPosPerFrame,center=True,image_props=image_props)
    vid1,_ = normalize_images(vid1,background_mean,background_sigma,part_mean+background_mean)

    vid3 = trajectories_to_video(trajs3,nPosPerFrame,center=True,image_props=image_props)
    vid3,_ = normalize_images(vid3,background_mean,background_sigma,part_mean+background_mean)

    vid5 = trajectories_to_video(trajs5,nPosPerFrame,center=True,image_props=image_props)
    vid5,_ = normalize_images(vid5,background_mean,background_sigma,part_mean+background_mean)

    vid7 = trajectories_to_video(trajs7,nPosPerFrame,center=True,image_props=image_props)
    vid7,_ = normalize_images(vid7,background_mean,background_sigma,part_mean+background_mean)

    return torch.Tensor(vid1),torch.Tensor(vid3), torch.Tensor(vid5), torch.Tensor(vid7)


single_prediction = False
use_regression_token = False
use_pos_encoding = True
tr_activation_fct = 'gelu'
# Get all transformer models, _s stands for small, _b for big models
models = get_transformer_models(patch_size, embed_dim, num_heads, hidden_dim, num_layers, dropout, use_pos_encoding=use_pos_encoding,tr_activation_fct=tr_activation_fct,name_suffix='_s', use_regression_token= use_regression_token, single_prediction=single_prediction)

models_very_small = get_transformer_models(patch_size, embed_dim//2, num_heads//2, hidden_dim//2, num_layers//2, dropout, use_pos_encoding=use_pos_encoding,tr_activation_fct=tr_activation_fct,name_suffix='_vs', use_regression_token= use_regression_token, single_prediction=single_prediction)
models.update(models_very_small)
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
resnet = MultiImageLightResNet(patch_size, single_prediction=single_prediction)



models.update({"resnet": resnet})

# Create 1 optimizer and scheuler for each model
optimizers = {name: optim.AdamW(model.parameters(), lr=1e-4) for name, model in models.items()}
schedulers = {name: optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.9) for name, opt in optimizers.items()}
print(models.keys())
loss_function = nn.MSELoss()



# Image generation parameters
nPosPerFrame = 10 
background_mean, background_sigma = 200,10
part_mean, part_sigma = 500,20
image_props={"upsampling_factor":5,
      "background_intensity": [background_mean,background_sigma],
      "particle_intensity": [part_mean,part_sigma],
      "resolution": 130e-9,
      "trajectory_unit" : 1000,
      "output_size": 7,
      "poisson_noise" : 1}




verbose = False

num_cycles = 100  # Number of dataset refreshes
batch_size = 16

mix_trajectories = True and not single_prediction

# number of time steps per trajectory (frames), will be divided by nPosPerFrame
nFrames = 30
T = nFrames * nPosPerFrame
# number of trajectories
N = 64

# Mean and variance of the trajectories of Ds
TrainingDs_list = [[1, 1], [3, 1], [5, 1], [7, 1]]


# Load validation dataset (fixed, does not change across cycles)
val_videos = load_validation_data()  # Returns (vid1, vid3, vid5, vid7)
val_labels = torch.tensor([1, 3, 5, 7], dtype=torch.float32)  # Corresponding labels

# Dictionary to store validation losses per model and dataset type
validation_losses = {name: {f"val_{label.item()}": [] for label in val_labels} for name in models.keys()}
for models_name in validation_losses.keys():
    validation_losses[models_name].update({"val_avg":[]})

# create all_labels to be saved later
all_gen_labels = np.array([])  # Empty array to start


for cycle in range(num_cycles):

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
        videos = trajectories_to_video(trajs, nPosPerFrame, center=True, image_props=image_props)
        videos, _ = normalize_images(videos, background_mean, background_sigma, part_mean + background_mean)

        all_videos.append(videos)

    # Concatenate all generated data
    all_videos = np.concatenate(all_videos, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)


    if mix_trajectories:
        num_sequences_per_label = all_videos.shape[0] // len(val_labels)  # Number of sequences per label
        quarter_sequences = num_sequences_per_label // 4  # N/4 sequences per label

        # Define the mixing pairs (modularly specify which labels to mix)
        start_index = 0

        mixing_pairs = [
            (1, 7, start_index),  # Mix label 1 with label 7
            (1, 5, start_index + quarter_sequences),  # Mix label 1 with label 5
            (3, 7, start_index + quarter_sequences),  # Mix label 3 with label 7
            (3, 5, start_index)   # Mix label 3 with label 5
        ]

        for label_a, label_b, start_idx in mixing_pairs:

            label_a_start_idx = torch.where(val_labels == label_a)[0].item() + start_idx
            label_b_start_idx = torch.where(val_labels == label_b)[0].item()* N + start_idx

            for i in range(quarter_sequences):
                # Randomly select the split index
                split_index = np.random.randint(nFrames // 2 - 5, nFrames // 2 + 5)
                idx_a = label_a_start_idx + i
                idx_b = label_b_start_idx + i

                # Mix the videos and labels using a temporary buffer
                temp_videos = all_videos[idx_a, split_index:].copy()
                all_videos[idx_a, split_index:] = all_videos[idx_b, split_index:]
                all_videos[idx_b, split_index:] = temp_videos

                temp_labels = all_labels[idx_a, split_index:].copy()
                all_labels[idx_a, split_index:] = all_labels[idx_b, split_index:]
                all_labels[idx_b, split_index:] = temp_labels

                if(verbose):
                    print(f"Mixing {idx_a} (label {label_a}) with {idx_b} (label {label_b}) at split index {split_index}")

    """
    MIXES TRAJECTORIES TOO MUCH

    if mix_trajectories:
        num_to_mix = int(0.3 * all_videos.shape[0])  # 30% of the data
        indices = np.arange(all_videos.shape[0])
        np.random.shuffle(indices)
        mix_indices = indices[:num_to_mix]

        for idx in mix_indices:
            # Randomly select another trajectory to mix with
            other_idx = np.random.choice(indices)
            while other_idx == idx:
                other_idx = np.random.choice(indices)

            # Randomly select the split index
            split_index = np.random.randint(nFrames//2 - 5, nFrames//2 + 5)

            # Mix the videos and labels using a temporary buffer
            temp_videos = all_videos[idx, split_index:].copy()
            all_videos[idx, split_index:] = all_videos[other_idx, split_index:]
            all_videos[other_idx, split_index:] = temp_videos

            temp_labels = all_labels[idx, split_index:].copy()
            all_labels[idx, split_index:] = all_labels[other_idx, split_index:]
            all_labels[other_idx, split_index:] = temp_labels
            print(f"Mixing {idx} with {other_idx} at split index {split_index}")
            print(f"Labels: {all_labels[idx,:]} <-> {all_labels[other_idx,:]}")
    """


    # Convert to tensors
    all_videos = torch.Tensor(all_videos)
    all_labels = torch.Tensor(all_labels).unsqueeze(-1)  # Add an extra dimension for single_prediction

    # Create a dataset and shuffle
    dataset = ImageDataset(all_videos, all_labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for name, model in models.items():
        model.train()
        optimizer = optimizers[name]
        scheduler = schedulers[name]

        for batch_images, batch_labels in dataloader:
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


    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            label_losses = []
            for vid, label_value in zip(val_videos, val_labels):
                # Adjust label shape based on single_prediction
                if not model.single_prediction:
                    batch_size, num_images, _, _ = vid.shape
                    label = torch.full((batch_size, num_images, 1), label_value)  # Shape: [batch_size, num_images, 1]
                else:
                    label = torch.full((vid.shape[0],), label_value).view(-1, 1)  # Shape: [batch_size, 1]
                
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




print(all_gen_labels.shape)
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
