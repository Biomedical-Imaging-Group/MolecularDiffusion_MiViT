from models import *
from andi_datasets.models_phenom import models_phenom
from helpersGeneration import *
from helpersPlot import *
from trainSettings import *
import datetime




### Models Settings ###
# Most of models settings are in Train_settings for unity
# Type of training: mixTrajectories for learning trajectories with a switch in the middle
mix_trajectories = True and not single_prediction
models, optimizers, schedulers = getTrainingModels()


### Training Settings ###
num_cycles = 100  # Number of dataset refreshes
batch_size = 16 # Number of sequences in 1 batch
N = 64 # Number of sequences in per value of D in Trainings_Ds
# Mean and variance of the trajectories of Ds
TrainingDs_list = [[1, 1], [3, 1], [5, 1], [7, 1]]
verbose = False



# Load validation dataset (fixed, does not change across cycles)
val_videos = load_validation_data(nFrames)  # Returns (vid1, vid3, vid5, vid7)
val_labels = torch.tensor([1, 3, 5, 7], dtype=torch.float32)  # Corresponding labels


# Dictionary to store validation losses per model and dataset type
validation_losses = {name: {f"val_{label.item()}": [] for label in val_labels} for name in models.keys()}
for models_name in validation_losses.keys():
    validation_losses[models_name].update({"val_avg":[]})
# create all_labels to be plotted later
all_gen_labels = np.array([])  # Empty array to start


print("Starting Training with parameters:")
print("### Model parameters ###")

print(datetime.datetime.now())


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


    # Mix Trajectories
    if mix_trajectories:
        num_sequences_per_label = all_videos.shape[0] // len(val_labels)  # Number of sequences per label
        quarter_sequences = num_sequences_per_label // 4  # N/4 sequences per label

        # Define the mixing pairs (modularly specify which labels to mix)
        start_index = 0

        # Labels to mix between each other
        # Currently for each label we have 2* 1/4 of sequecnes mixed and 1/2 normal 
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


    # Convert to tensors
    all_videos = torch.Tensor(all_videos)
    all_labels = torch.Tensor(all_labels).unsqueeze(-1)  # Add an extra dimension for single_prediction

    # Create a dataset and shuffle
    dataset = ImageDataset(all_videos, all_labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


    # Training Loop
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


    # Evaluation part
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
