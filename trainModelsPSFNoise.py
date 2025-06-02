from models import *
from andi_datasets.models_phenom import models_phenom
from helpersGeneration import *
from trainSettingsPSFNoise import *
import datetime


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def save_results(validation_losses, all_gen_labels, models,path_addition=""):
    save_path = "training_results_PSFNoise"+path_addition+".pth"
    results = {
        "validation_losses": validation_losses,
        "all_labels": all_gen_labels,  # Convert to NumPy for easier histogram plotting
        "model_weights": {name: model.state_dict() for name, model in models.items()}
    }
    torch.save(results, save_path)
    print(f"\nTraining results saved to {save_path}")



### Models Settings ###
# Most of models settings are in Train_settings for unity
# Type of training: mixTrajectories for learning trajectories with a switch in the middle
mix_trajectories = True and not single_prediction
models, optimizers, schedulers = getTrainingModels()

for name in models:
    print(name)
    models[name] = models[name].to(device)

### Training Settings ###
num_cycles = 100  # Number of dataset refreshes
# ToDo: Try if reducing batch_size makes the model learn the transitions
# ToDO: Try computing loss per timeStep, or add a loss term that favorises transitions see https://chatgpt.com/c/67efd8f6-52a4-8010-a0ca-09ea0b60fa3e
batch_size = 1 if adaptive_batch_size != -1 else 16 # Number of sequences in 1 batch
shuffle = True # if trajectories should be shuffled during training
N = 64 # Number of sequences in per value of D in Trainings_Ds
# Mean and variance of the trajectories of Ds
#TrainingDs_list = [[1, 1], [3, 1], [5, 1], [7, 1]]
TrainingDs_list = [[1, 1], [3, 1], [5, 1], [7, 1], [9,1], [10.2,1]]

printParams = True
verbose = False



# Load validation dataset (fixed, does not change across cycles)
val_videos = load_validation_data(nFrames, skip_inorder=True)  # Returns (vid1, vid3, vid5, vid7)
val_labels = torch.tensor([1, 3, 5, 7, 9], dtype=torch.float32)  # Corresponding labels

# Divide Labels by D_max to have values between 0 and 1 -> better optimizers
val_labels = val_labels 


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
        trajs, labels = models_phenom().single_state(N if TrainingDs[0] != 10.2 else N//2,  
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
        all_labels = all_labels 


        # Convert trajectories of D (pixels/s) to D (micro_m/ms)
        trajs = trajs / traj_div_factor
        videos = trajs_to_vid_psf_noise(trajs,nPosPerFrame,center=center,image_props=image_props, PSF_Settings=PSF_Settings, Noise_Settings=Noise_Settings)


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

            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()

            predictions = make_prediction(model, name, batch_images, eval=False)

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
                
                val_images = vid

                # Adjust label shape based on single_prediction
                if not single_prediction:
                    batch_size, _, _, num_images, _, _ = val_images.shape
                    label = torch.full((batch_size, num_images, 1), label_value, device=device)  # Shape: [batch_size, num_images, 1]
                else:
                    label = torch.full((val_images.shape[0],), label_value, device=device).view(-1, 1)  # Shape: [batch_size, 1]
                

                vid = vid.to(device)
                label = label.to(device)
                val_predictions = make_prediction(model, name, vid, True)

                val_predictions = val_predictions * D_max_normalization

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

    
    if (num_cycles - cycle -1< 5):
        save_results(validation_losses, all_gen_labels, models, path_addition=str(num_cycles - cycle))




print(f"Number of generated sequences: {all_gen_labels.shape}")
# --- Save everything for later analysis ---


save_results(validation_losses, all_gen_labels, models)
print(datetime.datetime.now())
