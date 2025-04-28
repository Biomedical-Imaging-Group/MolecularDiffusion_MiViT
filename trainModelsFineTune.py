from models import *
from andi_datasets.models_phenom import models_phenom
from helpersGeneration import *
from helpersPlot import *
from trainSettings import *
import datetime


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)




### Models Settings ###
# Most of models settings are in Train_settings for unity
# Type of training: mixTrajectories for learning trajectories with a switch in the middle
mix_trajectories = True and not single_prediction
models, optimizers, schedulers = getTrainingModels(lr=1e-5)

for name in models:
    models[name] = models[name].to(device)



models, optimizers, schedulers = getTrainingModels()

results = torch.load("training_results_single.pth",map_location=torch.device('cpu'))
model_weights = results["model_weights"]

# Load model weights
for name, model in models.items():
    if name in model_weights:
        model.load_state_dict(model_weights[name])
        print(f"Loaded weights for {name}")
    else:
        print(f"Warning: No saved weights found for {name}")


### Training Settings ###
num_cycles = 2  # Number of dataset refreshes
# ToDo: Try if reducing batch_size makes the model learn the transitions
# ToDO: Try computing loss per timeStep, or add a loss term that favorises transitions see https://chatgpt.com/c/67efd8f6-52a4-8010-a0ca-09ea0b60fa3e
shuffle = True # if trajectories should be shuffled during training
N = 10 # Number of sequences in per value of D in Trainings_Ds
# Mean and variance of the trajectories of Ds
printParams = True
verbose = False

tr_losses = {name:[] for name in models.keys()}


if(printParams):
    print("StartTime: ",datetime.datetime.now())
train_d_in_order = np.arange(0.1, 7.001, 0.1)


#batch_size = len(train_d_in_order) * N
batch_size = 10

for cycle in range(num_cycles):

    print(f"Cycle {cycle+1} out of {num_cycles}: {(cycle+1)/num_cycles * 100:.2f}%")
    # Generate a new batch of images and labels
    all_videos = []
    all_labels = []

    for D in train_d_in_order:
        trajs, labels = models_phenom().single_state(N, 
                                        L = 0,
                                        T = T,
                                        Ds = [D, 0], # Mean and variance
                                        alphas = 1)
        # Need to reshape generated trajectories because they are in format (T,N,dim), but we want them in (N,T,dim)
        trajs = trajs.transpose(1,0,2)
        # Convert trajectories of D (pixels/s) to D (micro_m/ms)
        trajs = trajs / traj_div_factor
        videos = trajectories_to_video(trajs, nPosPerFrame, center=True, image_props=image_props)
        videos, _ = normalize_images(videos, background_mean, background_sigma, part_mean + background_mean)

        all_videos.append(videos)
        all_labels.append(np.ones(N)*D)
    # Concatenate all generated data
    all_videos = np.concatenate(all_videos, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)


    # Convert to tensors
    all_videos = torch.Tensor(all_videos)
    all_labels = torch.Tensor(all_labels).unsqueeze(-1)  # Add an extra dimension for single_prediction

    # Create a dataset and shuffle
    dataset = ImageDataset(all_videos, all_labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle = mix_trajectories or shuffle)


    # Training Loop
    for name, model in models.items():
        model.train()
        optimizer = optimizers[name]
        scheduler = schedulers[name]

        for batch_images, batch_labels in dataloader:

            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            
            predictions = model(batch_images)

            loss = loss_function(predictions, batch_labels)
            loss.backward()
            optimizer.step()  # <-- This actually updates the model parameters
            tr_losses[name].append(loss.item())
            predictions = model(batch_images)
            loss = loss_function(predictions, batch_labels)
            tr_losses[name].append(loss.item())


        
        if(verbose):
            for p_name, param in model.named_parameters():
                if param.grad is not None:
                    print(f"{name}: {p_name}: Mean : {param.grad.abs().mean().item():.8f} Max:  {param.grad.abs().max().item():.8f}")
                    #print(model.reg_token)


print(f"Number of generated sequences: {train_d_in_order.shape}")
# --- Save everything for later analysis ---
save_path = "training_results_ft.pth"
results = {
    "validation_losses": {name:{"val_avg":tr_losses[name]} for name in models.keys()},
    "all_labels": train_d_in_order,  # Convert to NumPy for easier histogram plotting
    "model_weights": {name: model.state_dict() for name, model in models.items()}
}
torch.save(results, save_path)
print(f"\nTraining results saved to {save_path}")
print(datetime.datetime.now())
