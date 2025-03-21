from models import *
from andi_datasets.models_phenom import models_phenom
from helpersGeneration import *
from helpersPlot import *
from models import get_transformer_models


verbose = False

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


# Define model hyperparameters
patch_size = 7
embed_dim = 64
num_heads = 4
hidden_dim = 128
num_layers = 6
dropout = 0.0



# Get all transformer models, _s stands for small, _b for big models
models = get_transformer_models(patch_size, embed_dim, num_heads, hidden_dim, num_layers, dropout,name_suffix='_s')
models_big = get_transformer_models(patch_size, embed_dim*2, num_heads*2, hidden_dim*2, num_layers*2, dropout,name_suffix='_b')

models.update(models_big)
resnet = MultiImageLightResNet(patch_size)
models.update({"resnet": resnet})

# Create 1 optimizer and scheuler for each model
optimizers = {name: optim.Adam(model.parameters(), lr=1e-3) for name, model in models.items()}
schedulers = {name: optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.9) for name, opt in optimizers.items()}
print(models.keys())
loss_function = nn.MSELoss()



# Image generation parameters
nPosPerFrame = 10 
background_mean, background_sigma = 100,10
part_mean, part_sigma = 500,20
image_props={"upsampling_factor":10,
      "background_intensity": [background_mean,background_sigma],
      "particle_intensity": [part_mean,part_sigma],
      "resolution": 100e-9,
      "trajectory_unit" : 1000,
      "output_size": 7,
      "poisson_noise" : -1
        }





num_cycles = 2  # Number of dataset refreshes
batch_size = 32



# number of time steps per trajectory (frames), will be divided by nPosPerFrame
T = 200
# number of trajectories
N = 32

TrainingDs_list = [[1, 1], [3, 1], [5, 1], [7, 1]]


# Load validation dataset (fixed, does not change across cycles)
val_videos = load_validation_data()  # Returns (vid1, vid3, vid5, vid7)
val_labels = torch.tensor([1, 3, 5, 7], dtype=torch.float32)  # Corresponding labels

# Dictionary to store validation losses per model and dataset type
validation_losses = {name: {f"val_{label.item()}": [] for label in val_labels} for name in models.keys()}

# create all_labels to be saved later
all_gen_labels = np.array([])  # Empty array to start


for cycle in range(num_cycles):

    # Generate a new batch of 2000 images and labels
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
        labels = labels.transpose(1, 0, 2)[:, 0, 1]

        # Store all labels
        all_labels.append(labels)

        labels = torch.Tensor(labels).view(-1, 1)

        # Convert trajectories of D (pixels/s) to D (micro_m/ms)
        trajs = trajs / traj_div_factor
        videos = trajectories_to_video(trajs, nPosPerFrame, center=True, image_props=image_props)
        videos, _ = normalize_images(videos, background_mean, background_sigma, part_mean + background_mean)

        all_videos.append(videos)

    # Concatenate all generated data
    all_videos = np.concatenate(all_videos, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_gen_labels = np.append(all_gen_labels,all_labels)
    # Convert to tensors
    all_videos = torch.Tensor(all_videos)
    all_labels = torch.Tensor(all_labels).view(-1, 1)

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

        # --- Validation Phase ---
    for name, model in models.items():

        model.eval()
        with torch.no_grad():
            for vid, label in zip(val_videos, val_labels):
                val_predictions = model(vid)
                val_loss = loss_function(val_predictions, torch.full((vid.shape[0],), label).view(-1,1))
                avg_val_loss = val_loss.item()
                validation_losses[name][f"val_{label.item()}"].append(avg_val_loss)
                if(verbose):
                    print(f"{name} on val_{label.item()}: Validation Loss = {avg_val_loss:.4f}")



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