import os
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.abspath(os.path.join(current_path, '..'))
if parent_path not in sys.path:
    sys.path.insert(0, parent_path)
from utils import *
import torch
import torch.nn.functional as F
import random
import numpy as np
from models import *
from data import *
from visualization import *
from anchors import *
from relreps import *
from zero_shot import *

# For reproducibility and consistency across runs, we set a seed
set_random_seeds(43)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}: {torch.cuda.get_device_name(0)}")

# Load data
train_loader, test_loader, val_loader = load_mnist_data()
data = "MNIST"
loader = val_loader

### PARAMETERS ###
#NOTE: Conv_old gets best results with numbers, new with fashion
model = AE_conv #VariationalAutoencoder, AEClassifier, or Autoencoder
hidden_layer = (64, 128, 256, 512) # Use (128, 256, 512) for 100 dim, (64, 128, 256, 512) for 20 & 50 dim

# Hyperparameters for anchor selection
coverage_w = 0.90 # Coverage of embeddings
diversity_w = 1 - coverage_w # Pairwise between anchors

def zero_shot_different_seeds(number_of_seeds, model, dim, anchor_num, hidden_layer, coverage_w, diversity_w, device, train_loader, test_loader):

    mse_p_list, mse_rand_list = [] ,[]
    for i in tqdm(range(number_of_seeds)):
        set_random_seeds(i)
        # Run experiment. Return the Train embeddings
        model_list, emb_list_train, idx_list_train, labels_list_train, train_loss_list_AE, test_loss_list_AE, acc_list = train_AE(
            model=model,
            num_epochs=10,
            batch_size=256,
            lr=1e-3,
            device=device,      
            latent_dim=dim,
            hidden_layer=hidden_layer,
            nr_runs=2,
            save=False,
            verbose=False,
            train_loader=train_loader,
            test_loader=test_loader,
            input_dim=28*28,
            data=data
        )

        # Getting Tets and Validation embeddings (sorted by index)
        emb_list, idx_list, labels_list = get_embeddings(loader, model_list, device=device, verbose=False)

        # Creates a smaller dataset from the test embeddings with balanced class counts. It is sorted by index, so each run corresponds to each other
        small_dataset_emb, small_dataset_idx, small_dataset_labels = create_smaller_dataset(
            emb_list,
            idx_list,
            labels_list,
            samples_per_class=400
        )


        ####################################################
        # P ANCHORS
        ####################################################

        _, P_anchors_list = get_optimized_anchors(
            emb = small_dataset_emb,
            anchor_num=anchor_num,
            epochs=200,
            lr=1e-1,
            coverage_weight=coverage_w,
            diversity_weight=diversity_w,
            exponent=1,
            verbose=False,
            device=device,
        )
        anch_list = P_anchors_list

        # Compute relative coordinates for the embeddings
        relrep_list_p = compute_relative_coordinates_cossim(emb_list, anch_list)


        rel_decoder_p, rel_train_loss, rel_test_loss = train_rel_decoder(
            epochs=15,
            hidden_dims=hidden_layer,
            rel_model=rel_AE_conv_MNIST,
            model_list=model_list,
            relrep_list=relrep_list_p,
            idx_list=idx_list,
            loader=loader,
            nr_runs=None,
            device=device,
            show=False,
            verbose=False
        )


        ####################################################
        # RANDOM ANCHORS
        ####################################################


        # Find anchors and compute relative coordinates
        random_anchor_ids = random.sample(list(idx_list[0]), anchor_num)
        rand_anchors_list = select_anchors_by_id(model_list, emb_list, idx_list, random_anchor_ids, loader.dataset, show=False, device=device)
        

        anch_list = rand_anchors_list

        # Compute relative coordinates for the embeddings
        relrep_list_rand = compute_relative_coordinates_cossim(emb_list, anch_list)


        rel_decoder_rand, rel_train_loss, rel_test_loss = train_rel_decoder(
            epochs=15,
            hidden_dims=hidden_layer,
            rel_model=rel_AE_conv_MNIST,
            model_list=model_list,
            relrep_list=relrep_list_rand,
            idx_list=idx_list,
            loader=loader,
            nr_runs=None,
            device=device,
            show=False,
            verbose=False
        )

        to_tensor = transforms.ToTensor()
        target_images = []
        for img, _ in loader.dataset:
            # In case img is not already a tensor, convert it
            if not isinstance(img, torch.Tensor):
                img = to_tensor(img)
            img_flat = img.view(-1)
            target_images.append(img_flat)
        


        decoded_images_p = []
        decoded_images_rand = []


        with torch.no_grad():
            for relrep_p, relrep_rand in zip(relrep_list_p[1], relrep_list_rand[1]):

                if not isinstance(relrep_p, torch.Tensor):
                    relrep_p = torch.as_tensor(relrep_p)
                if not isinstance(relrep_rand, torch.Tensor):
                    relrep_rand = torch.as_tensor(relrep_rand)

                relrep_p = relrep_p.to(device)
                relrep_rand = relrep_rand.to(device)

                decoded_images_p.append(rel_decoder_p.decode(relrep_p).view(-1).to('cpu'))
                decoded_images_rand.append(rel_decoder_p.decode(relrep_rand).view(-1).to('cpu'))
        
        mse_p = F.mse_loss(torch.as_tensor(np.array(decoded_images_p)), torch.as_tensor(np.array(target_images)), reduction='mean')
        mse_rand = F.mse_loss(torch.as_tensor(np.array(decoded_images_rand)), torch.as_tensor(np.array(target_images)), reduction='mean')
        mse_p_list.append(mse_p), mse_rand_list.append(mse_rand)

        ####################################################
        # DISCREET GREEDY ANCHORS
        ####################################################
        """
        greedy_anchor_ids = greedy_one_at_a_time_single_euclidean(abs, idx_list_train,
                                                                num_anchors=anchor_num, repetitions=3,
                                                                diversity_weight=diversity_w, Coverage_weight=coverage_w, verbose=False)
        greedy_anchors_list = select_anchors_by_id(model_list, emb_list, idx_list, greedy_anchor_ids, loader.dataset, show=False, device=device)

        anch_list = greedy_anchors_list

        # Compute relative coordinates for the embeddings
        relrep_list = compute_relative_coordinates_cossim(emb_list, anch_list)


        rel_decoder, rel_train_loss, rel_test_loss = train_rel_decoder(
            epochs=15,
            hidden_dims=hidden_layer,
            rel_model=rel_AE_conv_MNIST,
            model_list=model_list,
            relrep_list=relrep_list,
            idx_list=idx_list,
            loader=loader,
            nr_runs=None,
            device=device,
            show=False,
            verbose=False
        )
        greedy_rel_test_loss_list.append(rel_test_loss), greedy_rel_train_loss_list.append(rel_train_loss)
        """


    return np.array(mse_p_list), np.array(mse_rand_list)



dims = [784, 300, 100, 50, 10, 2]

mse_p_list, mse_std_p_list = [], []
mse_rand_list, mse_std_rand_list = [], []
for dim in dims:
    print(f'dim={dim}')
    mse_p, mse_rand = zero_shot_different_seeds(10, model, dim, dim, hidden_layer, coverage_w, diversity_w, device, train_loader, test_loader)
    mse_p_list.append(np.mean(mse_p)), mse_rand_list.append(np.mean(mse_rand))
    mse_std_p_list.append(np.std(mse_p)), mse_std_rand_list.append(np.std(mse_rand))
# Create equal spacing positions for each dimension
x_positions = range(len(dims))

plt.figure(figsize=(8, 6))
# Plot the lines and capture the line objects to extract their colors
line_p, = plt.plot(x_positions, mse_p_list, marker='o', label='P', color='blue')
line_rand, = plt.plot(x_positions, mse_rand_list, marker='o', label='rand', color='orange')

# Use fill_between with the same colors as the lines
plt.fill_between(x_positions,
                 np.array(mse_p_list) - np.array(mse_std_p_list),
                 np.array(mse_p_list) + np.array(mse_std_p_list),
                 color=line_p.get_color(), alpha=0.2)
plt.fill_between(x_positions,
                 np.array(mse_rand_list) - np.array(mse_std_rand_list),
                 np.array(mse_rand_list) + np.array(mse_std_rand_list),
                 color=line_rand.get_color(), alpha=0.2)

plt.xlabel('Dimensions')
plt.ylabel('Mean Test Score')
plt.title('Test Mean Scores vs. Dimensions')
plt.xticks(x_positions, dims)
plt.legend()
plt.grid(True)
plt.show()