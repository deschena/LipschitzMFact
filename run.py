from helpers import *
from MFact import *
from training_loops import *
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset

from matplotlib import pyplot as plt
# =================================================================================================
# Setup matterial for all experiments
train_ratio = 0.7
test_ratio = 0
valid_ratio = 0.3
x_train, x_test, x_valid, movies, users = load_ml100k(training=train_ratio, test=test_ratio, valid=valid_ratio)

batch_size = 512 if torch.cuda.is_available() else 256
num_workers = 8 if torch.cuda.is_available() else 0 # crash with different values on CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)
criterion = nn.MSELoss()
nb_users = len(users)
nb_movies = len(movies)

train_dataset = MLDataset(x_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, shuffle=True, num_workers=num_workers, drop_last=True)

test_dataset = MLDataset(x_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=True)

valid_dataset = MLDataset(x_valid)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=True)

lrs = torch.tensor([1, 0.1, 0.01, 0.001])
nb_epochs = 500
labels = ["iid $\mathcal{N}$(0, 1)", "Global mean", "User mean", "Movie mean", "Random Acol", "K-means", "SVD"]
# Assign index of each entry to a global variable, useful later
NORMAL, GLOBAL, USER, MOVIE, ACOL, KMEANS, SVD = tuple(range(len(labels)))
DATA = "exp_data/"
IMG_DIR = "images/"
# =================================================================================================
def compare_inits():
    print("Start comparing all initializations")
    lrs = torch.tensor([1, 0.1, 0.01, 0.001])
    ranks = [8]
    n_init = 8
    max_iter = 150
    initializers = (
        lambda m: 0, # Leave normal initialization
        lambda m: m.global_mean_init(x_train),
        lambda m: m.user_mean_init(x_train),
        lambda m: m.movie_mean_init(x_train),
        lambda m: m.random_acol_init(x_train, p=15, seed=34),
        lambda m: m.k_means_init(x_train, n_init=n_init, max_iter=max_iter, seed=9123),
        lambda m: m.svd_init(x_train),
    )

    all_perfs_tensor = init_experiment( lrs, 
                                        ranks, 
                                        initializers, 
                                        nb_users=nb_users, 
                                        nb_movies=nb_movies, 
                                        nb_epochs=nb_epochs, 
                                        criterion=criterion, 
                                        train_loader=train_loader, 
                                        valid_loader=valid_loader, 
                                        device=device, 
                                        optim_algo="sgd",
                                        verbose=True)

    # Save tensor for later reuse
    torch.save(all_perfs_tensor, DATA + "all_perf_tensor.pth")

    print("--------------------------------")

# =================================================================================================
def plot_init_comparison(display_perf=False):
    # Plot performance with best learning rate
    all_perfs_tensor = torch.load(DATA + "all_perf_tensor.pth")
    all_perfs_tensor[0, :, -1, :] = 100 # This slice is full of NaN; with a lr of 1, the learning fails completely with SVD
    best_lr_index, best_lrs = get_best_lrs(all_perfs_tensor, lrs)
    # Actual performance
    #torch.min(all_perfs_tensor, dim)
    print("nb lrs:", len(lrs))
    if display_perf:
        for i, l in enumerate(labels):
            print(l)
            for lr_id, lr in enumerate(lrs):
                perf = all_perfs_tensor[lr_id, 0, i, :].min()
                print(l, "lr =", f"{lr.item() :.3f}", ". Perf:", f"{perf.item() :.3f}")
            print("----------------------")

    x = range(nb_epochs)
    plt.title("Performance of all initializations")
    for init_index, (lr_index, l) in enumerate(zip(best_lr_index, labels)):
        plt.loglog(x, all_perfs_tensor[lr_index, 0, init_index, :], label=l)
    plt.legend(ncol=2)
    plt.savefig(IMG_DIR + "perf_all.png")
    plt.tight_layout()
    plt.clf()

    plt.title("Performance of lower regime")
    plt.subplots_adjust(left=0.157) # Found by trial and error
    # Casting to list allows to use slicing
    for init_index, (lr_index, l) in list(enumerate(zip(best_lr_index, labels)))[1:4]: 
        plt.loglog(x, all_perfs_tensor[lr_index, 0, init_index, :], label=l)
    plt.legend()
    plt.savefig(IMG_DIR + "perf_all_lower.png")
    
    plt.clf()
    print(list(enumerate(labels)))
    plt.title("Performance of medium regime")
    # Normal initialization is at the begining of the tensor
    plt.loglog(x, all_perfs_tensor[best_lr_index[0], 0, 0, :], label=labels[0])
    # Casting to list allows to use slicing
    for init_index, (lr_index, l) in list(enumerate(zip(best_lr_index, labels)))[4:]: 
        plt.loglog(x, all_perfs_tensor[lr_index, 0, init_index, :], label=l)
    plt.legend(ncol=2, handleheight=0.5)
    plt.savefig(IMG_DIR + "perf_all_medium.png")
    
    plt.clf()

# =================================================================================================

def estimate_grads_and_dist_to_opt():
    print("Start estimation of gradients norm and distance to opt")
    seeds = np.array([  963072, 570729, 927488, 730314, 616072, 853904, 681950, 704191,
                        13172, 503881, 849671, 943374, 923157, 280791, 481669, 523867,
                        939646, 491078, 225283, 507134,  95489, 752582, 320643, 988243,
                        263986, 946342, 769265, 753687,  10593, 270306]) # len 30; 30 runs

    seeds = np.array([  963072, 570729, 927488]) # len 30; 30 runs

    lr = 0.1


    _, acol_Rs, acol_max_grads = grads_experiment(  lr=lr,
                                                        rank=8,
                                                        initializer=lambda m, s: m.random_acol_init(x_train, seed=s),
                                                        seeds=seeds,
                                                        nb_users=nb_users,
                                                        nb_movies=nb_movies,
                                                        nb_epochs=nb_epochs,
                                                        criterion=nn.MSELoss(),
                                                        train_loader=train_loader,
                                                        valid_loader=valid_loader,
                                                        device=device,
                                                        verbose=True)

    np.save(DATA + "acol_Rs.npy", acol_Rs)
    np.save(DATA + "acol_max_grads.npy", acol_max_grads)

    # ---
    seeds = np.array([  922271, 524185, 781156, 196710, 390187, 382636, 303213,  63660,
                        426223, 542171, 605670,  45379, 673166, 678453, 525363, 760933,
                        64509, 426041, 271468, 920433, 315167, 545615, 143606, 412001,
                        673647,  29779, 593164, 640659, 297462, 896832]) # len 30; 30 runs

    seeds = np.array([  922271, 524185, 781156]) # len 30; 30 runs

    _, normal_Rs, normal_max_grads = grads_experiment(  lr=lr,
                                                        rank=8,
                                                        initializer=lambda m, s: 0, # Normal init
                                                        seeds=seeds,
                                                        nb_users=nb_users,
                                                        nb_movies=nb_movies,
                                                        nb_epochs=nb_epochs,
                                                        criterion=nn.MSELoss(),
                                                        train_loader=train_loader,
                                                        valid_loader=valid_loader,
                                                        device=device,
                                                        verbose=True)

    np.save(DATA + "normal_Rs.npy", normal_Rs)
    np.save(DATA + "normal_max_grads.npy", normal_max_grads)

    # ---
    seeds = np.zeros(3) # Seed do not matter for global mean initialization

    _, mean_Rs, mean_max_grads = grads_experiment(  lr=lr,
                                                    rank=8,
                                                    initializer=lambda m, s: m.global_mean_init(x_train),
                                                    seeds=seeds,
                                                    nb_users=nb_users,
                                                    nb_movies=nb_movies,
                                                    nb_epochs=nb_epochs,
                                                    criterion=nn.MSELoss(),
                                                    train_loader=train_loader,
                                                    valid_loader=valid_loader,
                                                    device=device,
                                                    verbose=True)
    np.save(DATA + "mean_Rs.npy", mean_Rs)
    np.save(DATA + "mean_max_grads.npy", mean_max_grads)

    print("--------------------------------")

# =================================================================================================

def compare_grid_and_lipschitz_plot_aux(x_axis, base, lipschitz, lr_base, lr_valid, title, filename, title_size=25):
    plt.loglog(x_axis, base, label=f"$\gamma = {lr_base}$")
    plt.loglog(x_axis, lipschitz, label=f"Lipschitz $\gamma = {lr_valid}$")
    plt.legend()
    plt.title(title, fontsize=title_size)
    plt.tight_layout()
    plt.savefig(IMG_DIR + f"{filename}.png")
    plt.clf()
# =================================================================================================

def compare_grid_and_lipschitz_plot():
    # Plot kmeans
    k_valid, kmeans_base = torch.load(DATA + "acol_compare.pth")
    x_axis = range(len(k_valid))
    compare_grid_and_lipschitz_plot_aux(x_axis, kmeans_base, k_valid, 1, 0.153, "Random Acol", "acol_lipschitz")

    # Plot normal
    normal_valid, normal_base = torch.load(DATA + "normal_compare.pth")
    compare_grid_and_lipschitz_plot_aux(x_axis, normal_base, normal_valid, 1, 0.134, "Normal", "normal_lipschitz")

    # Plot global mean
    global_valid, global_base = torch.load(DATA + "global_compare.pth")
    compare_grid_and_lipschitz_plot_aux(x_axis, global_base, global_valid, 0.1, 0.097, "Global mean", "global_lipschitz")
# =================================================================================================

def compare_grid_and_lipschitz():
    # Compare training with lr found from lipschitz formula vs fixed stepsize
    # Compare Kmeans init
    print("Starting experiment to compare grid search and 'lipschitz-search'")
    acol_Rs = np.load(DATA + "acol_Rs.npy")
    acol_max_grads = np.load(DATA + "acol_max_grads.npy")
    model = MFact(nb_users, nb_movies, rank=8)
    model.random_acol_init(x_train)
    _, k_valid = test_lipschitz_lr( model,
                                    acol_Rs, 
                                    acol_max_grads, 
                                    train_loader,
                                    valid_loader,
                                    device,
                                    nb_epochs=nb_epochs)

    # Train model with a fixed stepsize
    model = MFact(nb_users, nb_movies, rank=8)
    model.k_means_init(x_train)
    opt = optim.SGD(model.parameters(), lr=1, momentum=0)
    _, kmeans_base, _, _, _ = train(model,
                                    opt,
                                    nn.MSELoss(),
                                    train_loader,
                                    valid_loader,
                                    nb_epochs,
                                    device,
                                    verbose="simple",
                                    )

    torch.save([k_valid, kmeans_base], DATA + "acol_compare.pth")
    # ---
    # Compare normal init
    normal_Rs = np.load(DATA + "normal_Rs.npy")
    normal_max_grads = np.load(DATA + "normal_max_grads.npy")
    model = MFact(nb_users, nb_movies, rank=8)
    _, normal_valid = test_lipschitz_lr(model,
                                        normal_Rs, 
                                        normal_max_grads, 
                                        train_loader,
                                        valid_loader,
                                        device,
                                        nb_epochs=nb_epochs)

    # Train normal init with fixed stepsize
    model = MFact(nb_users, nb_movies, rank=8)
    opt = optim.SGD(model.parameters(), lr=1, momentum=0)
    _, normal_base, _, _, _ = train(model,
                                    opt,
                                    nn.MSELoss(),
                                    train_loader,
                                    valid_loader,
                                    nb_epochs,
                                    device,
                                    verbose="simple",
                                    )
    torch.save([normal_valid, normal_base], DATA + "normal_compare.pth")
    # ---
    # Compare mean init
    mean_Rs = np.load(DATA + "mean_Rs.npy")
    mean_max_grads = np.load(DATA + "mean_max_grads.npy")
    model = MFact(nb_users, nb_movies, rank=8)
    model.global_mean_init(x_train)
    _, global_valid = test_lipschitz_lr(model,
                                        mean_Rs, 
                                        mean_max_grads, 
                                        train_loader,
                                        valid_loader,
                                        device,
                                        nb_epochs=nb_epochs)

    # Train model with a fixed stepsize
    model = MFact(nb_users, nb_movies, rank=8)
    model.global_mean_init(x_train)
    opt = optim.SGD(model.parameters(), lr=0.1, momentum=0)
    _, global_mean_base, _, _, _ = train(model,
                                    opt,
                                    nn.MSELoss(),
                                    train_loader,
                                    valid_loader,
                                    nb_epochs,
                                    device,
                                    verbose="simple",
                                    )
    torch.save([global_valid, global_mean_base], DATA + "global_compare.pth")
    print("--------------------------------")
# =================================================================================================

def main():

    #compare_inits()
    #plot_init_comparison()

    #estimate_grads_and_dist_to_opt()
    #compare_grid_and_lipschitz()
    compare_grid_and_lipschitz_plot()

# =================================================================================================


if __name__ == "__main__":
    main()
# =================================================================================================