import torch
from MFact import MFact
import torch
from torch import optim, nn

from helpers import *

def train(model, optimizer, criterion, train_loader, valid_loader, nb_epochs, device, verbose="simple", estimate=False):
    """Train our matrix factorization mode

    Args:
        model (nn.Module): MFact model to train
        optimizer (nn.optim): Optimizer to use, SGD w.o. momentum in general
        criterion (nn.Criterion): Performance loss
        train_loader (DataLoader): Training data loader
        valid_loader (DataLoader): Validation data loader
        nb_epochs (Int): Number of epochs to train for
        device (String): Device to use
        verbose (str, optional): How many details wanted on ellapsed training. "full" is more complete; "simple" is more concise. Defaults to "simple".
        estimate (bool, optional): Whether we should estimate gradients norm and distance to optimumß. Defaults to False.

    Returns:
        (List, List, torch.Tensor, List, torch.Tensor, torch.Tensor): 
        (Train loss over epochs, Validation loss over epochs, maximum gradient, mean gradient (1d tensor), distance between local minimum and origin (1d tensor))
    """
    train_loss = []
    valid_loss = []
    # Estimate gradient norm and distance to local minima
    if estimate:
        mean_gradients = []
        max_grad = -1
        # Copy original weights to compute distance to local minima
        init_w = model.W.weight.detach().clone()
        init_z = model.Z.weight.detach().clone()
    model.to(device)
    if verbose == "simple":
        print()
    for epoch in range(nb_epochs):
        model.train()
        curr_train_loss = 0
        epoch_grads = 0
        for usr_id, mv_id, ratings in train_loader:
            #
            usr_id = usr_id.to(device)
            mv_id = mv_id.to(device)
            ratings = ratings.to(device)
            optimizer.zero_grad()

            pred = model(usr_id, mv_id)
            error = criterion(pred, ratings)
            curr_train_loss += error
            error.backward()
            # If estimate, compute the gradient norm and check if it is higher than previous max
            if estimate:
                z_norm = model.Z.weight.grad.norm()
                w_norm = model.W.weight.grad.norm()
                total_norm = z_norm + w_norm
                epoch_grads += total_norm
                if total_norm > max_grad:
                    max_grad = total_norm
            optimizer.step()
        # Compute loss on training set
        train_loss.append(curr_train_loss.detach().cpu().numpy() / len(train_loader))
        # Compute loss on validation set
        valid_loss.append(test_error(model, valid_loader, criterion, device))
        if estimate:
            mean_gradients.append(epoch_grads / len(train_loader))
        if verbose == "full":
            print(f"Epoch {epoch + 1}/{nb_epochs}. Train loss: {train_loss[-1]}, Validation loss: {valid_loss[-1]}")
        elif verbose == "simple":
            print(f"\r{(epoch + 1) / nb_epochs * 100 :.2f}%", end="")
        
    if verbose == "simple":
        print() # Prints a carriage return as we removed them earlier with end=""
        
    if estimate:
        dist_w = (model.W.weight.detach() - init_w).norm()
        dist_z = (model.Z.weight.detach() - init_z).norm()
        return train_loss, valid_loss, max_grad, mean_gradients, dist_w + dist_z
    else:
        return train_loss, valid_loss, None, None, None
# ============================================================================================================

def init_experiment(lrs, 
                    ranks, 
                    initializers, 
                    nb_users, 
                    nb_movies, 
                    nb_epochs, 
                    criterion, 
                    train_loader, 
                    valid_loader, 
                    device, 
                    verbose=True,
                    optim_algo="adam"):
    """Train models for different combinations of the hyperparameters

    Args:
        lrs (List): List of learning rates to test
        ranks (List): List of ranks to test
        initializers (List): List of lambdas that initializes the created model
        nb_users (Int): Number of users in data
        nb_movies (Int): Number of movies in data
        nb_epochs (Int): Number of epochs to train
        criterion (nn.Criterion): Performance criterion
        train_loader (DataLoader): Training data loader
        valid_loader (DataLoader): Validation data loader
        device (String): Device to send data to
        verbose (bool, optional): Whether to print details of training. Defaults to True.
        optim_algo (str, optional): "sgd" or "adam". The learning rate is picked from lrs. Defaults to "adam".

    Raises:
        ValueError: if optim_algo is something else than "sgd" or "adam"

    Returns:
        torch.Tensor: Tensor with dim (#lrs, #ranks, #initializers, #epochs) 
    """
    results = torch.empty(len(lrs), len(ranks), len(initializers), nb_epochs)
    for init_id, init_fct in enumerate(initializers):
        for rank_id, rank in enumerate(ranks):
            for lr_id, lr in enumerate(lrs):
                if verbose:
                    print(f"Starting initializer: {init_id}, rank: {rank}, lr: {lr :.4f}")
                model = MFact(nb_users, nb_movies, rank)
                init_fct(model)
                if optim_algo == "adam":
                    optimizer = optim.Adam(model.parameters(), lr=lr, momentum=0)
                elif optim_algo == "sgd":
                    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0)
                else:
                    raise ValueError(f"Unsupported optimization algorithm {optim_algo}")
                
                _, valid_loss, _, _, _ = train( model, 
                                                optimizer, 
                                                criterion, 
                                                train_loader, 
                                                valid_loader, 
                                                nb_epochs, 
                                                device, 
                                                verbose="simple" if verbose == True else None)
                
                results[lr_id, rank_id, init_id, :] = torch.tensor(valid_loss)
                if verbose:
                    print("-----------")
    return results
# ============================================================================================================

def grads_experiment(lr, 
                    rank, 
                    initializer,
                    seeds,
                    nb_users, 
                    nb_movies, 
                    nb_epochs, 
                    criterion, 
                    train_loader, 
                    valid_loader, 
                    device, 
                    verbose=True):
    """Estimate gradients and distance to optimum multiple times

    Args:
        lr (float): Learning rate of SGD optimizer
        rank (Int): Rank of the model
        initializer (Callable): Function that initializes the matrix factorization model
        seeds (List): List of seeds for each training, used for reproducibility
        nb_users (Int): Number of users in data
        nb_movies (Int): Number of movies in data
        nb_epochs (Int): Number of epochs in data
        criterion (nn.Criterion): Performance measure
        train_loader (DataLoader): Loader of training data
        valid_loader (DataLoader): Validation data loader
        device (String): Device to use
        verbose (bool, optional): Whether to print details on training. Defaults to True.

    Returns:
        (torch.Tensor, List, List): 
        (All gradients norm at each epoch of each trainining, Distance from starting point at each training, Maximal gradient at each trainingß)
    """
    result = torch.empty((len(seeds), nb_epochs))
    Rs = [] # Distance from starting point to local min
    max_grads = []
    for run_i, s in enumerate(seeds):
        if verbose:
            print(f"Starting iter {run_i + 1}/{len(seeds)}")
        model = MFact(nb_users, nb_movies, rank)
        initializer(model, s)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0)
        _, _, max_grad, epoch_gradients, R = train( model,  
                                                    optimizer, 
                                                    criterion, 
                                                    train_loader, 
                                                    valid_loader, 
                                                    nb_epochs, 
                                                    device, 
                                                    estimate=True,
                                                    verbose="simple" if verbose == True else None)
        result[run_i, :] = torch.tensor(epoch_gradients)
        Rs.append(R)
        max_grads.append(max_grad)
        if verbose:
            print("---------------")

    return result, Rs, max_grads
# ============================================================================================================

def test_lipschitz_lr(model, Rs, max_grads, train_loader, valid_loader, device, nb_epochs=100):
    """Compute lipschitz-based learning rate and train with it

    Args:
        model (nn.Module): Model to train
        Rs (List): Distance to optimum
        max_grads (List): Max gradients over trainings
        train_loader (DataLoader): Loader of training data
        valid_loader (DataLoader): Loader of validation data
        device (String): Device to use
        nb_epochs (int, optional): Number of training epochs. Defaults to 100.

    Returns:
        (List, List): (Training loss at each epoch, Validation loss at each epochß)
    """
    R = np.median(Rs)
    B = np.median(max_grads)
    T = nb_epochs * len(train_loader)
    lr = R / (B * T ** 0.5)
    print(f"lr={lr}")
    optimizer = optim.SGD(  model.parameters(), 
                            lr = lr, 
                            momentum=0) # Lipschitz convex learning rate
    criterion = nn.MSELoss()
    train_loss, valid_loss, _, _, _ = train(model, 
                                            optimizer, 
                                            criterion, 
                                            train_loader, 
                                            valid_loader, 
                                            nb_epochs,
                                            device, 
                                            verbose="simple", 
                                            estimate=False)
    return train_loss, valid_loss
