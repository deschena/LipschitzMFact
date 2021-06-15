import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

MOVIES_GENRES = [
    "unknown",
    "action", 
    "adventure", 
    "animation", 
    "children", 
    "comedy", 
    "crime", 
    "documentary", 
    "drama", 
    "fantasy", 
    "film noir", 
    "horror", 
    "musical", 
    "mystery", 
    "romance", 
    "sci-fi", 
    "thriller", 
    "war", 
    "western"]



RATING_FIELDS = [
    'user_id', 
    'movie_id', 
    'rating'
]
# =================================================================================================

class MLDataset(Dataset):
    def __init__(self, ratings):
        super(MLDataset, self).__init__()
        self.users = torch.tensor(ratings['user_id'].values, dtype=torch.long)
        self.movies = torch.tensor(ratings["movie_id"].values, dtype=torch.long)
        self.ratings = torch.tensor(ratings["rating"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, index):
        return self.users[index], self.movies[index], self.ratings[index]
# =================================================================================================
    
def load_ml100k(seed=42, training=0.8, test=0.2, valid=0.0):
    """Load ml100k data and split

    Args:
        seed (int, optional): Random seed to shuffle data. Defaults to 42.
        training (float, optional): Ratio of training samples. Defaults to 0.8.
        test (float, optional): Ratio of test samples. Defaults to 0.2.
        valid (float, optional): Ratio of validation samples. Defaults to 0.0.

    Returns:
        5-tuple of DataFrames: (train data, test data, valid data, movies features, users features)
    """
    assert np.allclose(training + test + valid, 1.0), f"Received: {training + test + valid}"

    # 943 users, 1682 items, 100_000 ratings
    # ratings
    cols = list(range(0, 3)) # Drop last col, represents rating date
    ratings = pd.read_csv('data/ml-100k/u.data', delimiter='\t', names=RATING_FIELDS, engine='python', usecols=cols)

    # Movies data
    cols = list(range(5, 5 + len(MOVIES_GENRES))) # Columns to keep
    fields = MOVIES_GENRES
    movies = pd.read_csv('data/ml-100k/u.item', delimiter='|', names=fields, engine="python", usecols=cols)
    
    # User data: available fields in data user id | age | gender | occupation | zip code
    # We only keep the id since we don't care about the demographics here, the project is about optim. algorithms
    users = pd.read_csv('data/ml-100k/u.user', delimiter='|', names =["user_id"], engine="python", usecols=[0])

    # Change index to start at 0 instead of 1
    ratings.loc[:, "user_id"] -= 1
    ratings.loc[:, "movie_id"] -= 1
    ratings = ratings.sample(frac=1, replace=False, random_state=seed) # Effectively shuffle dataset
    train_stop = int(ratings.shape[0] * training)
    test_stop = int(ratings.shape[0] * test + train_stop)

    x_train = ratings.iloc[0: train_stop, :]
    x_test = ratings.iloc[train_stop: test_stop, :]
    x_valid = ratings.iloc[test_stop:, :]
    
    return x_train, x_test, x_valid, movies, users
# =================================================================================================

def global_mean(dataframe):
    """Compute mean of a dataframe

    Args:
        dataframe (pd.Dataframe): Pandas dataframe

    Returns:
        Number: mean
    """
    return dataframe["rating"].mean()
# =================================================================================================

def user_mean(dataframe, nb_users):
    """Return per-user mean dataframe. Fills the missing users entries with the global mean

    Args:
        dataframe (pd.Dataframe): Pandas dataframe
        nb_users (int): Total number of users, might not be the same as the number in dataframe due to train/test split.

    Returns:
        pd.Dataframe: Dataframe of per-user mean
    """
    default_value = global_mean(dataframe)
    grouped = dataframe.iloc[:, [0, 2]].groupby(by="user_id").mean()
    grouped = grouped.reindex(index = pd.RangeIndex(0, nb_users), fill_value=default_value)
    grouped = grouped.reset_index().rename(columns={"index": "user_id"})
    return grouped
# =================================================================================================

def movie_mean(dataframe, nb_movies):
    """Return per-movie mean dataframe. Fills

    Args:
        dataframe (pd.Dataframe): Pandas dataframe
        nb_movies (int): Total number of movies, same idea as for user_mean

    Returns:
        pd.Dataframe: Dataframe of per-movie mean
    """
    default_value = global_mean(dataframe)
    grouped = dataframe.iloc[:, [1, 2]].groupby(by="movie_id").mean()
    grouped = grouped.reindex(index = pd.RangeIndex(0, nb_movies), fill_value=default_value)
    grouped = grouped.reset_index().rename(columns={"index": "movie_id"})
    return grouped
# =================================================================================================

def test_error(model, test_loader, criterion, device):
    """Compute the loss on the test set

    Args:
        model (nn.Module): Model under assessment
        test_loader (DataLoader): Loader for test data
        criterion (nn.Loss): Performance criterion
        device (String): Device to use for tensor

    Returns:
        Number: Performance over the test set
    """
    model.eval()
    total_error = 0
    
    for usr_id, mv_id, ratings in test_loader:
        usr_id = usr_id.to(device)
        mv_id = mv_id.to(device)
        ratings = ratings.to(device)
        
        pred = model(usr_id, mv_id)
        total_error += criterion(pred, ratings)
    return total_error.detach().cpu().numpy() / len(test_loader)
# =================================================================================================

def add_missing_entries(x_train, nb_users, nb_movies, default_value=0):
    """Adds entries to x_train so that we can then build a data tensor with all users of the database. 
       Required because the train/test split might make some user appear in one of them only.

    Args:
        x_train (pd.Dataframe): Pandas dataframe
        nb_users (Int): Number of user in original data
        nb_movies (Int): Number of movies in original data
        default_value (int, optional): Default value to set in ratings of missing elements. Defaults to 0.

    Returns:
        pd.Dataframe: Modified dataframe
    """
    distinct_users = x_train["user_id"].unique()
    distinct_movies = x_train["movie_id"].unique()
    # Ranges used to compute which users & movies are missing due to split
    users = pd.RangeIndex(nb_users)
    movies = pd.RangeIndex(nb_movies)

    # Add a default rating for each user missing due to split in original data
    # Useful for creating the data matrix X later
    missing_users = users.drop(distinct_users)
    new_user_data = {   "user_id": missing_users, 
                        "movie_id": np.zeros(len(missing_users)), 
                        "rating": np.ones(len(missing_users)) * default_value
                    }

    x_train = x_train.append(pd.DataFrame(columns=["user_id", "movie_id", "rating"], data=new_user_data))
    # Add a default rating for each movie missing due to split in original data
    # Useful for creating the data matrix X later
    missing_movies = movies.drop(distinct_movies)
    new_movie_data = {  "user_id": np.zeros(len(missing_movies)),
                        "movie_id": missing_movies,
                        "rating": np.ones(len(missing_movies)) * default_value
                    }
    return x_train.append(pd.DataFrame(columns=["user_id", "movie_id", "rating"], data=new_movie_data))
# =================================================================================================

def least_squares(W, X, lambda_):
    """Computes the regularized least squares solution for initializing the Z matrix

    Args:
        W (torch.tensor): Movies embedding matrix
        X (torch.tensor): Data tensor
        lambda_ (Number): Regularizer

    Returns:
        torch.tensor: Least squares solution
    """
    W_t = W.t()
    rank = W.shape[1]
    Z_t = (W_t @ W + lambda_ * torch.eye(rank)).inverse() @ W_t @ X
    return Z_t.t()
# =================================================================================================

def data_tensor(x_train, nb_users, nb_movies, default_value=0):
    """Create a user/movie ratings tensor

    Args:
        x_train (pd.Dataframe): pandas dataframe of user/movie pairs representing ratings
        nb_users (Int): Number of users in original database
        nb_movies (Int): Number of movies in original database
        default_value (int, optional): Default value of unknown entriesß. Defaults to 0.

    Returns:
        torch.tensor: Data tensor
    """
    completed_dataframe = add_missing_entries(x_train, nb_users, nb_movies, default_value=default_value)
    X = torch.tensor(
        completed_dataframe
        .pivot(index="movie_id", columns="user_id", values="rating")
        .fillna(0)
        .to_numpy()
        , dtype=torch.float32
    )
    return X
# =================================================================================================

def get_best_lrs(perf_tensor, lrs):
    """Extract the best learning rate and its position from tensor of experiments (see grid-search exp in run.py)

    Args:
        perf_tensor (torch.tensor): Tensor of shape (#lr, #ranks tested, #initializers, #epochs) containing the result of the lr experiment
        lrs (list): List of the learning rates used

    Returns:
        (Int, Number): (Index of best lr, best lr)
    """
    # perf_tensor is of shape (#lr, #ranks tested, #initializers, #epochs)
    # compute index of best learning rate for each inits
    best_perf = torch.min(perf_tensor, dim=3).values.min(dim=0)
    best_lr_index = best_perf.indices.flatten()
    best_lrs = lrs[best_lr_index]
    return best_lr_index, best_lrs
# =================================================================================================