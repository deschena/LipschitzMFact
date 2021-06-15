import torch
from torch import nn
from helpers import *
from sklearn.cluster import KMeans

class MFact(nn.Module):
    # Note: we believe the code for inits is straightforward to understand with the report, hence we did not add a lot of comments, as it is also quite short
    def __init__(self, nb_users, nb_movies, rank, use_sparse=False):

        super(MFact, self).__init__()
        # Movies matrix
        self.rank = rank
        self.C = None # Rebalance constant
        self.nb_users = nb_users
        self.nb_movies = nb_movies
        self.W = nn.Embedding(num_embeddings=nb_movies, 
                            embedding_dim=rank, sparse=use_sparse)
        # Users matrix
        self.Z = nn.Embedding(num_embeddings=nb_users,
                            embedding_dim=rank, sparse=use_sparse)
                            
# ============================================================================================================
    def init_params(self, init_z, init_w):
        with torch.no_grad():
            self.W.weight[:, :] = init_w
            self.Z.weight[:, :] = init_z

# ============================================================================================================
    def global_mean_init(self, x_train):
        default_value = (global_mean(x_train) / self.rank) ** 0.5
        Z = torch.full(self.Z.weight.shape, default_value)
        W = torch.full(self.W.weight.shape, default_value)
        self.init_params(Z , W)

# ============================================================================================================
    def user_mean_init(self, x_train):
        init_value = user_mean(x_train, self.nb_users).to_numpy()[:, 1]
        Z = torch.tensor(init_value).unsqueeze(1).expand(-1, self.rank)
        W = torch.full(self.W.weight.shape, 1 / self.rank)
        self.init_params(Z, W)

# ============================================================================================================
    def movie_mean_init(self, x_train):
        init_value = movie_mean(x_train, self.nb_movies).to_numpy()[:, 1]
        Z = torch.full(self.Z.weight.shape, 1 / self.rank)
        W = torch.tensor(init_value).unsqueeze(1).expand(-1, self.rank)
        self.init_params(Z, W)

# ============================================================================================================
    def random_acol_init(self, x_train, p=100, lambda_=0.01, default_value=0, seed=42):
        np.random.seed(seed)
        distinct_users = x_train["user_id"].unique()
        W = torch.zeros(size=(self.nb_movies, self.rank))
        X = data_tensor(x_train, self.nb_users, self.nb_movies, default_value=default_value)

        for r in range(self.rank):
            selection = np.random.choice(distinct_users, size=p)
            W[:, r] = torch.mean(X[:, selection], dim=1)
        # Compute least squares solution to init Z
        Z = least_squares(W, X, lambda_)
        self.init_params(Z, W)
    
# ============================================================================================================
    def k_means_init(self, x_train, n_init=5, max_iter=50, lambda_=0.01, default_value=0, seed=42):

        X = data_tensor(x_train, self.nb_users, self.nb_movies, default_value).numpy()
        kmeans = KMeans(n_clusters=self.rank, random_state=seed, n_init=n_init, max_iter=max_iter).fit(X.T)
        centroids = kmeans.cluster_centers_.T
        X = torch.tensor(X)
        W = torch.tensor(centroids)
        Z = least_squares(W, X, lambda_)
        self.init_params(Z, W)
    
# ============================================================================================================
    def svd_init(self, x_train, default_value=0):
        X = data_tensor(x_train, self.nb_users, self.nb_movies, default_value)
        U, S, V_t = torch.linalg.svd(X)
        W = U[:, :self.rank] * S[:self.rank]
        Z = V_t[:self.rank, :].t()
        self.init_params(Z, W)

# ============================================================================================================
    def forward(self, users, movies):
        users_pred = self.Z(users)
        movies_pred = self.W(movies)
        # Mimic dot product, we only care about entries related to given ratings
        return torch.sum(users_pred * movies_pred, dim=1)