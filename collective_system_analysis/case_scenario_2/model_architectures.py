import os

import torch
from torch import nn
from torchviz import make_dot


def intermediate_mapping(user_ids, user_id_mapping):
    '''
    users ids range from 1 to 2649429, with gaps. However we have only 480189 users.
    To avoid having larger embeddings than necessary, we will map each user id to a smaller range.
    '''
    # user_ids could be one or more user ids
    if isinstance(user_ids, int):
        return torch.tensor(user_id_mapping[user_ids])
    return torch.tensor([user_id_mapping[i.item()] for i in user_ids])


class NCF(nn.Module):

    def __init__(self, n_features, user_id_mapping, hidden_layers=[64, 32, 16, 8]):
        super(NCF, self).__init__()

        self.embedding_size = 128

        self.n_features = n_features
        self.user_id_mapping = user_id_mapping

        # Embedding layers
        self.user_embedding_gmf = nn.Embedding(480190, self.embedding_size)
        self.movie_embedding_gmf = nn.Embedding(17771, self.embedding_size)
        self.user_embedding_mlp = nn.Embedding(480190, self.embedding_size)
        self.movie_embedding_mlp = nn.Embedding(17771, self.embedding_size)

        # GMF part
        self.gmf = nn.Linear(self.embedding_size, 1)

        # MLP part
        mlp_layers = []
        input_size = self.embedding_size * 2
        for layer_size in hidden_layers:
            mlp_layers.append(nn.Linear(input_size, layer_size))
            mlp_layers.append(nn.ReLU())
            input_size = layer_size
        self.mlp = nn.Sequential(*mlp_layers)
        self.final_mlp = nn.Linear(hidden_layers[-1], 1)

        # Final prediction layer
        self.final = nn.Linear(2, 1)

    def forward(self, X):

        if len(X) > 1:
            user = X[:, 0].to(torch.int64)
            movie = X[:, 1].to(torch.int64)
        else:
            user = X[0][0].to(torch.int64)
            movie = X[0][1].to(torch.int64)

        # GMF part
        if len(X) > 1:
            user_emb_gmf = self.user_embedding_gmf(intermediate_mapping(user, self.user_id_mapping))
            movie_emb_gmf = self.movie_embedding_gmf(movie)
            gmf_out = self.gmf(user_emb_gmf * movie_emb_gmf)
            user_emb_mlp = self.user_embedding_mlp(intermediate_mapping(user, self.user_id_mapping))
            movie_emb_mlp = self.movie_embedding_mlp(movie)
            mlp_in = torch.cat([user_emb_mlp, movie_emb_mlp], dim=-1)
            mlp_out = self.final_mlp(self.mlp(mlp_in))

            final_in = torch.cat([gmf_out, mlp_out], dim=-1)
            out = self.final(final_in)
            return out
        else:
            user_emb_gmf = self.user_embedding_gmf(intermediate_mapping(int(X[0][0].item()), self.user_id_mapping))
            movie_emb_gmf = self.movie_embedding_gmf(movie)
            gmf_out = self.gmf(user_emb_gmf * movie_emb_gmf)
            user_emb_mlp = self.user_embedding_mlp(intermediate_mapping(int(X[0][0].item()), self.user_id_mapping))
            movie_emb_mlp = self.movie_embedding_mlp(movie)

            mlp_in = torch.cat([user_emb_mlp, movie_emb_mlp], dim=-1)
            mlp_out = self.final_mlp(self.mlp(mlp_in))

            final_in = torch.cat([gmf_out, mlp_out], dim=-1)
            out = self.final(final_in.unsqueeze(0))
            return out


class RecommendationNN(nn.Module):
    '''
    Recommendation neural network model based on collaborative filtering.
    '''

    def __init__(self, n_features, user_id_mapping):
        super().__init__()
        self.n_features = n_features
        self.user_id_mapping = user_id_mapping
        self.user_embedding = nn.Embedding(480190, 128)
        self.item_embedding = nn.Embedding(17771, 128)
        self.fc1 = nn.Linear(128, 1)

    def forward(self, X, **kwargs):

        if len(X) > 1:
            users = X[:, 0].to(torch.int64)
            items = X[:, 1].to(torch.int64)
            user_embedding = self.user_embedding(intermediate_mapping(users, self.user_id_mapping))
            item_embedding = self.item_embedding(items)
            X = torch.mul(user_embedding, item_embedding)
            X = self.fc1(X)
            return X

        user_embedding = self.user_embedding(intermediate_mapping(int(X[0][0].item()), self.user_id_mapping))
        item_embedding = self.item_embedding(X[0][1].to(torch.int64))
        X = torch.mul(user_embedding, item_embedding)
        X = self.fc1(X.unsqueeze(0))
        return X


def print_nn_architecture(model_name):
    if model_name == 'RecommendationNN':
        model = RecommendationNN(1)
    x = torch.tensor([[2400647, 11573]])
    y = model(x)
    dot = make_dot(y, params=dict(model.named_parameters()))
    dot.format = 'png'
    dot.render('nn_model', directory=os.getcwd(), cleanup=True)