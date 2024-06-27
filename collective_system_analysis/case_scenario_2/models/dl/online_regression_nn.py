'''
Online neural network to tackle a regression problem, implemented using the river ml library on the following case scenario:
Collaborative Movie Recommendation System for Hotels: guests can select a movie profile upon check-in.
Hotels (agents) exchange data with nearby hotels, sharing entries consisting of the info about the guest and the chosen
profile, enhancing the guest experience through personalized recommendations.

The dataset used is contained in the folder 'data' and is described as follows:
TRAINING DATASET FILE DESCRIPTION
================================================================================

The file "training_set.tar" is a tar of a directory containing 17770 files, one
per movie.  The first line of each file contains the movie id followed by a
colon.  Each subsequent line in the file corresponds to a rating from a customer
and its date in the following format:

CustomerID,Rating,Date

- MovieIDs range from 1 to 17770 sequentially.
- CustomerIDs range from 1 to 2649429, with gaps. There are 480189 users.
- Ratings are on a five star (integral) scale from 1 to 5.
- Dates have the format YYYY-MM-DD.

MOVIES FILE DESCRIPTION
================================================================================

Movie information in "movie_titles.txt" is in the following format:

MovieID,YearOfRelease,Title

- MovieID do not correspond to actual Netflix movie ids or IMDB movie ids.
- YearOfRelease can range from 1890 to 2005 and may correspond to the release of
  corresponding DVD, not necessarily its theaterical release.
- Title is the Netflix movie title and may not correspond to
  titles used on other sites.  Titles are in English.

Our goal is to predict the rating of a movie given a user and a movie id.

The types of incremental learning we are interested in are domain incremental learning and class incremental learning.
Domain incremental learning means new instances of data can be added to the dataset, while class incremental learning
means new classes can be added to the dataset.
'''
import torch
#from river import neural_net as nn
from river import metrics
from river.evaluate import progressive_val_score

from collective_system_analysis.case_scenario_2.models.online_regression import OnlineRegression


class OnlineRegressionNN(OnlineRegression):

    def __init__(self, model, model_name, should_resume=False, resumed_model_id=None):
        super().__init__(model, model_name)
        self.should_resume = should_resume
        self.model_resumed = False  # needed given the lazy loading of the model
        self.resumed_model_id = resumed_model_id

    def train(self, sample):
        x = {
            'user': sample['user'],
            'item': sample['item']
        }
        self.model.learn_one(x=x, y=sample["Rating"])
        if self.should_resume and not self.model_resumed:
            # needed given the lazy loading of the model only when the first sample is being processed
#            self.model.learn_one(x=x, y=sample["Rating"])
            self.model_resumed = True
            self.resume_model()
        loss = abs(sample["Rating"] - self.model.predict_one(x=x))

        return loss

    def train_many(self, dataset):
        X = dataset.drop(columns=['Rating'])
        y = dataset['Rating']
        self.model.learn_many(X, y)
        if self.should_resume and not self.model_resumed:
            # needed given the lazy loading of the model only when the first sample is being processed
#            self.model.learn_one(x=X.iloc[0], y=y.iloc[0])
            self.model_resumed = True
            self.resume_model()
        losses = []
        for index, row in dataset.iterrows():
            x = {
                'user': row['user'],
                'item': row['item']
            }
            loss = abs(row['Rating'] - self.model.predict_one(x=x))
            losses.append(loss)

        return losses

    def resume_model(self):
        with open('running_agents_models/agent_{}_model.pt'.format(self.resumed_model_id), 'rb') as f:
            checkpoint = torch.load(f)
            self.model.module.load_state_dict(checkpoint['model'])
            self.model.optimizer.load_state_dict(checkpoint['optimizer'])

    def evaluate_progressive(self, dataset):
        metric = metrics.MAE() + metrics.RMSE()
        x_y = []
        #correct_predictions = 0
        for index, row in dataset.iterrows():
            x = {
                'user': row['user'],
                'item': row['item']
            }
            y = row['Rating']
            x_y.append((x, y))
            '''
            prediction = self.model.predict_one(x=x)
            if int(prediction) == y:
                correct_predictions += 1
            '''
        score = progressive_val_score(x_y, model=self.model, metric=metric)
        return score.data[0], score.data[1]

