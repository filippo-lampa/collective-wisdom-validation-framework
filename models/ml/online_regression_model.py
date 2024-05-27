'''
Perform regression using online learning machine learning models from the river ml library on the following case scenario:
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

from river import metrics, datasets
from river.evaluate import progressive_val_score, iter_progressive_val_score

from models.online_regression import OnlineRegression


class OnlineRegressionModel(OnlineRegression):
    def __init__(self, model, model_name):
        super().__init__(model, model_name)

    def train(self, sample):
        x = {
            'user': sample['user'],
            'item': sample['item']
        }
        self.model.learn_one(user=sample['user'], item=sample["item"], y=sample["Rating"], x=x)
        loss = abs(sample["Rating"] - self.model.predict_one(user=sample['user'], item=sample['item'], x=x))
        return loss


    def evaluate(self, dataset):
        metric = metrics.MAE() + metrics.RMSE()
        x_y = []
        correct_predictions = 0
        for index, row in dataset.iterrows():
            x = {
                'user': row['user'],
                'item': row['item']
            }
            y = row['Rating']
            x_y.append((x, y))
            prediction = self.model.predict_one(user=x['user'], item=x['item'], x=x)
            if int(prediction) == y:
                correct_predictions += 1
        score = progressive_val_score(x_y, model=self.model, metric=metric)
        return score.data[0], score.data[1], correct_predictions / len(dataset)
