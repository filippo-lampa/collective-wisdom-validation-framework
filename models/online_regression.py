import math
from abc import ABC, abstractmethod
from sklearn.metrics import mean_squared_error, mean_absolute_error


class OnlineRegression(ABC):
    def __init__(self, model, model_name):
        self.model = model
        self.model_name = model_name

    @abstractmethod
    def train(self, samples):
        pass

    @abstractmethod
    def train_many(self, dataset):
        pass

    @abstractmethod
    def evaluate_progressive(self, dataset):
        pass

    def predict(self, samples):
        for x, y in samples.items():
            self.model.predict_one({x: y})
        return self.model.predict_one({x: y})

    def evaluate(self, evaluation_set):
        predictions = []
        for index, row in evaluation_set.iterrows():
            x = {
                'user': row['user'],
                'item': row['item']
            }
            predictions.append(self.model.predict_one(x=x, user=x['user'], item=x['item']))
        mae = mean_absolute_error(evaluation_set['Rating'], predictions)
        rmse = math.sqrt(mean_squared_error(evaluation_set['Rating'], predictions))
        return mae, rmse





