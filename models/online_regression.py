from abc import ABC, abstractmethod

class OnlineRegression(ABC):
    def __init__(self, model, model_name):
        self.model = model
        self.model_name = model_name

    def train(self, samples):
        for x, y in samples.items():
            self.model.learn_one({x: y})
        self.evaluate(self.model)

    @abstractmethod
    def evaluate(self, dataset):
        pass

    def predict(self, samples):
        for x, y in samples.items():
            self.model.predict_one({x: y})
        return self.model.predict_one({x: y})

