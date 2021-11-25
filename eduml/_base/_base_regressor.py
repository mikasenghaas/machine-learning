from ._base import BaseModel

from ..metrics import mean_squared_error

class BaseRegressor(BaseModel):
    def __init__(self):
        super().__init__() 

    def score(self, X):
        if not self.is_fitted():
            raise ModelNotFittedError(f'{self._model_name} is not fitted yet.')

        training_preds = self.predict(self.X)
        return mean_squared_error(self.y, training_preds)
