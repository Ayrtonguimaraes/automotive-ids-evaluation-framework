import typing

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODELS_FACTORY = {
    "RandomForestClassifier": RandomForestClassifier
}

class SklearnClassifier():
    def __init__(self, model_hyperparams: typing.Dict):
        super(SklearnClassifier, self).__init__()
        self._model_name = model_hyperparams["model_name"]
        self._model_params = model_hyperparams["model_params"]

        if self._model_name not in MODELS_FACTORY:
            raise KeyError(f"Selected model {self._model_name} is NOT available!")

        self._model = self.__build_pipeline()

    def __build_pipeline(self):
        pipeline_steps = [
            ('scaler', StandardScaler()),
            ('clf', MODELS_FACTORY[self._model_name](**self._model_params))
        ]

        return Pipeline(pipeline_steps)

    def reset(self):
        # Rebuild the same pipeline (scaler + classifier) so every fold is trained the same way
        self._model = self.__build_pipeline()

    def train(self, X_data, y_data):
        self._model = self._model.fit(X_data, y_data)

    def predict(self, X_data):
        y_pred = self._model.predict(X_data)

        return y_pred

    def predict_proba(self, X_data):
        y_pred_proba = self._model.predict_proba(X_data)

        return y_pred_proba

