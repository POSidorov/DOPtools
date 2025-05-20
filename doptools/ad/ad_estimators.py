from sklearn.base import BaseEstimator, TransformerMixin, OutlierMixin, clone
from copy import deepcopy
from pandas import DataFrame

class PipelineWithAD(BaseEstimator):
    def __init__(self, pipeline, ad_type, threshold=None):
        self.ad_type = ad_type
        self.pipeline = pipeline
        self.threshold = threshold
        if self.ad_type == "FragmentControl":
            self.ad_estimator = FragmentControl(self.pipeline)
        elif self.ad_type == "BoundingBox":
            self.ad_estimator = BoudingBox(self.pipeline)

    def fit(self, X, y=None):
        self.is_fitted_ = True
        self.pipeline.fit(X, y)
        self.ad_estimator.fit(X, y)
        return self

    def predict(self, X, y=None):
        res = []
        for i in range(len(X)):
            if isinstance(X, DataFrame):
                x = X.iloc[i]
            else:
                x = [X[i]]
            res.append((self.pipeline.predict(x)[0], self.ad_estimator.predict(x)[0]))
        return pd.DataFrame(res, columns=["Predicted", "AD"])

class FragmentControl(BaseEstimator, OutlierMixin):
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.fragmentor = deepcopy(pipeline[0])
        self.feature_names = pipeline[0].get_feature_names()

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def predict(self, X, y=None):
        res = []
        for i in range(len(X)):
            if isinstance(X, DataFrame):
                x = X.iloc[i]
            else:
                x = [X[i]]
            self.fragmentor.fit(x)
            features = self.fragmentor.get_feature_names()
            if len(set(features) - set(self.feature_names))>0:
                res.append(-1)
            else:
                res.append(1)
        return(res)

class BoundingBox(BaseEstimator, OutlierMixin):
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.fragmentor = deepcopy(pipeline[0])

    def fit(self, X, y=None):
        self.is_fitted_ = True
        descs = self.fragmentor.fit_transform(X)
        self.min_limits = descs.min(axis=0)
        self.max_limits = descs.max(axis=0)
        return self

    def predict(self, X, y=None):
        res = []
        for i in range(len(X)):
            if isinstance(X, DataFrame):
                x = X.iloc[i]
            else:
                x = [X[i]]
            desc = self.fragmentor.transform(x)
            value = 1
            for c in desc.columns:
                if desc.iloc[0][c]>self.max_limits[c] or desc.iloc[0][c]<self.min_limits[c]:
                    value = -1
            res.append(value)
        return res