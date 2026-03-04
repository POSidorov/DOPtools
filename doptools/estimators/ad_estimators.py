from copy import deepcopy
from typing import Any, Iterable, List, Optional, Union

import pandas as pd
from pandas import DataFrame
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.datasets import load_svmlight_file
from sklearn.utils.validation import check_is_fitted


class FragmentControl(BaseEstimator, OutlierMixin):
    def __init__(self, pipeline: Any) -> None:
        self.pipeline: Any = pipeline
        self.fragmentor: Any = deepcopy(pipeline[0])
        self.feature_names: List[str] = []
        try:
            check_is_fitted(self.pipeline)
            self.feature_names = pipeline[0].get_feature_names()
        except Exception:
            print("The pipeline is not fitted, you should fit it.")

    def fit(self, X: Any, y: Optional[Iterable[Any]] = None) -> "FragmentControl":
        self.pipeline.fit(X, y)
        self.fragmentor = deepcopy(self.pipeline[0])
        self.feature_names = self.pipeline[0].get_feature_names()
        self.is_fitted_ = True
        return self

    def predict(
        self, X: Union[DataFrame, List[Any]], y: Optional[Iterable[Any]] = None
    ) -> List[int]:
        res: List[int] = []
        for i in range(len(X)):
            if isinstance(X, DataFrame):
                x = X.iloc[i]
            else:
                x = [X[i]]
            self.fragmentor.fit(x)
            features = self.fragmentor.get_feature_names()
            if len(set(features) - set(self.feature_names)) > 0:
                res.append(-1)
            else:
                res.append(1)
        return res


class BoundingBox(BaseEstimator, OutlierMixin):
    def __init__(self, pipeline: Any) -> None:
        self.pipeline: Any = pipeline
        self.fragmentor: Any = deepcopy(pipeline[0])

    def fit(
        self,
        X: Any,
        y: Optional[Iterable[Any]] = None,
        svm_file: Optional[str] = None,
    ) -> "BoundingBox":
        self.is_fitted_ = True
        if svm_file is not None:
            d, _ = load_svmlight_file(svm_file)
            descs = d.toarray()
        else:
            descs = self.fragmentor.fit_transform(X)
        self.min_limits = descs.min(axis=0)
        self.max_limits = descs.max(axis=0)
        return self

    def predict(
        self, X: Union[DataFrame, List[Any]], y: Optional[Iterable[Any]] = None
    ) -> List[int]:
        res: List[int] = []
        for i in range(len(X)):
            if isinstance(X, DataFrame):
                x = X.iloc[i]
            else:
                x = [X[i]]
            desc = self.fragmentor.transform(x)
            value = 1
            for c in desc.columns:
                if (
                    desc.iloc[0][c] > self.max_limits[c]
                    or desc.iloc[0][c] < self.min_limits[c]
                ):
                    value = -1
            res.append(value)
        return res


class PipelineWithAD(BaseEstimator):
    def __init__(
        self, pipeline: Any, ad_type: str, threshold: Optional[float] = None
    ) -> None:
        self.ad_type: str = ad_type
        self.pipeline: Any = pipeline
        self.threshold: Optional[float] = threshold
        if self.ad_type == "FragmentControl":
            self.ad_estimator = FragmentControl(self.pipeline)
        elif self.ad_type == "BoundingBox":
            self.ad_estimator = BoudingBox(  # type: ignore[name-defined]  # noqa: F821
                self.pipeline
            )

    def fit(self, X: Any, y: Optional[Iterable[Any]] = None) -> "PipelineWithAD":
        self.is_fitted_ = True
        self.pipeline.fit(X, y)
        self.ad_estimator.fit(X, y)
        return self

    def predict(
        self, X: Union[DataFrame, List[Any]], y: Optional[Iterable[Any]] = None
    ) -> DataFrame:
        res: List[tuple[Any, Any]] = []
        for i in range(len(X)):
            if isinstance(X, DataFrame):
                x = X.iloc[i]
            else:
                x = [X[i]]
            res.append((self.pipeline.predict(x)[0], self.ad_estimator.predict(x)[0]))
        return pd.DataFrame(res, columns=["Predicted", "AD"])

    def predict_within_AD(
        self, X: Union[DataFrame, List[Any]], y: Optional[Iterable[Any]] = None
    ) -> DataFrame:
        res: List[Any] = []
        indices: List[int] = []
        for i in range(len(X)):
            if isinstance(X, DataFrame):
                x = X.iloc[i]
            else:
                x = [X[i]]
            if self.ad_estimator.predict(x)[0] == 1:
                res.append(self.pipeline.predict(x)[0])
                indices.append(i)
        return pd.DataFrame(res, columns=["Predicted"], index=indices)
