from sklearn.base import BaseEstimator
from typing import Tuple
from sklearn import base
from doptools.estimators.ad_estimators import *
import pandas as pd

class ConsensusModel(BaseEstimator):
    def __init__(self, pipelines):
        self.model_type = "R"
        self.ad_type = None
        if isinstance(pipelines[0], Tuple):
            self.names = [p[0] for p in pipelines]
            self.pipelines = [p[1] for p in pipelines]
        else:
            self.names = ["model"+str(i+1) for i in range(len(pipelines))]
            self.pipelines = pipelines

        if isinstance(self.pipelines[0], PipelineWithAD):
            self.ad_type = self.pipelines[0].ad_type
            if issubclass(self.pipelines[0].pipeline[-1].__class__, base.ClassifierMixin):
                self.model_type = "C"
        else:
            if issubclass(self.pipelines[0][-1].__class__, base.ClassifierMixin):
                self.model_type = "C"

    def fit(self, X, y=None):
        for p in self.pipelines:
            p.fit(X, y)
        self.is_fitted_ = True
        return self

    def predict(self, X, y=None, output="all"):
        preds = []

        if self.ad_type is None:
            preds = np.array([p.predict(X) for p in self.pipelines]).T
            res = pd.DataFrame(preds, columns=self.names)
            if self.model_type == "R":
                res["Pred.Avg."] = res.mean(axis=1)
            if self.model_type == "C":
                res["Pred.Avg."] = res.mode(axis=1)
            res["Pred.StD."] = res.std(axis=1)
        else:
            res = pd.concat([p.predict(X) for p in self.pipelines], axis=1)
            col_names = []
            for n in self.names:
                col_names += [n, "AD_"+n]
            res.columns = col_names
            if self.model_type == "R":
                res["Pred.Avg."] = res[self.names].mean(axis=1)
            if self.model_type == "C":
                res["Pred.Avg."] = res[self.names].mode(axis=1)
            res["Pred.StD."] = res[self.names].std(axis=1)
            res["%AD"] = (res[["AD_"+n for n in self.names]]>0).sum(axis=1)/len(self.names)
        
        if output=="avg":
            return res[["Pred.Avg.", "Pred.StD."]]
        elif output=="all":
            return res
        elif output=="preds":
            return res[self.names]

    def predict_within_AD(self, X, y=None, output="all"):
        if self.ad_type is None:
            return self.predict(X, y, output)
        else:
            preds = self.predict(X, y, output="all")
            preds = preds[preds["%AD"]>0]
            if output=="avg":
                return preds[["Pred.Avg.", "Pred.StD."]]
            elif output=="all":
                return preds
            elif output=="preds":
                return preds[self.names]