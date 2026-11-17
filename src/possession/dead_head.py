"""A binary alive/dead head exposed through the common H/A/D probability API."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class DeadBallClassifier(ClassifierMixin, BaseEstimator):
    """Learn P(D) only; leave conditional H/A neutral for a separate team model.

    Preprocessing is fitted inside each training fold. Single-target folds use a
    constant classifier rather than borrowing a model fitted to held-out labels.
    """
    def __init__(self, C=1.):
        self.C = C

    def fit(self, x, y):
        if not np.isfinite(self.C) or self.C <= 0:
            raise ValueError('dead-head logistic C must be finite and positive')
        dead = (np.asarray(y) == 2).astype(np.int8)
        self.model_ = (DummyClassifier(strategy='prior') if np.unique(dead).size < 2 else
                       make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                                     LogisticRegression(C=self.C, max_iter=3000)))
        self.model_.fit(x, dead)
        self.classes_ = np.arange(3)
        self.n_features_in_ = x.shape[1]
        return self

    def predict_proba(self, x):
        classes = list(self.model_.classes_)
        dead = (self.model_.predict_proba(x)[:, classes.index(1)] if 1 in classes else
                np.zeros(len(x)))
        return np.column_stack(((1 - dead) / 2, (1 - dead) / 2, dead))

    def predict(self, x):
        return self.classes_[self.predict_proba(x).argmax(axis=1)]


class ContextDeadBallClassifier(DeadBallClassifier):
    """Average the original phase head and a head with availability history.

    Both preprocessors and classifiers are fitted on the same training rows.
    The original columns come first; history columns follow them. H/A remains
    neutral, so this model supplies only dead-ball evidence.
    """
    def __init__(self, C=.3, base_features=18):
        self.C = C
        self.base_features = base_features

    def fit(self, x, y):
        x = np.asarray(x)
        if (x.ndim != 2 or not isinstance(self.base_features, (int, np.integer))
                or isinstance(self.base_features, bool) or not 0 < self.base_features < x.shape[1]):
            raise ValueError('context head needs original columns followed by extra features')
        self.base_ = DeadBallClassifier(self.C).fit(x[:, :self.base_features], y)
        self.context_ = DeadBallClassifier(self.C).fit(x, y)
        self.classes_ = np.arange(3)
        self.n_features_in_ = x.shape[1]
        return self

    def predict_proba(self, x):
        x = np.asarray(x)
        return .5*self.base_.predict_proba(x[:, :self.base_features])+.5*self.context_.predict_proba(x)
