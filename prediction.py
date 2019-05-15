import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import median_absolute_error, r2_score, make_scorer
from util.plot import plot_dictionary


class RetentionTimePrediction:

    def __init__(self, x_file_path='../data/x.csv', y_file_path='../data/y.csv'):
        self.X = np.genfromtxt(x_file_path, delimiter=',')
        self.y = np.genfromtxt(y_file_path, delimiter=',').reshape(self.X.shape[0], 1)
        self.x_learning, self.x_test, self.y_learning, self.y_test = train_test_split(self.X, self.y, test_size=0.1,
                                                                                      random_state=0)
        self.scorers = {"median": make_scorer(median_absolute_error, greater_is_better=False),
                        "r2": make_scorer(r2_score)}
        self.metrics = {
            "median": median_absolute_error,
            "r2": r2_score
        }
        self.methods = ["LR", "SVR_rbf", "SVR_linear", "GB"]
        self.val_errors = {}
        self.test_errors = {}

    def train(self, method, error_func):
        model = None
        val_error = 0.0
        if method is "LR":
            model, val_error = self.linear_regression_train(error_func)
        elif method is "SVR_rbf":
            model, val_error = self.svr_train("rbf", error_func)
        elif method is "SVR_linear":
            model, val_error = self.svr_train("poly", error_func)
        elif method is "GB":
            model, val_error = self.gb_train(error_func)
        return model, val_error

    def best_method(self, error_func):
        for method in self.methods:
            print("Starting " + method)
            model, validation_error = self.train(method, error_func)
            self.val_errors[method] = validation_error
            # self.test_errors[method] = self.test_error(model, error_func)
            print("Ended " + method + " with validation error:", validation_error)

        plot_dictionary(self.val_errors, "method", "validation error", "method to validation error")

    def test_error(self, model, error_function="median"):
        y_pred = model.predict(self.x_test)
        return self.metrics[error_function](self.y_test, y_pred)

    def linear_regression_train(self, scorer="median"):
        ridge = Ridge()
        parameters = {'alpha': list(np.arange(10e-5, 1, 10e2))}
        clf = GridSearchCV(ridge, parameters, cv=10, return_train_score=True, scoring=self.scorers[scorer])
        clf.fit(self.x_learning, self.y_learning)
        return clf.best_estimator_, clf.best_score_

    def svr_train(self, kernel="rbf", scorer="median"):
        svr = SVR(gamma='scale', kernel=kernel)
        parameters = {'C': list(np.arange(1, 10, 1)), 'epsilon': list(np.arange(0.01, 1, 0.1))}
        clf = GridSearchCV(svr, parameters, cv=10, return_train_score=True, scoring=self.scorers[scorer])
        clf.fit(self.x_learning, self.y_learning)
        return clf.best_estimator_, clf.best_score_

    def gb_train(self, scorer="median"):
        gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                        max_depth=1, random_state=0, loss='ls')
        parameters = {
            "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
            "min_samples_split": np.linspace(0.1, 0.5, 12),
            "min_samples_leaf": np.linspace(0.1, 0.5, 12),
            "max_depth": [3, 5, 8],
            "max_features": ["log2", "sqrt"],
            "criterion": ["friedman_mse", "mae"],
            "subsample": [0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
            "n_estimators": [10, 100, 1000]
        }
        clf = GridSearchCV(gbr, parameters, cv=10, return_train_score=True, scoring=self.scorers[scorer])
        clf.fit(self.x_learning, self.y_learning)
        return clf.best_estimator_, clf.best_score_


if __name__ == "__main__":
    ret_time_pred = RetentionTimePrediction()
    ret_time_pred.best_method("median")
    print(ret_time_pred.X.shape, ret_time_pred.y.shape)
