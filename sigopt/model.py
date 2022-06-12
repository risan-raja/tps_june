import sklearn.datasets, sklearn.metrics
from xgboost import XGBClassifier
import sigopt

X, y = sklearn.datasets.load_iris(return_X_y=True)
Xtrain, ytrain = X[100:], y[100:]
sigopt.log_dataset('iris 2/3 training, full test')
sigopt.params.setdefault("n_estimators", 200)
model = XGBClassifier(n_estimators=sigopt.params.n_estimators)
sigopt.log_model('xgboost')
model.fit(Xtrain, ytrain)
pred = model.predict(X)
sigopt.log_metric("accuracy", sklearn.metrics.accuracy_score(pred, y))
