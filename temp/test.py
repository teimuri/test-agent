from clearml import Task
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import argparse

# Initialize ClearML task
task = Task.init()
params = {
    'n_estimators': None,
    'max_depth': None,
}
params = task.connect(params)

print(f"n_estimators: {params['n_estimators']}, max_depth: {params['max_depth']}")
# Load dataset
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
clf = RandomForestClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'])
clf.fit(X_train, y_train)

# Evaluate
acc = accuracy_score(y_test, clf.predict(X_test))

# Report metric to ClearML
logger = task.get_logger()
logger.report_scalar(
    title='accuracy',      # This is objective_metric_title
    series='test',         # This is objective_metric_series
    value=acc,
    iteration=0)
print(f"Accuracy: {acc:.4f}")
