from clearml import Task
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import argparse

# Initialize ClearML task
task = Task.init()
params = {
    # 'n_estimators': 100,  # default value
    # 'max_depth': 5,       # default value
}
params = task.connect(params)

print(f"n_estimators: {params['n_estimators']}, max_depth: {params['max_depth']}")
# Load dataset
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
clf = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth)
clf.fit(X_train, y_train)

# Evaluate
acc = accuracy_score(y_test, clf.predict(X_test))

# Report metric to ClearML
task.get_logger().report_scalar("accuracy", "test", iteration=0, value=acc)
print(f"Accuracy: {acc:.4f}")
