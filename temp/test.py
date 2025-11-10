from clearml import Task
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import argparse

# Initialize ClearML task
task = Task.init(project_name="HPO Example2", task_name="train", reuse_last_task_id=False)

# Parse hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument("--n_estimators", type=int, default=100)
parser.add_argument("--max_depth", type=int, default=3)
args = parser.parse_args()

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
