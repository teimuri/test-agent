from clearml import Task, TaskTypes, Optimizer

# Create HPO controller task
task = Task.init(project_name="HPO Example", task_name="HPO Controller", task_type=TaskTypes.optimizer)

# Define optimization
optimizer = Optimizer(
    base_task_id="your-train-task-id-here",
    execution_queue="default",
    optimizer_class="RandomSearch",
    objective_metric_title="accuracy",
    objective_metric_series="test",
    objective_metric_sign="max",
    total_max_jobs=10
)


optimizer.set_param_range("Args/n_estimators", [50, 100, 200])
optimizer.set_param_range("Args/max_depth", [2, 3, 5, 7])

# Start optimization
optimizer.start()
print("HPO started!")
