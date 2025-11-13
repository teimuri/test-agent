from clearml import Task
from clearml.automation import HyperParameterOptimizer, UniformIntegerParameterRange, DiscreteParameterRange
from clearml.automation.optuna import OptimizerOptuna
from datetime import datetime

# Create a fresh template task each time
base_task = Task.create(
    project_name='taha-sama',
    task_name=f'train_model_template_{datetime.now().strftime("%Y%m%d_%H%M%S")}',  # Unique name
    task_type=Task.TaskTypes.training,
    repo='https://github.com/teimuri/test-agent',
    branch='HPO',
    commit=None,
    script='temp/test.py',
    working_directory='.',
)

base_task.set_packages([
    'scikit-learn',
    'clearml',
    'networkx<3.5',
])

base_task.set_parameters({
    'General/n_estimators': 100,
    'General/max_depth': 5,
})

# Initialize the HPO controller
hpo_task = Task.init(
    project_name='taha-sama',
    task_name='HPO_optimizer',
    task_type=Task.TaskTypes.optimizer,
    reuse_last_task_id=False  # Always create new HPO task
)

# Create optimizer
optimizer = HyperParameterOptimizer(
    base_task_id=base_task.id,
    hyper_parameters=[
        UniformIntegerParameterRange('General/n_estimators', min_value=50, max_value=200, step_size=50),
        DiscreteParameterRange('General/max_depth', values=[2, 3, 5, 7]),
    ],
    objective_metric_title='accuracy',
    objective_metric_series='test',
    objective_metric_sign='max',
    max_number_of_concurrent_tasks=2,
    optimizer_class=OptimizerOptuna,
    execution_queue='taha-san_queue',
    total_max_jobs=10,
    min_iteration_per_job=0,
    max_iteration_per_job=0,
)

optimizer.set_report_period(0.1)
optimizer.start()
optimizer.wait()

top_experiments = optimizer.get_top_experiments(top_k=3)
print("Top experiments:", [t.id for t in top_experiments])
optimizer.stop()