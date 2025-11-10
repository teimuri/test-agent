from clearml import Task
from clearml.automation import HyperParameterOptimizer, UniformIntegerParameterRange, DiscreteParameterRange
from clearml.automation import RandomSearch, GridSearch  # or other strategy

from clearml import Task

from clearml import Task

task = Task.init(
    project_name='taha-sama',
    task_name='train_model_from_repo',
    task_type=Task.TaskTypes.training
)

# Configure the git repository
task.set_repo(
    repo='https://github.com/teimuri/test-agent',
    branch='HPO',  # or 'master', 'dev', etc.
    commit=None,  # Use None for latest, or specify a commit hash
)

task.set_script(
    entry_point='temp/test.py',  # Path to your training script in the repo
    working_dir='.',  # Working directory relative to repo root (use '.' for root)
)

task.set_packages([
         'scikit-learn',
         'clearml',
         'networkx<3.5',  # Pin to older version compatible with Python 3.10
         # Add other packages you need
     ])

# Task.enqueue(task=task.id,queue_name="taha-san_queue")
# task = Task.init(
#         project_name="taha-sama",
#         task_name="train_model_from_repo",
#         repo="https://github.com/teimuri/test-agent.git",
#         branch="HPO",               # or any branch name
#         commit=None,                 # or a specific commit hash
#         script="temp/train.py",           # file inside that repo
#         working_directory=".",       # relative to repo root
#         task_type=Task.TaskTypes.training,
#         )

# base_task = task
# task.set_packages([
#          'scikit-learn',
#          'clearml',
#          'networkx<3.5',  # Pin to older version compatible with Python 3.10
#          # Add other packages you need
#      ])
params = {
    'n_estimators': 100,
    'max_depth': 5,
}

params = task.connect(params)
task.mark_stopped()
# Task.enqueue(task=task.id,queue_name="taha-san_queue")

# initialize the HPO controller task
task = Task.init(
    project_name='Hyper-Parameter Optimization',
    task_name='Automatic Hyper-Parameter Optimization',
    task_type=Task.TaskTypes.optimizer,
    reuse_last_task_id=False
)

# connect args
args = {
    'n_estimators': 100,
    'max_depth': 5,
}
args = task.connect(args)
task.mark_stopped()

from clearml.automation.optuna import OptimizerOptuna
search_strategy = OptimizerOptuna

# create optimizer
optimizer = HyperParameterOptimizer(
    base_task_id=base_task.id,
    hyper_parameters=[
        UniformIntegerParameterRange('General/n_estimators', min_value=50, max_value=200, step_size=50),
        DiscreteParameterRange('General/max_depth', values=[2,3,5,7]),
    ],
    objective_metric_title='accuracy',
    objective_metric_series='test',
    objective_metric_sign='max',
    max_number_of_concurrent_tasks=2,
    optimizer_class=search_strategy,
    execution_queue='taha-san_queue',
    total_max_jobs=10,
    min_iteration_per_job=0,
    max_iteration_per_job=0,
)

# start optimization
optimizer.set_report_period(0.1)
optimizer.start()
optimizer.wait()
top_experiments = optimizer.get_top_experiments(top_k=3)
print("Top experiments:", [t.id for t in top_experiments])
optimizer.stop()
