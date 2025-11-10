from clearml import Task
from clearml.automation import HyperParameterOptimizer, UniformIntegerParameterRange, DiscreteParameterRange
from clearml.automation import RandomSearch, GridSearch  # or other strategy

# initialize the HPO controller task
task = Task.init(
    project_name='Hyper-Parameter Optimization',
    task_name='Automatic Hyper-Parameter Optimization',
    task_type=Task.TaskTypes.optimizer,
    reuse_last_task_id=False
)

# connect args
args = {
    'template_task_id': None,
    'run_as_service': False,
}
args = task.connect(args)

if not args['template_task_id']:
    args['template_task_id'] = Task.get_task(
        project_name='Your Project',
        task_name='Your Base Training Task'
    ).id

# choose search strategy
try:
    from clearml.automation.optuna import OptimizerOptuna
    search_strategy = OptimizerOptuna
except ImportError:
    try:
        from clearml.automation.hpbandster import OptimizerBOHB
        search_strategy = OptimizerBOHB
    except ImportError:
        search_strategy = RandomSearch

# create optimizer
optimizer = HyperParameterOptimizer(
    base_task_id=args['template_task_id'],
    hyper_parameters=[
        UniformIntegerParameterRange('Args/n_estimators', min_value=50, max_value=200, step_size=50),
        DiscreteParameterRange('Args/max_depth', values=[2,3,5,7]),
    ],
    objective_metric_title='accuracy',
    objective_metric_series='test',
    objective_metric_sign='max',
    max_number_of_concurrent_tasks=2,
    optimizer_class=search_strategy,
    execution_queue='taha-san_queue',
    total_max_jobs=10
)

# start optimization
optimizer.set_report_period(0.1)
optimizer.start()
optimizer.wait()
top_experiments = optimizer.get_top_experiments(top_k=3)
print("Top experiments:", [t.id for t in top_experiments])
optimizer.stop()
