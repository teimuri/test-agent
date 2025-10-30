"""
ClearML Task that clones a git repo and runs training code
This creates a task that ClearML agents can execute remotely
"""

from clearml import Task

# Create the task
task = Task.init(
    project_name='taha-sama',
    task_name='Training from Git Repo',
    task_type=Task.TaskTypes.training
)

# Configure the git repository
task.set_repo(
    repo='https://github.com/teimuri/test-agent',
    branch='main',  # or 'master', 'dev', etc.
    commit=None,  # Use None for latest, or specify a commit hash
)

# Set the script to run from the repo
task.set_script(
    entry_point='test.py',  # Path to your training script in the repo
    working_dir='.',  # Working directory relative to repo root (use '.' for root)
)

# Configure hyperparameters that will be passed to your script
# These can be modified in the UI before running
hyperparameters = {
    'General': {
        'epochs': 100,
        'batch_size': 32,
        'learning_rate': 0.001,
    }
}
task.connect(hyperparameters)
# Set required Python packages
# Option 1: Auto-detect from requirements.txt in repo
task.set_packages([
    'torch==2.1.2',
    'numpy',
    'pandas',
    'networkx<3.0',
])
# Option 2: Manually specify packages
# task.set_packages([
#     'torch==2.0.0',
#     'numpy>=1.20.0',
#     'pandas',
# ])

# Set base Docker image (optional, for containerized execution)
# task.set_base_docker(
#     docker_image='nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04',
#     docker_arguments='--ipc=host',
# )

# Add environment variables (optional)
# task.set_environment(
#     CUDA_VISIBLE_DEVICES='0',
#     PYTHONPATH='/workspace',
# )

# Add script arguments (optional)
# task.set_script_args(
#     '--config', 'config.yaml',
#     '--output-dir', './outputs',
# )
print(f"Task created: {task.id}")
print(f"Task URL: {task.get_output_log_web_page()}")

# Mark task as completed (required before enqueuing)
task.mark_stopped()
# Enqueue the task to be executed by an agent
print("\nEnqueuing task...")
Task.enqueue(task=task.id, queue_name='taha-san_queue')  # Change 'default' to your queue name

print(f"\nTask enqueued successfully!")
print(f"Make sure you have an agent running on queue 'default':")
print("  clearml-agent daemon --queue default --docker")
