# Core package of DeepJanus
This folder contains the DeepJanus algorithm and a set of abstract classes and utilities 
that are useful to easily adapt DeepJanus to specific problems.

The main algorithm can be found in `deepjanus/nsga2.py`.

## Installation
This package has been written and tested using Python 3.11.

To install it, cd to this directory and run `pip install .`.

## Usage
This package is not a standalone utility, it is meant to help apply the DeepJanus approach 
to different problems.

You will need to implement the following abstract classes:
- `Config`
- `Problem`
- `Individual`
- `Member`
- `Evaluator`
- `Mutator`

To start a DeepJanus run you will need seeds. There is a utility class in `deepjanus.seed_pool` to help preparing them:
```python
from deepjanus.seed_pool import SeedFileGenerator

# Define a generator that produces members of your problem
def seed_candidate_generator():
    while True:
        # Your code to generate new members here
        # ...
        yield member

# The seed generator will select candidates that satisfy requirements for all given problems
# and will save them to 'data/seeds/<folder_name>/'
seed_generator = SeedFileGenerator([problem1, problem2], folder_name, seed_candidate_generator())
seed_generator.generate_seeds(quantity)
```

To run DeepJanus (`YourConfig` and `YourProblem` are the subclasses you implemented):
```python
from datetime import datetime
from deepjanus.archive import SmartArchive
from deepjanus.log import log_setup
from deepjanus.nsga2 import main

# Define problem and tool configuration
config = YourConfig()
archive = SmartArchive(config.TARGET_ERROR, config.ARCHIVE_THRESHOLD)
problem = YourProblem(config, archive)

# Enable logging (optional)
# You can install package 'rich' for better console logs
log_setup.setup_console_log(config.FOLDERS.log_ini)
log_setup.setup_file_log(problem.experiment_path
                         .joinpath(datetime.strftime(datetime.now(), '%d-%m-%Y_%H-%M-%S') + '.log'))

# Start DeepJanus
main(problem)
```