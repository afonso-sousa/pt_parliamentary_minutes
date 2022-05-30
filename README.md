# Portuguese Parliamentary Minutes Dataset

## Quick Start

### Requirements and Installation

**a. Clone the repository.**
```shell
git clone https://github.com/AfonsoSalgadoSousa/pt_parliamentary_minutes.git
```
**b. Install dependencies.**
This project uses Anaconda as its package manager. Make sure you have it installed. For more info check [Anaconda's official documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) on environment management.
We have compiled an `enviroment.yml` file with all the required dependencies. To install them, simply run:
```shell
conda env create -f environment.yml
```

### Model Training and Evaluation
Simply run:
```shell
python nlp_vote_prediction.py --model <'nb'|'lr'|'bert'>