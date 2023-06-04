# Hierarchies of Reward Machines - Learning
Implementation of the _policy and hierarchy learning_ algorithms described in the paper 
[Hierarchies of Reward Machines](#references). The hierarchies themselves and the environments are implemented 
[in this repository](https://github.com/ertsiger/hrm-formalism-envs).

1. [Installation](#installation)
   1. [Installation of a Conda Environment](#installation-of-a-conda-environment)
   2. [Installation of ILASP](#installation-of-ilasp)
   3. [Installation of Baseline Software](#installation-of-baseline-software)
2. [Usage](#usage)
   1. [Running the Code](#running-the-code)
   2. [Running the Baselines](#running-the-baselines)
   3. [Generating Experiments](#generating-experiments)
3. [Reproducibility](#reproducibility)
   1. [Generating the Paper Experiments](#generating-the-paper-experiments)
   2. [Collecting the Metrics and Plotting Curves](#collecting-the-metrics-and-plotting-curves)
4. [Citation](#citation)
5. [References](#references)

> **Disclaimer:** In line with our previous work [(Furelos Blanco et al., 2021)](#references), we used the term _hierarchy
of subgoal automata_ instead of _hierarchy of reward machines_ during the initial stages of the work; hence, the code
employs the former term. For instance, the term IHSA stood for "Induction of Hierarchies of Subgoal Automata", which we
now name LHRM ("Learning of HRM") in the paper instead.

## Installation
The code runs on Linux and MacOS computers with Python 3; however, we highlight that: 
1. All experiments ran on Ubuntu 20.04 computers.
2. The version software used for learning the reward machines, ILASP, is not available for Mac computers with Apple
processors. The software is nonetheless usable if 
[Rosetta](https://developer.apple.com/documentation/apple-silicon/about-the-rosetta-translation-environment) is 
installed, but we have not performed extensive tests in this setting.

Consequently, the instructions below have been designed for Linux systems and slight variations might be required for MacOS.

### Installation of a Conda Environment
We recommend using a [Conda](https://docs.conda.io/en/latest/) environment to run the code. To create such environment,
you can use the [Anaconda installer](https://www.anaconda.com/products/distribution), which is easy to follow. If you
need further installation instructions, you can find them [here](https://docs.anaconda.com/anaconda/install/index.html).
Once completed, go to the directory where the installation has been made (by default, your home directory) and run the
following command (note that, by default, the folder containing the distribution is named `anaconda3`) to activate the 
Conda environment:
```
$ source anaconda3/bin/activate
```

If you want to deactivate any Conda environment at some point, you just need to run:
```
$ conda deactivate
```

Next, we need to create a Conda environment running Python 3.7 and install there all the required packages. We describe 
two methods for doing this below. In both cases, [PyTorch](https://pytorch.org/) is installed for CPUs; however, the code
is prepared to run on GPUs if installed for the appropriate platform.

#### Importing the Conda Environment
We recommend this option for Linux systems since it enables using the setup we employed in our experiments. We have attempted to
follow this method in Mac systems, but it has not worked for us. The following command will create an enviroment called 
`py37` and install all the packages:
```
$ conda env create -f environment.yml
```

Then, you can run the command below to activate the created environment:
```
$ conda activate py37
```

#### Create the Conda Environment and Install the Required Packages
The alternative is to first create a Conda environment manually and then install the required dependencies separately.
To complete the first step, we create an environment called `py37` and activate it with the following commands:
```
$ conda create --name py37 python=3.7
$ conda activate py37
```

Next, we install the required dependencies using the `requirements.txt` file as follows:
```
$ pip install -r requirements/requirements.txt --find-links https://download.pytorch.org/whl/torch_stable.html
```

In case CNNs are trained using CPUs, we recommend installing the [`libjemalloc` library](https://anaconda.org/conda-forge/libjemalloc), 
which prevents memory leaks from occurring:
```
$ conda install -c conda-forge libjemalloc 
```
This issue seems to be common according to several sources [[1](https://discuss.pytorch.org/t/possible-memory-leak-on-cpu/49559/2), 
[2](https://github.com/pytorch/pytorch/issues/22127), [3](https://discuss.pytorch.org/t/memory-leaks-at-inference/85108/11), 
[4](https://github.com/pytorch/pytorch/issues/29809)]. Loading this library using `LD_PRELOAD` prior to running the code
alleviated the problem with the leaks. For an example, you can check how it is used in the experiment generator 
[here](src/config/generators/config_generator.py).

### Installation of ILASP
To install ILASP [(Law et al., 2015)](#references), the system used for learning the hierarchies of reward machines, 
we first need to install some packages using the following command:
```
$ apt-get -y install bison build-essential cmake doxygen flex g++ liblua5.3-dev libpython2.7 re2c ruby wget
```

Next, we need to install the ILASP binary as well as clingo, the ASP solver it depends on. To simplify the process,
you can use an [installation script](#automatic) that will download and compile the binaries for you; alternatively, you 
can do it [by yourself](#manual).

#### Automatic
To install ILASP and clingo binaries and libraries manually, run the following commands:
```
$ cd src
$ python -m ilasp.install_ilasp
```

#### Manual
To install ILASP and clingo binaries and libraries manually, follow these steps:
1. Create two folders inside `src`:
   1. `bin` - It will contain the `ILASP` and `clingo` (a program invoked by ILASP) binaries.
   2. `lib` - It will contain clingo's shared libraries.
2. Download the `ILASP` binary from [here](https://github.com/ilaspltd/ILASP-releases/releases/tag/v4.1.2) and place it 
in the `bin` folder.
3. Download the source code for clingo from [here](https://github.com/potassco/clingo/releases/tag/v5.5.0) (in `zip`
or `tar.gz`) and extract the content.
4. Open the resulting folder and, following the instructions of `INSTALL.md`, run the command below to compile the code:
```
$ cmake -H. -B. -DCMAKE_BUILD_TYPE=Release
$ cmake --build .
```
5. Open the resulting `bin` folder. Copy the `clingo` binary to the `bin` folder we created before. Copy `libclingo.so`, 
  `libclingo.so.4`, and `libclingo.so.4.0` to our `lib` folder. 

### Installation of Baseline Software
The approaches for learning flat hierarchies [(Toro Icarte et al., 2019; Xu et al., 2020; Hasanbeig et al., 2021)](#references) 
we compare to in the paper use software different from ILASP. As described in the paper, we solely evaluate the 
performance of the reward machine learning part (i.e., not the policy learning one). The relevant parts of the code have
been copied (and sometimes slightly modified for fairness and ease the collection of metrics, as described in our paper) 
and can be found [here](src/baselines). Details on running these baselines are provided later.

#### General
To install the requirements for these baselines, run the following command:
```
$ pip install -r requirements/requirements_baselines.txt 
```

You may have problems installing [PySat](https://pysathq.github.io/installation/) in OSX systems.
This package is a requirement of JIRP [(Xu et al., 2020)](#references); hence, if you do not intend to use JIRP, you can comment out
the `python-sat` package in the requirements file mentioned above.

#### DeepSynth (Hasanbeig et al., 2021)
This system uses [CBMC](https://www.cprover.org/cbmc/) to learn the reward machines. 
For simplicity, we copy the installation instructions refereed in DeepSynth's repo, which are found [here](https://github.com/natasha-jeppu/Trace2Model).
```
$ git clone https://github.com/diffblue/cbmc.git
$ cd cbmc
$ git reset --hard 25ba4e6a600b033df7dbaf3d19437afd8b9bdd1c
$ make -C src minisat2-download
$ make -C src
```

The directory where CBMC is installed will be needed to run DeepSynth, as detailed in the following sections.

## Usage

### Running the Code
To run our policy and/or hierarchy learning algorithms, the command line is:
```
$ python run_algorithm.py [-h] config_file
```
where `config_file` is a JSON file with the experiment configuration. We [later](#generating-experiments) explain how
you can create these configuration files through a simple script. The `config/examples/ihsa` folder provides several 
illustrative configuration files we have used in our experiments:

| Folder                      | Description                                                                             |
|-----------------------------|-----------------------------------------------------------------------------------------|
| `01-cw-op`                  | Learn the HRMs for the CraftWorld tasks in the open plan (OP) setting.                  |
| `02-ww-wod`                 | Learn the HRMs for the WaterWorld tasks in the without-deadends (WOD) setting.          |
| `03-cw-op-mb-learn-flat`    | Learn the flat HRM for CraftWorld's MilkBucket in the OP setting.                       |
| `04-cw-op-book-learn-flat`  | Learn the flat HRM for CraftWorld's Book task in the OP setting.                        |
| `05-cw-frl-mb-exploit-flat` | Exploits a flat HRM for CraftWorld's MilkBucket in the four rooms with lava (FRL) setting. |
| `06-cw-frl-mb-exploit-non-flat`| Exploits a non-flat HRM for CraftWorld's MilkBucket in the FRL setting.                 |
|`07-cw-frl-bq-exploit-flat` | Exploits a flat HRM for CraftWorld's BookQuill in the FRL setting.                      |
|`08-cw-frl-bq-exploit-nonflat-hrm` | Exploits a non-flat HRM for CraftWorld's BookQuill in the FRL setting.                  |

### Running the Baselines
To run the baselines for learning flat hierarchies (i.e., regular reward machines), the following command is run:
```
$ python run_baseline.py [-h]  --algorithm ALGORITHM --domain DOMAIN --task TASK --seed SEED 
                         --output_dir OUTPUT_DIR [--num_instances NUM_INSTANCES] 
                         [--num_episodes NUM_EPISODES] [--episode_length EPISODE_LENGTH] 
                         [--timeout TIMEOUT] [--deepsynth_cbmc_path DEEPSYNTH_CBMC_PATH] 
                         [--jirp_max_states JIRP_MAX_STATES] [--lrm_max_states LRM_MAX_STATES] 
                         [--use_lava]
```
where 
* `ALGORITHM` is the name of the baseline algorithm (`deepsynth`, `jirp`, or `lrm`),
* `DOMAIN` is the name of the domain (`craftworld` or `waterworld`),
* `TASK` is the name of the domain's task (check mappings [here](src/utils/rl_utils.py)),
* `SEED` is the random seed for the experiment,
* `OUTPUT_DIR` is the directory where results are stored,
* `NUM_INSTANCES` is the number of environment instances used to collect the traces,
* `NUM_EPISODES` is the number of episodes during which traces are collected,
* `EPISODE_LENGTH` is the length of the episodes,
* `TIMEOUT` is the time during which the algorithm can learn reward machines,
* `DEEPSYNTH_CBMC_PATH` is the path to the [CBMC binary](#installation-of-baseline-software) for DeepSynth,
* `JIRP_MAX_STATES` is the maximum number of states that an RM learned by JIRP can have,
* `LRM_MAX_STATES` is the maximum number of states that an RM learned by LRM can have, and
* `--use_lava` creates CraftWorld grids with lava if enabled.

The code creates grids in the open-plan (OP) setting for CraftWorld, and without-deadends (WOD) for WaterWorld. We
provide a couple of pre-generated commands in `config/examples/baselines`:

| Folder                   | Description                                             |
|--------------------------|---------------------------------------------------------|
| `01-cw-op-mb-deepsynth`  | Learn the RM for CraftWorld's MilkBucket using DeepSynth. |
| `02-cw-op-mb-jirp`       | Learn the RM for CraftWorld's MilkBucket using JIRP.    |
| `03-cw-op-mb-lrm`        | Learn the RM for CraftWorld's MilkBucket using LRM.     |
| `04-ww-wod-rg-deepsynth` | Learn the RM for WaterWorld's RG using DeepSynth.       |
| `05-ww-wod-rg-jirp`      | Learn the RM for WaterWorld's RG using JIRP.            |
| `06-ww-wod-rg-lrm`       | Learn the RM for WaterWorld's RG using LRM.             |

For DeepSynth, make sure the path to CBMC points to your actual installation.

### Generating Experiments
To ease the usage of the code, we provide code to generate configuration files automatically. The following table
summarizes the generator to be called for each algorithm:

| Algorithm | Command                                                          |
|-----------|------------------------------------------------------------------|
| Ours      | `python -m config.generators.rl.ihsa.ihsa_hrl_config_generator`  |
| CRM       | `python -m config.generators.rl.ihsa.ihsa_crm_config_generator`  |
| DeepSynth | `python -m config.generators.baseline.baseline_config_generator` |                                          
| JIRP      | `python -m config.generators.baseline.baseline_config_generator` |                                          
| LRM       | `python -m config.generators.baseline.baseline_config_generator` |

We refer the user to the descriptions provided by the `--help` (or `-h`) arguments for a complete description of all the
arguments involved in each generation. For guidance, you can check the experiments generated for the paper, which
invokes these scripts with specific values (see [next section](#reproducibility)).

<a name="root-disclaimer"></a>
> **Important:** The code assumes the virtual environment (i.e. the `anaconda3` folder) and the repository (the `hrm-learning`
folder) are installed in the path specified by the `--root_experiments_path` argument. In the case of DeepSynth, the
CBMC installation must also be done in that folder.

These scripts generate as many folders as runs the user specifies in the arguments. These folders share a common prefix
(`batch`) and the suffix corresponds to the run identifier. Each of these folders (often) contains the following three
files:
* A configuration file `config.json` for the experiments using our algorithm.
* A `run.sh` file that runs the appropriate script (`run_algorithm` or `run_baseline`) with the appropriate arguments
  (the configuration file above for our algorithm, or some specific arguments for the baselines). The script makes sure
  that the `py37` virtual environment and the `libjemalloc` mentioned in the installation are loaded.
* An [HTCondor](https://htcondor.org/) configuration file `condor.config`. Our experiments leveraged this software to
distribute the experiments across the machines in the Department of Computing at Imperial College London automatically.
This file will not be useful to external users, but you can adapt the configuration generator (see [here](src/config/generators/config_generator.py))
to your particular needs if you have local access to HTCondor.

## Reproducibility
To ensure the results reported in the paper are reproducible, we provide scripts for generating
the experiment configuration files (for our algorithm) or the commands (for the RM learning baselines), as well as for
plotting the curves or collecting the metrics.

Timed experiments ran on 3.40GHz Intel® Core™i7-6700 processors. Non-timed experiments also ran on 2.90GHz 
Intel® Core™i7-10700, 4.20GHz Intel®Core™ i7-7700K, and 3.20GHz Intel® Core™ i7-8700 processors.

> **Disclaimer:** The experiments in the paper were run with a version of the code that employed calls to `np.random.choice`
instead of `random.sample` (if sampling a few items without replacement) and `random.choice` (if a single item is sampled). 
Some time after, we noticed the former was much slower than the latter and hence applied the change; however, in turn,
the results from the paper might not be obtained. The modified lines are [[1](src/reinforcement_learning/ihsa_hrl_algorithm.py#L660),
[2](src/reinforcement_learning/ihsa_hrl_algorithm.py#L930), [3](src/reinforcement_learning/learning_algorithm.py#L293),
[4](src/reinforcement_learning/replay.py#L20), [5](src/utils/math_utils.py#L6)]. If you want to revert the changes,
replace `random.sample(X, N)` by `np.random.choice(X, N, replace=False)`, and `random.choice(X)` by `np.random.choice(X)`.

### Generating the Paper Experiments
To generate the experiments described in the paper, you can use the command:
```
$ python -m experiments.generator [-h] --root ROOT --experiments_folder EXPERIMENTS_FOLDER --collection_script_path COLLECTION_SCRIPT_PATH
```
where `ROOT` is the existing directory where `EXPERIMENTS_FOLDER` will be created, and `COLLECTION_SCRIPT_PATH` is the
path to a file where all the [plotting](src/utils/plot_curves.py) and 
[metric collection](src/utils/stats) commands will be written. 
The `ROOT` should be specified as explained [here](#root-disclaimer).

We describe the experiments represented by each of the folders generated within `ROOT/EXPERIMENTS_FOLDER` below. The 
innermost folders represent the 5 different runs for each experiments (`batch_1`, ..., `batch_5`), each containing the
three files mentioned [here](#generating-experiments).

#### `01-default`
Experiments for learning HRMs with the default setting (see paper for details). There are two subfolders:
* `craftworld` - Experiments in the CraftWorld domain. There is a folder for each grid layout considered in the paper: 
`open_plan` (OP), `open_plan_lava` (OPL), `four_rooms` (FR) and `four_rooms_lava` (FRL).
* `waterworld` - Experiments in the WaterWorld domain. There is a folder for each setting considered in the paper: 
`wo_black` (WOD) and `w_black` (WD).

#### `02-flat`
Experiments for learning flat HRMs. Unlike the previous experiments, a single task is considered a time (i.e. there is 
no curriculum). The following subfolders contains domain folders (`craftworld` and `waterworld`), which in turn contain 
specific tasks for which the flat HRM is to be learned:
* `with_goal_base` - Experiments where the first HRM is learned using a set of goal examples. As described in the paper,
a set of goal traces is first collected and, when it reaches a certain size, a subset of the shortest traces is taken 
and used to learn the first HRM.
* `without_goal_base` - Experiments where the first HRM is learned using a single goal example.

There is also a folder for each of the RM learning methods we employ as baselines: `deepsynth`, `jirp` and `lrm`.

#### `03-restricted-hypothesis-space`
An ablation of the `01-default` experiments where the hypothesis space is biased to consider only the HRMs that 
are actually needed to learn a given HRM, e.g. only allow calling Leather and Paper when learning the HRM for Book
instead of also allowing to call Batter, Bucket, Compass, Quill and Sugar.

#### `04-exploration`
An ablation of the `01-default` experiments where other exploration schemes are used to find goal traces. By default
(i.e., `01-default`), our algorithm uses primitive actions, call option policies and formula option policies to explore
the environment; crucially, the last two help explore more efficiently. We consider two additional settings here, one
per folder:
* `acFalse_fTrue_autTrue` - Use formula and call option policies, but not primitive actions. These were not included in
the paper since results did not change much from those observed in `01-default`.
* `acTrue_fFalse_autFalse` - Use primitive actions only.

#### `05-goal-collection`
An ablation of the `01-default` experiments where the learning first HRM is learned using a single goal trace example 
instead of a set of short examples, akin to the experiments in `02-flat/without_goal_base`.

#### `06-handcrafted`
Experiments for exploiting flat and non-flat handcrafted HRMs. Inside each domain folder (`craftworld` or `waterworld`),
each subfolder has the name of the setting (e.g., grid layout, such as `open_plan`) followed by one of the following
suffixes:
* `fFalse` - our algorithm + a non-flat HRM.
* `fTrue` - our algorithm + a flat HRM.
* `crm` - the CRM algorithm [(Toro Icarte et al., 2022)](#references) + a flat HRM.

#### `results`
See [next section](#collecting-the-metrics-and-plotting-curves) for details.

### Collecting the Metrics and Plotting Curves
Along the experiment folders, the generation script in the [previous section](#generating-the-paper-experiments) also
generated a `results` folder in `ROOT/EXPERIMENTS_FOLDER` containing all the configuration files needed for plotting the 
results and collecting the metrics when running the script in `COLLECTION_SCRIPT_PATH`. The resulting plots and metrics
(in JSON format) will be saved in the `results` folder.

For further processing the results, we provide a [summarization script](src/experiments/summarizer.py) which puts all
results together into tables like the ones in the paper (in both spreadsheet and LaTeX formats). You can run it using:
```
$ python -m experiments.summarizer [-h] in_results_path out_path
```
where `in_results_path` is the path to the `results` folder above (i.e., `ROOT/EXPERIMENTS_FOLDER/results`) and
`out_path` is the path where the tables will be exported.

## Citation
If you use this code in your work, please use the following citation:
```
@inproceedings{FurelosBlancoLJBR23,
  author       = {Daniel Furelos-Blanco and
                  Mark Law and
                  Anders Jonsson and
                  Krysia Broda and
                  Alessandra Russo},
  title        = {{Hierarchies of Reward Machines}},
  booktitle    = {Proceedings of the 40th International Conference on Machine Learning (ICML)},
  year         = {2023}
}
```

## References
* Law, M.; Russo, A.; and Broda, K. 2015. [_The ILASP System for Learning Answer Set Programs_](www.ilasp.com).
* Toro Icarte, R.; Klassen, T. Q.; Valenzano, R. A.; and McIlraith, S. A. 2018. [_Using Reward Machines for High-Level Task Specification and Decomposition in Reinforcement Learning_](http://proceedings.mlr.press/v80/icarte18a.html). Proceedings of the 35th International Conference on Machine Learning (ICML). [**Code**](https://github.com/RodrigoToroIcarte/reward_machines).
* Toro Icarte, R.; Waldie, E.; Klassen, T. Q.; Valenzano, R. A.; Castro, M. P.; and McIlraith, S. A. 2019. [_Learning Reward Machines for Partially Observable Reinforcement Learning_](https://papers.nips.cc/paper_files/paper/2019/file/532435c44bec236b471a47a88d63513d-Paper.pdf). Proceedings of the 33rd Advances in Neural Information Processing Systems (NeurIPS) Conference. [**Code**](https://bitbucket.org/RToroIcarte/lrm/).
* Xu, Z.; Gavran, I.; Ahmad, Y.; Majumdar, R.; Neider, D.; Topcu, U.; and Wu, B. 2020. [_Joint Inference of Reward Machines and Policies for Reinforcement Learning_](https://ojs.aaai.org/index.php/ICAPS/article/view/6756/6610). Proceedings of the International Conference on Automated Planning and Scheduling (ICAPS). [**Code**](https://github.com/corazza/stochastic-reward-machines), [**Code**](https://github.com/logic-and-learning/AdvisoRL).*
* Hasanbeig, M.; Jeppu, N. Y.; Abate, A.; Melham, T.; and Kroening, D. 2021. [_DeepSynth: Automata Synthesis for Automatic Task Segmentation in Deep Reinforcement Learning_](https://ojs.aaai.org/index.php/AAAI/article/view/16935/16742). Proceedings of the 35th AAAI Conference on Artificial Intelligence (AAAI). [**Code**](https://github.com/grockious/deepsynth).
* Toro Icarte, R.; Klassen, T. Q.; Valenzano, R. A.; and McIlraith, S. A. 2022. [_Reward Machines: Exploiting Reward Function Structure in Reinforcement Learning_](https://jair.org/index.php/jair/article/view/12440). Journal of Artificial Intelligence Research 73. [**Code**](https://github.com/RodrigoToroIcarte/reward_machines).
* Furelos-Blanco, D.; Law, M.; Jonsson, A.; Broda, K.; and Russo, A. 2023. [_Hierarchies of Reward Machines_](https://arxiv.org/abs/2205.15752). Proceedings of the 40th International Conference on Machine Learning (ICML).

<sup>* Both codebases were referred to us by the authors. We employed the code from the first in our experiments.</sup>

