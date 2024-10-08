import itertools
from simple_slurm import Slurm
import platform
import subprocess

is_windows = platform.system() == "Windows"
params_to_vary = {
    "experiment_name": [ "DebugMode", ],
    "learningrate": [0.001],
    "seed": [x for x in range(1)],
    "cost_function": [ "cut_off_cost", ], 
    "add_ratio": [ 0.02, ], # what proportion of the pool is added to the training set
    'poolname' : ['lno_dummy_full'], 
    "batch-size": [32],
    "add_size": [4], # how many images are added each step
    "add_step": [ 1, ], 
    "foldername": [ "DataLNO", ],
    "epochs": [100],
    "image-size": [ 200, ],

    "offset": [ 128, ],
}

keys = sorted(params_to_vary.keys())

vals = [params_to_vary[k] for k in keys]

param_combinations = list(itertools.product(*vals))  # list of tuples
print(len(param_combinations))
for i in range(len(param_combinations)):
    slurm = Slurm(
        mail_type="FAIL",
        partition="sm3090",
        N=1,
        n=8,
        time="0-03:35:00",
        mem="10G",
        gres="gpu:RTX3090:1",
    )

    cur_function = "python train_nested.py "

    for j, key in enumerate(keys):

        cur_function += "--" + key + " " + str(param_combinations[i][j]) + " "

    if is_windows:
        # print(cur_function)
        subprocess.call(cur_function, shell=True)

    else:
        slurm.sbatch(cur_function)
