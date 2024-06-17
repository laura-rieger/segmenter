
import itertools
import os

from simple_slurm import Slurm
import platform

is_windows = platform.system() == "Windows"
params_to_vary = {
    "experiment_name": [
        "LearningRate",
    ],
    "learningrate": [  0.1,  0.01,  0.001,  0.0001,  0.00001,  0.000001, 1 ],
    "seed": [x for x in range(3)],
    "cost_function": ["random_cost",],
    "add_ratio": [.0, ],
    "batch-size": [128,],
    "foldername": [ "lno", ],
    "poolname": [ "lno", ], 
    "epochs": [ 300, ],
    "image-size": [ 128, ],
    "offset": [ 64, ],
}

keys = sorted(params_to_vary.keys())
vals = [params_to_vary[k] for k in keys]

param_combinations = list(itertools.product(*vals))  # list of tuples
print(len(param_combinations))
for i in range(len(param_combinations)):
    slurm = Slurm(
        mail_type="FAIL",
        partition="sm3090el8",
        N=1,
        n=8,
        time="0-01:15:00",
        mem="10G",
        gres="gpu:RTX3090:1",
    )

    cur_function = "python train_nested.py "

    for j, key in enumerate(keys):

        cur_function += "--" + key + " " + str(param_combinations[i][j]) + " "

    if is_windows:
        print(cur_function)
        # subprocess.call(cur_function, shell=True)

    else:
        slurm.sbatch(cur_function)
