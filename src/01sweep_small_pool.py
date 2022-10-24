import itertools
import os

from simple_slurm import Slurm
import platform

is_windows = platform.system() == "Windows"
params_to_vary = {
    "experiment_name": [
        "LNOSweepLong3",
    ],
    "learningrate": [0.01],
    "seed": [x for x in range(1, 4)],
    "cost_function": ["Mean", "Random"],
    "add_ratio": [0, 0.05, 0.1, 0.3],
    "batch-size": [128],
    "scale": [
        0.5,
    ],
    "foldername": [
        "lno_halfHour",
    ],
    "epochs": [
        50,
    ],
    "image-size": [
        128,
    ],
    "add_step": [
        2,
    ],
    "offset": [
        128,
    ],
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
        time="0-00:15:00",
        mem="10G",
        gres="gpu:RTX3090:1",
    )

    cur_function = "python train.py "

    for j, key in enumerate(keys):

        cur_function += "--" + key + " " + str(param_combinations[i][j]) + " "

    if is_windows:
        print(cur_function)
        # subprocess.call(cur_function, shell=True)

    else:
        slurm.sbatch(cur_function)
