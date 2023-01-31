import itertools
from simple_slurm import Slurm
import platform

is_windows = platform.system() == "Windows"
params_to_vary = {
    "experiment_name": [
        "12largeSamples",
    ],
    "learningrate": [0.001],
    "seed": [x for x in range(3)],
    "cost_function": [
         "uncertainty_cost",   "random_cost"
    ], 
    "add_ratio": [
           0.25,
    ],
    'poolname' : ['lno'],
    "batch-size": [32],
    "scale": [
        0.5,
    ],
    "foldername": [
        "lno_halfHour",
    ],
    "epochs": [250],
    "image-size": [
        256,
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
        time="0-00:35:00",
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
