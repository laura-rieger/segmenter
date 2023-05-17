import itertools
from simple_slurm import Slurm
import platform

is_windows = platform.system() == "Windows"
params_to_vary = {
    "experiment_name": [
        "WriteOut",
    ],
    "learningrate": [0.0005],
    "seed": [x for x in range(1)],
    "cost_function": [
         "uncertainty_cost",  
    ], 
    "add_ratio": [
           0.02,
    ],
    'poolname' : ['lno'],
    "batch-size": [128],
    "add_size": [4],
    "add_step": [
        3,
    ],
    "foldername": [
        "lno_halfHour",
    ],
    "epochs": [100],
    "image-size": [
        128,
    ],

    "offset": [
        64,
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
        time="0-03:35:00",
        mem="10G",
        gres="gpu:RTX3090:1",
    )

    cur_function = "python train_writeout.py "

    for j, key in enumerate(keys):

        cur_function += "--" + key + " " + str(param_combinations[i][j]) + " "

    if is_windows:
        print(cur_function)
        # subprocess.call(cur_function, shell=True)

    else:
        slurm.sbatch(cur_function)
