import itertools
from simple_slurm import Slurm
import platform

is_windows = platform.system() == "Windows"
params_to_vary = {
    "experiment_name": [
        "BActiveLearning",
    ],

    "seed": [x for x in range(5)],
    "cost_function": [ 'cut_off_cost','random_cost'],
    "add_ratio": [ .02,],
    'poolname' : ['lno'],
    "batch-size": [128,],

    "learningrate": [ 0.0001,],
    "add_step": [ 1, ],
    "add_size": [ 4, ], 
    "foldername": [ "lno_halfHour", ],

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
        time="0-02:35:00",
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
