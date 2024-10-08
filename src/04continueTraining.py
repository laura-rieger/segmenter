import itertools
import os

from simple_slurm import Slurm
import platform

is_windows = platform.system() == "Windows"
params_to_vary = {
    "progress_folder": ['0007565832'],
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
        time="0-00:20:00",
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
