import itertools
from simple_slurm import Slurm
import platform

is_windows = platform.system() == "Windows"

slurm = Slurm(
    mail_type="FAIL",
    partition="xeon40el8",
    N=1,
    n=40,
    time="0-02:35:00",
    mem="10G",
)

cur_function = "python Weka_complete.py "

slurm.sbatch(cur_function)
