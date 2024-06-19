
from pathlib import Path

from perqueue import PersistentQueue, Task, Workflow, CyclicalGroup

here = Path(__file__).parent

t1 = Task(here / "train.py", {'poolname': 'pq_lno', 'add_ratio':.02}, "1:sm3090_devel:2h")
t2 = Task(here / "predict_select.py", {}, "1:sm3090_devel:30m")
t3 = Task(here / "human.py", {}, "1:sm3090_devel:10m")

cg = CyclicalGroup([t1,t2, t3], max_tries=10)
wf = Workflow({t1: [], cg: [t1]})
with PersistentQueue() as PQ:
    PQ.submit(wf)
