from pathlib import Path

from perqueue import PersistentQueue
from perqueue.task_classes.task import Task
from perqueue.task_classes.task_groups import CyclicalGroup, Workflow, DynamicWidthGroup, StaticWidthGroup
from perqueue.constants import DYNAMICWIDTHGROUP_KEY


here = Path(__file__).parent
t1 = Task(here / "train.py", {'poolname': 'pq_lno', 'add_ratio':.02, 'needed_accuracy': 0.95}, "1:sm3090_devel:4h")
t2 = Task(here / "human.py", {}, "1:sm3090_devel:10m")

cg = CyclicalGroup([t2, t1], max_tries=10)
wf = Workflow({t1: [], cg: [t1]})

with PersistentQueue() as pq:
    pq.submit(wf)



