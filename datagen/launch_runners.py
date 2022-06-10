import sys
sys.path.append("/media/beans/ssd/bespoke")
from constants import *
import subprocess

clear_obs_per_sec()
set_should_stop(False)
for i in range(N_RUNNERS):
    datagen_id = ("00"+str(i))[-2:]
    subprocess.Popen(f"bash /media/beans/ssd/bespoke/datagen/launch_runner.sh {datagen_id}", shell=True)


# sudo pkill -f blenders_for_dataloader
