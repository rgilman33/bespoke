import sys
sys.path.append("/home/beans/bespoke")
from constants import *
import subprocess

clear_obs_per_sec()
set_should_stop(False)
for i in range(N_RUNNERS):
    
    datagen_id = ("00"+str(i))[-2:]    
    dataloader_root = f"{BLENDER_MEMBANK_ROOT}/dataloader_{datagen_id}"
    if not os.path.exists(dataloader_root): os.makedirs(dataloader_root)

    # Capture stdout for each process independently for debugging
    stdout_f = f"{dataloader_root}/stdout.txt"
    if os.path.exists(stdout_f): os.remove(stdout_f)
    stdout = open(stdout_f, 'w')

    stderr_f = f"{dataloader_root}/stderr.txt"
    if os.path.exists(stderr_f): os.remove(stderr_f)
    stderr = open(stderr_f, 'w')

    subprocess.Popen(f"bash /home/beans/bespoke/datagen/launch_runner.sh {datagen_id}", shell=True, stdout=stdout, stderr=stderr)

time.sleep(10)

# Subprocesses are running, now monitor them. When they're all finished, exit script
import re
while True:
    r = subprocess.check_output("ps aux | grep -i blenders_for_dataloader", shell=True)
    # ps = [m.start() for m in re.finditer('/home/beans/blenders_for_dataloader/tmp/fustbol3_\d\d', str(r))]
    ps = [int(p[-2:]) for p in re.findall(r"/home/beans/blenders_for_dataloader/tmp/fustbol3_\d\d", str(r))]
    num_processes_running = len(ps) # manual, hacky
    print(f"There are {num_processes_running} bpy runners still running.")
    if num_processes_running==0: break
    if num_processes_running < N_RUNNERS:
        missing_runners = [i for i in range(N_RUNNERS) if i not in ps]
        print(f"Missing runners: {missing_runners}")
    time.sleep(10)
    
print("All processes finished!")
# # sudo pkill -f blenders_for_dataloader
