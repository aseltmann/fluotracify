import os
import pprint
import subprocess

ramlist = os.popen("free -th").readlines()[-1].split()[1:]

print("No of CPUs in system:", os.cpu_count())
print("No of CPUs the current process can use:", len(os.sched_getaffinity(0)))
print("load average:", os.getloadavg())
print("os.uname(): ", os.uname())
print("PID of process:", os.getpid())
print(f"RAM total: {ramlist[0]}, RAM used: {ramlist[1]}, RAM free: {ramlist[2]}")
print("The current directory:", os.getcwd())
print("My disk usage:", subprocess.run(["df", "-h"]))
print("My conda environment:", subprocess.run(["conda", "list"]))
pprint.pprint(dict(os.environ), sort_dicts=False)
