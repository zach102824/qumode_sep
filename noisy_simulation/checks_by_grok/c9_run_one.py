import sys, runpy, os
os.chdir("/workspace/noisy_checks"); sys.path.insert(0, "/workspace/noisy_checks")
import mutations; mutations.apply(sys.argv[1])
runpy.run_path(sys.argv[2], run_name="__main__")
