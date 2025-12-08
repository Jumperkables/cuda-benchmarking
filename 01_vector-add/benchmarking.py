import os
import subprocess

block_sizes = [32, 64, 128, 256, 512, 1024]

for bs in block_sizes:
    print(f"\n======= Block Size: {bs} =======")
    cmd = ["./vector_add"]
    env = os.environ.copy()
    env["BLOCK_SIZE"] = str(bs)
    subprocess.run(cmd, env=env)