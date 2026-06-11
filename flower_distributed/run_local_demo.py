"""Launch the full networked Flower deployment on localhost (server + 16 clients).

Spawns one ``server.py`` and 16 ``client.py`` processes that talk over real
gRPC, runs FedAvg for a few rounds, and prints the server log. Proves the
distributed Flower stack runs end to end.

    python run_local_demo.py --rounds 5 --penalty l1 --C 0.1
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--penalty", default="l1")
    ap.add_argument("--C", type=float, default=0.1)
    ap.add_argument("--l1_ratio", type=float, default=0.5)
    a = ap.parse_args()

    # 1) build the shared global scaler (federated summary statistics)
    subprocess.run([PY, os.path.join(HERE, "prepare_scaler.py")], check=True)

    # 2) start the Flower server
    server = subprocess.Popen(
        [PY, "-u", os.path.join(HERE, "server.py"), "--rounds", str(a.rounds),
         "--address", "127.0.0.1:8080"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    time.sleep(4)  # let the server bind

    # 3) start the 16 regional clients
    clients = []
    for r in range(1, 17):
        clients.append(subprocess.Popen(
            [PY, os.path.join(HERE, "client.py"), "--region", str(r),
             "--server", "127.0.0.1:8080", "--penalty", a.penalty,
             "--C", str(a.C), "--l1_ratio", str(a.l1_ratio), "--local_iters", "5"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))

    # 4) stream the server log
    out, _ = server.communicate()
    for c in clients:
        c.wait()
    print(out)


if __name__ == "__main__":
    main()
