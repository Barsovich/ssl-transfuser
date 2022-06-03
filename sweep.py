import time
import datetime
import logging
from pathlib import Path
import os
import subprocess
import pandas as pd


logging.basicConfig(level=logging.INFO)

SUBMIT = True

SLURM_DIR = "./slurm/"

# SBATCH --output=/home/gsimmons/logs/transfuser_%j_next_frame_{next_frame_coef}_cross_modal_{cross_modal_coef}.out
# SBATCH --error=/home/gsimmons/logs/transfuser_%j_next_frame_{next_frame_coef}_cross_modal_{cross_modal_coef}.err
SLURM_SCRIPT = """#!/bin/bash

#SBATCH --job-name=transfuser_train
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gsimmons@ucdavis.edu
#SBATCH --output=/home/gsimmons/logs/transfuser.out
#SBATCH --error=/home/gsimmons/logs/transfuser.err
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=2:00:00

source /opt/anaconda3/etc/profile.d/conda.sh && \\
cd /home/gsimmons/ssl-transfuser/ && \\
pwd && \\
conda activate transfuser && \\
python ssl_transfuser/train.py \\
  --epochs 100 \\
  --val_every 50 \\
  --next_frame_prediction_loss_coef {next_frame_coef} \\
  --cross_modal_prediction_loss_coef {cross_modal_coef} \\
  --sweep_id {sweep_id} \\
  --max-steps {max_steps} \\
  --logdir /home/gsimmons/ssl-transfuser/logs/transfuser_next_frame_{next_frame_coef}_cross_modal_{cross_modal_coef} \\
"""

MAX_STEPS = 1800


def get_dependency_jobid(max_concurrent_jobs):
    output = subprocess.run(["squeue"], capture_output=True)

    with open("tmp.ssv", "w") as f:
        f.write(output.stdout.decode("utf-8"))
    sq = pd.read_csv("tmp.ssv", sep="\t")
    sq = sq.iloc[:, 0].str.split(r"\s+", expand=True)
    sq.columns = [
        "index",
        "jobid",
        "partition",
        "jobname",
        "user",
        "state",
        "time",
        "command",
        "reason",
    ]

    if len(sq[sq["user"] == "gsimmons"]) >= max_concurrent_jobs:
        max_jobid = sq[sq["user"] == "gsimmons"]["jobid"].astype(int).max()
        jobid = max_jobid - (max_concurrent_jobs - 1)
    else:
        jobid = None
    return jobid


if __name__ == "__main__":
    NEXT_FRAME_COEF_VALUES = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
    CROSS_MODAL_COEF_VALUES = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
    # NEXT_FRAME_COEF_VALUES = [0.0, 0.1, 0.5, 1.0, 5.0, 10.0]
    # CROSS_MODAL_COEF_VALUES = [0.0, 0.1, 0.5, 1.0, 5.0, 10.0]

    sweep_id = "'" + str(datetime.datetime.now()) + "'"

    for next_frame_coef in NEXT_FRAME_COEF_VALUES:
        for cross_modal_coef in CROSS_MODAL_COEF_VALUES:
            script = SLURM_SCRIPT.format(
                next_frame_coef=next_frame_coef,
                cross_modal_coef=cross_modal_coef,
                sweep_id=sweep_id,
                max_steps=MAX_STEPS,
            )
            script_path = (
                Path(SLURM_DIR)
                / "model_training"
                / f"transfuser_train_next_frame_{next_frame_coef}_cross_modal_{cross_modal_coef}.slurm"
            )
            with open(script_path, "w") as f:
                logging.info(f"Writing script to {script_path}")
                f.write(script)
            if SUBMIT:
                time.sleep(1)
                dependency_jobid = get_dependency_jobid(max_concurrent_jobs=3)
                if dependency_jobid:
                    cmd = [
                        "sbatch",
                        f"--dependency=afterany:{dependency_jobid}",
                        str(script_path.absolute()),
                    ]
                else:
                    cmd = ["sbatch", str(script_path.absolute())]

                logging.info(f"Running {' '.join(cmd)}")
                subprocess.run(cmd)
