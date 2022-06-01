from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

SLURM_DIR = "./slurm/"
SLURM_SCRIPT = """#!/bin/bash

export CARLA_ROOT=carla
export CARLA_SERVER=${{CARLA_ROOT}}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${{CARLA_ROOT}}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${{CARLA_ROOT}}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:leaderboard
export PYTHONPATH=$PYTHONPATH:leaderboard/team_code
export PYTHONPATH=$PYTHONPATH:scenario_runner

export LEADERBOARD_ROOT=leaderboard
export CHALLENGE_TRACK_CODENAME=SENSORS
export PORT={port} # same as the carla server port
export TM_PORT={tm_port} # port for traffic manager, required when spawning multiple servers/clients
export DEBUG_CHALLENGE=0
export REPETITIONS=1 # multiple evaluation runs
export ROUTES=leaderboard/data/{routes_dir}/routes_town{town}_{route_variant}.xml
export TEAM_AGENT=leaderboard/team_code/auto_pilot.py # agent
export TEAM_CONFIG=aim/log/aim_ckpt # model checkpoint, not required for expert
export CHECKPOINT_ENDPOINT=results/results_town_{town}_{route_variant}.json # results file
export SCENARIOS=leaderboard/data/scenarios/town{town}_all_scenarios.json
export SAVE_PATH=data/Town{town} # path for saving episodes while evaluating
export RESUME=True


python3 ${{LEADERBOARD_ROOT}}/leaderboard/leaderboard_evaluator.py \\
--scenarios=${{SCENARIOS}}  \\
--routes=${{ROUTES}} \\
--repetitions=${{REPETITIONS}} \\
--track=${{CHALLENGE_TRACK_CODENAME}} \\
--checkpoint=${{CHECKPOINT_ENDPOINT}} \\
--agent=${{TEAM_AGENT}} \\
--agent-config=${{TEAM_CONFIG}} \\
--debug=${{DEBUG_CHALLENGE}} \\
--record=${{RECORD_PATH}} \\
--resume=${{RESUME}} \\
--port=${{PORT}} \\
--trafficManagerPort=${{TM_PORT}}
"""

def write_data_generation(port, tm_port, town, route_variant):
    outpath = (
            Path(SLURM_DIR) / "data_generation" / f"town{town}_{route_variant}.slurm"
        )
    outpath.parent.mkdir(parents=True, exist_ok=True)
    if town == "05":
        routes_dir = "validation_routes"
    else:
        routes_dir = "training_routes"
    script = SLURM_SCRIPT.format(
            port=port,
            tm_port=tm_port,
            town=town,
            routes_dir=routes_dir,
            route_variant=route_variant,
        )
    logging.info(f"Writing {outpath}")
    with open(outpath, "w") as f:
        f.write(script)



def write_carla_launcher(port):
    outpath = (
                Path(SLURM_DIR) / "launch_carla" / f"town{town}_{route_variant}.slurm"
            )
    outpath.parent.mkdir(parents=True, exist_ok=True)
    carla_launcher = f"SDL_VIDEODRIVER=offscreen SDL_HINT_CUDA_DEVICE=0 /home/gsimmons/ssl-transfuser/carla/CarlaUE4.sh --world-port={port} -opengl"
    logging.info(f"Writing {outpath}")
    with open(outpath, "w") as f:
        f.write(carla_launcher)

base_port = 2000
base_tm_port = 5000
for i, town in enumerate(["01", "02", "03", "04", "05", "06", "07", "10"]):
    for j, route_variant in enumerate(["short", "tiny"]):
        port = base_port + 100 * (2 * i + j)
        tm_port = base_tm_port + 100 * (2 * i + j)
        write_data_generation(port, tm_port, town, route_variant)
        write_carla_launcher(port)
        