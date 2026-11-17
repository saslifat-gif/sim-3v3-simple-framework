"""Use the supported simulator speed command and measure real throughput.

Run inside the simulator's Python environment. No physics or referee tuning.
"""
import argparse
import json
import logging
import sys
import time
sys.path.insert(0, '/usr/local/booster_robot/booster_robocup_sim')
from core.physics_transport import PhysicsCommand, PhysicsTransportConfig
from transport.websocket.physics_ws_client import PhysicsWebSocketClient

parser=argparse.ArgumentParser()
parser.add_argument('--speed',type=float,default=10)
parser.add_argument('--seconds',type=float,default=12)
parser.add_argument('--restore',action='store_true')
args=parser.parse_args()
if not 0<args.speed<=10 or args.seconds<1:
    raise SystemExit('Speed must be in (0,10]; measurement must last at least one second')
logging.basicConfig(level=logging.ERROR)
client=PhysicsWebSocketClient(PhysicsTransportConfig(),owner='rl-speed-benchmark')
client.subscribe_physics_state(lambda state:None)
client.subscribe_simulation_state(lambda state:None)
client.connect()
try:
    deadline=time.monotonic()+20
    while client.get_latest_physics_state() is None or client.get_latest_simulation_state() is None:
        if time.monotonic()>deadline:raise RuntimeError('Missing simulator state')
        time.sleep(.1)
    original=client.get_latest_simulation_state().target_speed_factor or 1.0
    client.send_command(PhysicsCommand.SET_SPEED,{'speed_factor':args.speed})
    time.sleep(3)
    control=client.get_latest_simulation_state()
    if control.paused:raise RuntimeError('Simulator paused; throughput is not measurable')
    start_sim=client.get_latest_physics_state().time
    start_wall=time.monotonic()
    time.sleep(args.seconds)
    stop_sim=client.get_latest_physics_state().time
    elapsed=time.monotonic()-start_wall
    state=client.get_latest_simulation_state()
    print('SPEED_RESULT '+json.dumps(dict(requested=args.speed,reported_target=state.target_speed_factor,
        reported_actual=state.actual_speed_factor,measured_sim_seconds_per_wall_second=(stop_sim-start_sim)/elapsed,
        wall_seconds=elapsed,sim_seconds=stop_sim-start_sim,profile=state.runtime_profile)),flush=True)
finally:
    if args.restore:
        client.send_command(PhysicsCommand.SET_SPEED,{'speed_factor':locals().get('original',1.0)})
        time.sleep(1)
        current=client.get_latest_simulation_state()
        print('RESTORED_SPEED '+json.dumps(dict(target=current.target_speed_factor if current else None)),flush=True)
    client.disconnect()
