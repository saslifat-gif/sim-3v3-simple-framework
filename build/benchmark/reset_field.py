"""Reset through the simulator's own transport API; verify the clock reset."""
import json
import sys
import time

sys.path.insert(0, '/usr/local/booster_robot/booster_robocup_sim')
from core.physics_transport import PhysicsCommand, PhysicsTransportConfig
from transport.websocket.physics_ws_client import PhysicsWebSocketClient

client = PhysicsWebSocketClient(PhysicsTransportConfig(), owner='policy-benchmark')
client.subscribe_physics_state(lambda state: None)
client.connect()
try:
    deadline = time.monotonic()+20
    while client.get_latest_physics_state() is None and time.monotonic() < deadline:
        time.sleep(.1)
    before = client.get_latest_physics_state()
    if before is None:
        raise RuntimeError('No physics state; refusing an unverified reset')
    before_time = before.time
    client.send_command(PhysicsCommand.RESET)
    deadline = time.monotonic()+40
    while time.monotonic() < deadline:
        after = client.get_latest_physics_state()
        if after is not None and after.time < 2 and after.time < before_time:
            print('RESET_RESULT '+json.dumps(dict(before=before_time, after=after.time)), flush=True)
            break
        time.sleep(.1)
    else:
        raise RuntimeError('Physics reset was not observed')
finally:
    client.disconnect()
