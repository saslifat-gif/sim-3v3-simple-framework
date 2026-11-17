"""Observe a live ROS match without issuing robot commands. Run inside the simulator."""
import argparse
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.framework.config import SoccerConfig
from src.framework.ros_source import RosContextSource
from src.framework.types import ADULT_FIELD_DIMENSIONS, Context, GameState, Penalty
from src.ml_policy import state_from_context
from src.param import SHOT_TARGET_COUNT, SHOT_POST_MARGIN_M
from src.utils.tactics import choose_shot_target


parser = argparse.ArgumentParser()
parser.add_argument('--seconds', type=int, default=660)
parser.add_argument('--output', required=True)
parser.add_argument('--episode', required=True)
args = parser.parse_args()
source = RosContextSource(SoccerConfig())
source.start()
count = 0
next_report = 0
started = time.monotonic()
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
try:
    with output.open('w') as file:
        while time.monotonic() - started < args.seconds:
            now = time.monotonic()
            snap = source.get_snapshot()
            robots = list(snap.teammates.values()) + list(snap.opponents.values())
            if now-started >= next_report:
                print(f'Recorder: elapsed={now-started:.0f}s state={snap.game.state.value if snap.game else None} visible={sum(r.pose is not None for r in robots)}/6 ball={snap.ball is not None} examples={count}', flush=True)
                next_report += 10
            fresh = (snap.ball is not None and now - snap.ball.last_seen_at < 0.5
                     and len(robots) == 6 and all(r.pose is not None and now-r.last_seen_at < 0.5 for r in robots))
            if fresh and snap.game is not None and snap.game.state == GameState.PLAYING and not snap.game.stopped:
                ctx = Context(now, 0.2, 1, ADULT_FIELD_DIMENSIONS, snap.game, snap.ball, snap.teammates, snap.opponents)
                available = [r for r in snap.teammates.values()
                             if snap.game.get_player_state(1, r.player_id).penalty == Penalty.NONE]
                if available:
                    shooter = min(available, key=lambda r: math.hypot(r.pose.x-snap.ball.x, r.pose.y-snap.ball.y))
                    if snap.ball.x < ctx.field.length / 2 - 0.05:
                        target = choose_shot_target(ctx, shooter.player_id)
                        bx, by = snap.ball.x, snap.ball.y
                        crossing = by + (target[1]-by) * (ctx.field.length/2-bx) / (target[0]-bx)
                        half = ctx.field.goal_width/2-SHOT_POST_MARGIN_M
                        label = round((crossing/half+1)*(SHOT_TARGET_COUNT-1)/2)
                        record = dict(episode=args.episode, elapsed=now-started, state=state_from_context(ctx, shooter.player_id), label=label)
                        file.write(json.dumps(record)+'\n')
                        count += 1
                        if count % 100 == 0:
                            file.flush()
                            print(f'Collected {count} live teacher examples', flush=True)
            if snap.game is not None and snap.game.state == GameState.FINISHED and count > 100:
                break
            time.sleep(0.2)
finally:
    source.stop()
print(f'Finished: {count} samples -> {output}', flush=True)
if count < 10:
    raise SystemExit('Insufficient live observations; check ROS connectivity')
