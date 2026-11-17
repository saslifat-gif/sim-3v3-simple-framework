Simplified Version of Booster Champion 3v3 Soccer Tournament agent

This strategy framework has been greatly simplified to make it easier for players to implement, but it still needs optimization to achieve the best results. Players should try it out thoroughly in a simulation environment and only use it in the competition after successful debugging.

How to use：
1. Download Booster Studio
2. Open this project from Booster Studio
3. Waiting for environment prepare
4. Select a Virtual Robot
5. Just click "Activate, build, deploy and run agent"

more info:
https://booster.feishu.cn/wiki/RwYkw3004iV39hk6uJOcu2ctnah

Current formation (introduced in v1.3.2):
- All three robots play outfield. The closest active player attacks; the other two support in separate wide lanes and can press the ball.
- Ready and kickoff positions place all three in midfield. Opponent-restart retreat rules still apply to every player.
- The saved v1.3.1 package and its 5–0 / 2–1 results used the earlier goalkeeper formation; those results do not measure this new formation.

Local imitation model (v1.2.0):
- `src/models/shot_policy.py` is the trained, portable shooting-lane policy.
- `src/ml_policy.py` runs it with standard Python only. Roles, movement and restart rules remain coded.
- Set `USE_LEARNED_SHOOTING = False` in `src/param.py` to use the original shot planner.
- `training/collect_teacher.py` passively records live ROS observations and labels them with the v1.1.2 shot planner; run it in the simulator's ROS environment.
- Install `training/requirements.txt` into a separate training environment, then run `training/train_imitation.py training/data/teacher-match-01.jsonl` to reproduce training and export.
- `training/artifacts/training-report.json` records chronological held-out agreement, augmentation counts and export parity. These metrics measure imitation, not playing strength.

Match-reward training:
- `training/match_search.py` performs bounded evolution-strategy policy search against the unchanged `com.booster.default3v3ai` opponent.
- It adjusts legal kick power, goalkeeper depth, support positioning, support pressing and shooting-lane weights. It retains the imitation policy as a lane-selection feature.
- Every reward comes from a full 600-second match after a verified physics reset. The reward is goal difference plus 0.03 per goal scored, minus 0.1 per own penalty.
- A candidate must win a complete match by at least five goals to reach the requested target. It then plays a new complete match with the same package to assess repeatability; the replay score is reported separately.
- Checkpoints, signed packages, match samples and event logs are saved under `build/match-training/`. `search.json` records measured results; generated `src/models/match_policy.py` contains the current candidate, which can be unproven during a search.
- To stop between polling checks, create `build/match-training/stop-requested`. Remove that file before intentionally restarting training. Restarting `python training/match_search.py` preserves completed results.
- The Windows training runner uses the installed Booster Studio SDK and its existing local development container. It does not require an external AI API.

Reinforcement-learning pilot:
- Corrected training uses a 1x speed default to match wall-clock robot controls. It keeps action records open through READY/SET and observation gaps, explicitly accounts for pre-action rewards, and trains the terminal tail. Completed episodes must reconcile their observed environment rewards with the referee's goals and penalties. New runs use `--run-directory`, `--initial-model`, `--training-speed` and `--tensorboard-directory`; `build/rl-corrected` starts from the earlier evaluated v1.4.5 weights, with its value estimator reset for the corrected returns.
- Training now defaults to `--update-seconds 20`: on-policy actor-critic updates run inside the bot approximately every 20 wall-clock seconds during a match, using a bootstrapped value estimate at chunk boundaries. At least four fresh transitions are required; pauses or missing observations can delay an update. Evaluation remains frozen. TensorBoard's `online/` charts show actual update intervals, rewards, KL and weight changes; individual checkpoints are saved under `build/rl-training/online-checkpoint-*.json`.
- Continue from saved weights with `.venv-training/Scripts/python.exe training/rl_train.py --additional-episodes 12`. Episode packages and evaluation names are versioned so the original pilot remains available.
- TensorBoard: `.venv-training/Scripts/python.exe -m tensorboard.main --logdir build/rl-training/tensorboard --host 127.0.0.1 --port 6006 --reload_interval 5`, then open `http://127.0.0.1:6006/#scalars`. Charts include episode rewards, scores, penalties, action counts, gradient size and actual simulator speed. Live charts update during a match; episode metrics update after validation.
- Run `.venv-training/Scripts/python.exe training/rl_train.py` on Windows with Docker and Booster Studio installed. The pilot runs four full training matches, then one evaluation match at normal speed.
- `src/rl_policy.py` uses a centralized softmax policy to choose team tactics for all three outfield players every two game-clock seconds. It adjusts pressing, width, support distance and kick power; walking primitives and restart legality remain coded.
- Training samples actions and records transitions. The host computes discounted rewards-to-go, subtracts a learned linear value baseline, and updates the policy weights with a bounded policy-gradient step. See [OpenAI Spinning Up's policy-gradient reference](https://spinningup.openai.com/en/latest/algorithms/vpg.html).
- Rewards are +10 per goal scored, -10 per goal conceded, -0.5 per new own penalty, and a small ball-progress shaping term. Every rollout is checked against the exact checkpoint probabilities before updating.
- The simulator receives a 10x speed target through its supported API. Reports distinguish that target from measured throughput; hardware may sustain substantially less. The pilot restores a 1x target for evaluation and on exit.
- Models, rollouts, update statistics, event logs and the normal-speed score are saved under `build/rl-training/`. Create `build/rl-training/stop-requested` to stop the pilot at a polling check.
- Production inference uses `training=False` and standard Python only. A successful training run does not by itself establish that the policy beats the baseline.
- Completed pilot: v1.4.5 trained from 843 transitions in four valid matches, then won 5–2 against the default AI in a full 1x evaluation with no own penalties. The 10x target delivered about 3.36x game-clock throughput during training, excluding builds. Accelerated rollouts had extra `motionInSet` penalties; their wall-clock control timing differs from 1x play. This single evaluation does not establish consistent superiority. Evidence is in `training/artifacts/rl-pilot-summary.json`.


GitHub source snapshot:
- Based on BoosterRobotics/sim-3v3-simple-framework; the original MIT license is retained.
- Source, tests, framework documentation, training scripts, and a saved online checkpoint are included.
- `training/checkpoints/latest-model.json` records update 169 at export. The source policy may reflect the start of the current match; the JSON checkpoint contains the more recent online weights.
- Build packages, virtual environments, raw rollouts, TensorBoard logs, local device settings, and the machine handoff note are excluded.
- Training scripts currently contain this installation's Booster Studio paths, robot/container IDs, and references to older built packages. Adapt those settings and build the required packages before running them on another machine.
- `training/artifacts/` and historical `build/` results referenced above are local experiment outputs and are not bundled in this source snapshot.
