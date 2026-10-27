"""Compare full/state-only viewer sync on the G1 scene, without DDS or a policy.

Runs full/state-only/state-only/full at 200 Hz physics and 50 Hz viewer updates.
Checks that the physics trajectories match and the viewer receives current state.
Do not interact with the viewer during the comparison.
"""

import argparse
import json
from pathlib import Path
import threading
import time

import mujoco
import numpy as np

from gear_sonic.utils.mujoco_sim.base_sim import DefaultEnv
from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig


def stats(samples):
    return dict(zip(("p50", "p95", "max"), np.percentile(samples, [50, 95, 100]).tolist()))


def benchmark(seconds):
    config = SimLoopConfig(interface="lo").load_wbc_yaml()
    threads_before = set(threading.enumerate())
    env = DefaultEnv(config, onscreen=True)
    results = []
    trajectories = []
    try:
        model, data = env.mj_model, env.mj_data
        # Stabilize contacts once; each comparison starts from exactly this state.
        for _ in range(100):
            mujoco.mj_step(model, data)
        state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        initial = np.empty(mujoco.mj_stateSize(model, state_spec))
        mujoco.mj_getState(model, data, initial, state_spec)
        steps = max(4, round(seconds / env.sim_dt))
        viewer_every = max(1, round(config["VIEWER_DT"] / env.sim_dt))
        for state_only in (False, True, True, False):
            mujoco.mj_resetData(model, data)
            mujoco.mj_setState(model, data, initial, state_spec)
            mujoco.mj_forward(model, data)
            env.viewer_state_only = state_only
            env._viewer_initial_sync = True
            env.update_viewer()
            sync_ms, work_ms, trajectory = [], [], []
            viewer_error = 0.0
            started = next_step = time.monotonic()
            for index in range(steps):
                if not env.viewer.is_running():
                    raise RuntimeError("Viewer closed before the comparison completed")
                step_start = time.perf_counter_ns()
                mujoco.mj_step(model, data)
                if index % viewer_every == 0:
                    sync_start = time.perf_counter_ns()
                    env.update_viewer()
                    sync_ms.append((time.perf_counter_ns() - sync_start) / 1e6)
                work_ms.append((time.perf_counter_ns() - step_start) / 1e6)
                if index % viewer_every == 0:
                    with env.viewer.lock():
                        viewer_error = max(viewer_error, float(np.max(np.abs(
                            env.viewer.d.qpos - data.qpos
                        ))))
                trajectory.append(np.concatenate((data.qpos, data.qvel)))
                now = time.monotonic()
                next_step = max(next_step + env.sim_dt, now - env.sim_dt)
                if next_step > now:
                    time.sleep(next_step - now)
            wall_seconds = time.monotonic() - started
            trajectories.append(np.asarray(trajectory))
            results.append({
                "state_only": state_only,
                "sync_calls": len(sync_ms),
                "viewer_sync_ms": stats(sync_ms),
                "physics_and_sync_work_ms": stats(work_ms),
                "work_exceeding_physics_period": int(np.count_nonzero(
                    np.asarray(work_ms) > env.sim_dt * 1000
                )),
                "sim_realtime_factor": steps * env.sim_dt / wall_seconds,
                "viewer_qpos_max_error": viewer_error,
            })
        errors = [float(np.max(np.abs(t - trajectories[0]))) for t in trajectories]
        if max(errors) > 1e-9 or max(r["viewer_qpos_max_error"] for r in results) > 1e-9:
            raise RuntimeError(f"Physics/viewer state mismatch: {errors}, {results}")
        return {
            "mujoco_version": mujoco.__version__,
            "physics_hz": 1 / env.sim_dt,
            "viewer_hz": 1 / (env.sim_dt * viewer_every),
            "steps_per_run": steps,
            "scene": config["ROBOT_SCENE"],
            "trajectory_max_errors": errors,
            "runs": results,
            "note": "Viewer/physics comparison only; excludes PICO, policy, DDS and display scanout.",
        }
    finally:
        env.viewer.close()
        # close() requests exit; let the GUI thread finish before Python teardown.
        for thread in set(threading.enumerate()) - threads_before:
            thread.join(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds must be finite and positive")
    result = benchmark(args.seconds)
    report = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report)
    print(report, end="")


if __name__ == "__main__":
    main()
