"""Print, once a second, how many messages the G1 body and the Dex3 hands send.

Usage (on the robot, from /home/unitree/GEAR-SONIC)::

    .venv_sim/bin/python gear_sonic/scripts/dex3_hand_check.py          # until Ctrl+C
    .venv_sim/bin/python gear_sonic/scripts/dex3_hand_check.py --seconds 5

Columns: ``low`` is ``rt/lowstate`` (body, about 1000 per second), ``hl``/``hr``
are ``rt/dex3/{left,right}/state`` (hands, about 500-800 per second when they
answer), ``cl``/``cr`` are ``rt/dex3/{left,right}/cmd`` (commands, only while a
controller drives the hands). ``hl``/``hr`` at 0 means the hands are not talking
to the robot, even if their LEDs blink green: see docs/g1_teleop_recording.md,
"Power on and checks". The first three joint positions and the ``lost`` counter
(``press_sensor_state[0].lost``) of each hand follow. ``lost`` must stay flat.

Read-only: it only subscribes.
"""

from __future__ import annotations

import argparse
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_, LowState_

TOPICS = (
    ("low", "rt/lowstate", LowState_),
    ("hl", "rt/dex3/left/state", HandState_),
    ("hr", "rt/dex3/right/state", HandState_),
    ("cl", "rt/dex3/left/cmd", HandCmd_),
    ("cr", "rt/dex3/right/cmd", HandCmd_),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--interface", default="enP8p1s0", help="NIC on 192.168.123.x (default: enP8p1s0).")
    parser.add_argument("--seconds", type=float, default=None, help="Stop after this many seconds.")
    args = parser.parse_args()

    ChannelFactoryInitialize(0, args.interface)
    counts = {key: 0 for key, _, _ in TOPICS}
    last: dict = {}

    def make_callback(key):
        def callback(message):
            counts[key] += 1
            last[key] = message

        return callback

    subscribers = []
    for key, topic, message_type in TOPICS:
        subscriber = ChannelSubscriber(topic, message_type)
        subscriber.Init(make_callback(key), 10)
        subscribers.append(subscriber)

    start = time.monotonic()
    try:
        while args.seconds is None or time.monotonic() - start < args.seconds:
            before = dict(counts)
            time.sleep(1.0)
            rates = {key: counts[key] - before[key] for key in counts}
            line = f"{time.strftime('%H:%M:%S')} {rates}"
            for key in ("hl", "hr"):
                message = last.get(key)
                if message is None:
                    continue
                q = [round(motor.q, 2) for motor in message.motor_state][:3]
                lost = message.press_sensor_state[0].lost if len(message.press_sensor_state) else None
                line += f" {key}_q0..2={q} {key}_lost={lost}"
            print(line, flush=True)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
