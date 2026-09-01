#!/usr/bin/env python3
"""Raw-sniff MAVLink v1/v2 traffic on udpin ports; report unique message IDs.

No MAVSDK involved. Distinguishes 'PX4 not sending' from 'MAVSDK not seeing'.
Never sends anything, so PX4's peer-lock state is unaffected.
"""
from __future__ import annotations

import argparse
import socket
import time

NAMES = {0: "HEARTBEAT", 1: "SYS_STATUS", 24: "GPS_RAW_INT", 30: "ATTITUDE",
         32: "LOCAL_POSITION_NED", 33: "GLOBAL_POSITION_INT", 74: "VFR_HUD",
         105: "HIGHRES_IMU", 141: "ALTITUDE", 230: "ESTIMATOR_STATUS",
         245: "EXTENDED_SYS_STATE", 331: "ODOMETRY"}


def sniff(port: int, seconds: float) -> dict[int, int]:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("0.0.0.0", port))
    s.settimeout(0.5)
    counts: dict[int, int] = {}
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            data, _ = s.recvfrom(4096)
        except socket.timeout:
            continue
        i = 0
        while i < len(data) - 10:
            if data[i] == 0xFD:  # mavlink v2
                msgid = data[i + 7] | (data[i + 8] << 8) | (data[i + 9] << 16)
                counts[msgid] = counts.get(msgid, 0) + 1
                i += 12 + data[i + 1]
            elif data[i] == 0xFE:  # mavlink v1
                msgid = data[i + 5]
                counts[msgid] = counts.get(msgid, 0) + 1
                i += 8 + data[i + 1]
            else:
                i += 1
    s.close()
    return counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ports", default="14540,14541,14542,14543,14544")
    ap.add_argument("--seconds", type=float, default=8.0)
    args = ap.parse_args()
    for p in [int(x) for x in args.ports.split(",")]:
        try:
            counts = sniff(p, args.seconds)
        except OSError as exc:
            print(f"port {p}: BIND FAILED ({exc})")
            continue
        total = sum(counts.values())
        ids = sorted(counts)
        named = [f"{NAMES.get(i, i)}:{counts[i]}" for i in ids]
        has_pos = 32 in counts or 33 in counts or 331 in counts
        print(f"port {p}: {total} msgs, {len(ids)} ids, position_data={'YES' if has_pos else 'NO'}")
        print(f"   {' '.join(named[:14])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
