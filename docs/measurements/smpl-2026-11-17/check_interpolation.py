"""Compare old normalized lerp with current SLERP, independent SciPy oracle and timing."""
from pathlib import Path
import json
import time
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from gear_sonic.scripts.pico_manager_thread_server import _interp_pose_axis_angle

def previous(p0,p1,a):
    q0,q1=Rotation.from_rotvec(p0).as_quat(),Rotation.from_rotvec(p1).as_quat()
    out=np.empty_like(q0)
    for i in range(len(q0)):
        b=q1[i] if q0[i]@q1[i]>=0 else -q1[i]
        q=(1-a)*q0[i]+a*b
        out[i]=q/np.linalg.norm(q)
    return Rotation.from_quat(out).as_rotvec()

def main():
    rng=np.random.default_rng(17)
    start=Rotation.random(21,random_state=rng).as_rotvec()
    end=Rotation.random(21,random_state=rng).as_rotvec()
    report={'scope':'CPU helper timing, not system latency; synthetic angular intervals, not measured tracking gaps','timing_us':{},'constant_turn_error_deg':[]}
    for name,func in [('previous',previous),('slerp',_interp_pose_axis_angle)]:
        for _ in range(100): func(start,end,.25)
        times=[]
        for _ in range(3000):
            t=time.perf_counter_ns();func(start,end,.25);times.append((time.perf_counter_ns()-t)/1000)
        report['timing_us'][name]={'median':float(np.median(times)), 'p95':float(np.percentile(times,95))}
    for angle in (1,5,20,60,120,179.9):
        p0=np.zeros((21,3)); p1=p0.copy(); p1[:,0]=np.radians(angle)
        oracle=Slerp([0,1],Rotation.from_euler('x',[0,angle],degrees=True))([.25])
        row={'input_rotation_deg':angle}
        for name,func in [('previous',previous),('slerp',_interp_pose_axis_angle)]:
            row[name]=float(np.degrees((oracle.inv()*Rotation.from_rotvec(func(p0,p1,.25))).magnitude()).max())
        report['constant_turn_error_deg'].append(row)
    Path(__file__).with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
