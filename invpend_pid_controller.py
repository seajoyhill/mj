"""控制 slider 的最小示例 (小车-倒立摆 PID)

模型里的执行器是 <motor name="slide" joint="slider" gear="1" ctrlrange="-20 20">,
gear=1 且是 motor, 所以控制接口就一行:

    data.ctrl[0] = 施加在小车上的力 [N]     # 范围 [-20, 20], 由 ctrlrange 限制

状态: qpos = [cart_x, hinge], qvel = [cart_vx, hinge_v]
符号: 正力 -> 小车沿 +x 加速 -> 杆被惯性甩向 -theta,
      所以杆是正倾角时要用"正力"去救(正反馈), 下面 Kp_th 为正;
      外环反过来: 想让小车往 -x 走, 得先让杆朝 -x 倒, 所以 Kp_x 带负号(写在 th_ref 里)。

实测: 从 |th0|<=0.3, |x0|<=2.5 都能回到竖直且 x->0。

用法: python3 invpend_pid_controller.py [--viewer] [--seconds 20]
"""

import argparse
import time

import mujoco
import numpy as np

Kp_th, Kd_th = 200.0, 40.0             # 内环: 稳杆
Kp_x, Kd_x, TH_MAX = 0.5, 0.5, 0.05    # 外环: 定车 (输出是"期望倾角", 必须限幅, 否则摆极限环)


def policy(m, d):
    """控制器: 唯一要记住的接口就是 data.ctrl[0]"""
    x, th = d.qpos[0], d.qpos[1]
    dx, dth = d.qvel[0], d.qvel[1]

    th_ref = np.clip(-Kp_x * x - Kd_x * dx, -TH_MAX, TH_MAX)  # 外环: 把小车拉回 x=0 所需的期望倾角
    d.ctrl[0] = np.clip(Kp_th * (th - th_ref) + Kd_th * dth, -20, 20)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--viewer", action="store_true", help="打开实时可视化")
    ap.add_argument("--seconds", type=float, default=20.0, help="仿真时长 [s]")
    args = ap.parse_args()

    m = mujoco.MjModel.from_xml_path("model/inverted_pendulum.xml")
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)   # keyframe "upright": 竖直平衡点
    d.qpos[1] = 0.2                        # 先推歪 0.2 rad

    if args.viewer:
        # 注意: 别写 `import mujoco.viewer`, 那会把 mujoco 变成局部名
        from mujoco import viewer as mj_viewer

        with mj_viewer.launch_passive(m, d) as v:
            t0 = time.perf_counter()
            while v.is_running() and d.time < args.seconds:
                policy(m, d)
                mujoco.mj_step(m, d)
                if (slack := d.time - (time.perf_counter() - t0)) > 0:
                    time.sleep(slack)      # 节流到 1x 实时, 否则窗口一闪而过
                v.sync()
    else:
        while d.time < args.seconds:
            policy(m, d)
            mujoco.mj_step(m, d)

    print(f"t={d.time:.1f}s  x={d.qpos[0]:+.5f}  th={d.qpos[1]:+.5f}  ctrl={d.ctrl[0]:+.2f}")


if __name__ == "__main__":
    main()
