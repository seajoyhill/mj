"""单摆的 PD 控制器 (目标: 竖直向上)

= 模型 / 接口 (model/pendulum.xml) =
  关节 hinge:  qpos = [theta],  qvel = [theta_dot],  nq = nv = nu = 1
  执行器 <motor joint="hinge" gear="1" ctrllimited="true" ctrlrange="-5 5">, 所以

      data.ctrl[0] = 施加在铰链上的力矩 [N*m]      # 名义范围 [-5, 5]

  约定: qpos = 0 是竖直下垂, qpos = pi 是竖直向上 (本文件的目标)
  铰点处有效转动惯量 I = 0.0369 kg*m^2, 重力力矩系数 m*g*l = 0.736
  ctrlrange 只截断"实际施加的力矩", 不会改写 data.ctrl[0] 本身

= 控制框图 (就是 policy() 里那一行) =

 theta_ref = pi --> (S1) --> [ Kp=200 ] --> (S2) --> [ ctrlrange ] --> [ plant ]--+-----+--->theta
                       -^                     -^                                  |     |
                        |                      |                                  |     |
                        |                      +--[ Kd=40 ]<----------------------+     |
                        +---------------------------------------------------------------+

  (S1) 误差求和点:   e = theta_ref - theta
  (S2) 控制量求和点: u = Kp*e + Kd*(0 - theta_dot)      <-- 只有 P 和 D, 没有 I
  [ ctrlrange ]      限幅: 实际力矩 = clip(u, -5, +5)
  [ plant ]          被控对象: I*theta_dd = tau_g(theta) + tau
  theta / theta_dot  直接读 data.qpos[0] / data.qvel[0], 没有单独的微分器
  右边两个 + 是状态输出的分叉点, 不是求和点
  要做成真正的 PID, 是往 (S2) 再并一条 u += Ki * integral(e dt) 的支路

= 实测 (dt = 0.01, 从 theta0 = pi - 0.3 出发, 8 s) =
  Kp=200, Kd=40 -> 不收敛: u 的原始峰值 60 N*m, 95% 的时间撞上 +-5 限幅,
                   最后停在离 pi 约 0.05 rad, |theta_dot| 约 0.5 rad/s 的极限环
  根因: 显式阻尼项要满足 dt*Kd/I < 2, 即 Kd < 2*0.0369/0.01 = 7.4;
        Kd=40 时 dt*Kd/I = 10.9 >> 2, 数值上发散, 被 +-5 限幅兜成了极限环
  能收敛: Kp=25, Kd=5 (峰值 7.5 N*m, 基本不饱和);  Kp=5, Kd=1 (更慢)

用法: python3 pend_pid_controller.py [--viewer] [--plot] [--seconds 20]

  跑的时候每步都把 (t, theta, theta_dot, u) 存进 log, 所以 --plot 能直接画出
  当前角度 theta(t) 与目标角度 theta_ref 的对比、误差、以及控制量 u。
"""

import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import mujoco
import numpy as np

Kp_th, Kd_th = 10.0, 1.0
TH_REF = 3.14                   # 目标角度 [rad], 约等于 pi (竖直向上)


def policy(m, d, th_target=TH_REF):
    """控制器: 唯一要记住的接口就是 data.ctrl[0]"""

    d.ctrl[0] = Kd_th * (0.0 - d.qvel[0]) + Kp_th * ( th_target - d.qpos[0] )


def record(log, d):
    """把这一步的状态抄进 log, 跑完后再画图 (用 list 存, 比 numpy 逐个 append 快)"""

    log["t"].append(d.time)
    log["th"].append(d.qpos[0])
    log["thd"].append(d.qvel[0])
    log["u"].append(d.ctrl[0])


def wrap(a):
    """把角度折到 (-pi, pi]"""

    return (a + np.pi) % (2.0 * np.pi) - np.pi


def to_arrays(log, th_ref=TH_REF):
    """log -> (t, theta, 误差, u) 四个 numpy 数组"""

    t = np.array(log["t"])
    th = np.array(log["th"])
    err = np.array([wrap(a - th_ref) for a in th])
    u = np.array(log["u"])
    return t, th, err, u


def plot(t, th, err, u, umax, timestep):
    """画图: 上面是当前角度 vs 目标角度, 中间是误差, 下面是控制量"""

    import matplotlib
    import matplotlib.pyplot as plt

    # 默认字体没有中文, 不设会画成一堆方框
    matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Droid Sans Fallback",
                                              "WenQuanYi Micro Hei", "DejaVu Sans"]
    if sys.platform == "darwin":
        # macOS 专用: 优先使用苹果苹方；按系统实际提供的字族选择。
        from matplotlib import font_manager
        available_fonts = {font.name for font in font_manager.fontManager.ttflist}
        for family in ("PingFang SC", "PingFang HK", "PingFang TC"):
            if family in available_fonts:
                matplotlib.rcParams["font.sans-serif"].insert(0, family)
                break
    matplotlib.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(9.5, 8.0))

    ax[0].plot(t, th, color="#1f77b4", lw=1.4, label="当前角度 theta")
    ax[0].axhline(TH_REF, color="#d62728", ls="--", lw=1.2,
                  label="目标角度 theta_ref = %.2f" % TH_REF)
    span = max(np.abs(err).max(), 1e-3) * 1.2      # 纵轴以目标角度为中心, 偏差才看得清
    ax[0].set_ylim(TH_REF - span, TH_REF + span)
    ax[0].set_ylabel("角度 [rad]")
    ax[0].legend(loc="lower right", fontsize=9)
    ax[0].set_title("单摆 PD: 当前角度 vs 目标角度    Kp=%.0f  Kd=%.0f  dt=%.3f s"
                    % (Kp_th, Kd_th, timestep))

    ax[1].plot(t, err, color="#2ca02c", lw=1.4, label="误差 theta - theta_ref (折到 +-pi)")
    ax[1].axhline(0.0, color="k", lw=0.8)
    ax[1].set_ylabel("误差 [rad]")
    ax[1].legend(loc="upper right", fontsize=9)
    ax[1].grid(alpha=0.3)

    ax[2].plot(t, u, color="#ff7f0e", lw=1.2, label="控制量 u = data.ctrl[0]")
    ax[2].axhline(umax, color="#d62728", ls="--", lw=1.0)
    ax[2].axhline(-umax, color="#d62728", ls="--", lw=1.0)
    ax[2].text(t[0], umax, " ctrlrange +-%.0f N*m" % umax, color="#d62728",
               fontsize=9, va="bottom")
    ax[2].set_ylabel("力矩 [N*m]")
    ax[2].set_xlabel("时间 t [s]")
    ax[2].legend(loc="upper right", fontsize=9)
    ax[2].grid(alpha=0.3)

    fig.tight_layout()


def show_plot(t, th, err, u, umax, timestep):
    if sys.platform == "darwin":
        # macOS 专用修复: mjpython 在工作线程运行脚本，Matplotlib 的 macosx
        # 后端只能在系统主线程创建窗口。用普通 Python 子进程显示图表。
        with tempfile.TemporaryDirectory() as tmpdir:
            plot_path = Path(tmpdir) / "pendulum_plot.npz"
            np.savez(plot_path, t=t, th=th, err=err, u=u,
                     umax=umax, timestep=timestep)
            subprocess.run([sys.executable, str(Path(__file__).resolve()),
                            "--_plot-data", str(plot_path)], check=True)
        return

    plot(t, th, err, u, umax, timestep)
    import matplotlib.pyplot as plt
    plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--viewer", action="store_true", help="打开实时可视化")
    ap.add_argument("--plot", action="store_true", help="跑完后画曲线图 (角度/误差/力矩)")
    ap.add_argument("--seconds", type=float, default=20.0, help="仿真时长 [s]")
    ap.add_argument("--_plot-data", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args._plot_data is not None:
        # macOS 专用修复: 此入口由普通 Python 子进程执行，窗口位于系统主线程。
        with np.load(args._plot_data) as data:
            plot(data["t"], data["th"], data["err"], data["u"],
                 float(data["umax"]), float(data["timestep"]))
        import matplotlib.pyplot as plt
        plt.show()
        return

    m = mujoco.MjModel.from_xml_path("model/pendulum.xml")
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)   # keyframe "down": qpos=0 (竖直下垂)
    d.qpos[0] = 0                      # 起点: 竖直向上

    log = {"t": [], "th": [], "thd": [], "u": []}

    if args.viewer:
        # 注意: 别写 `import mujoco.viewer`, 那会把 mujoco 变成局部名
        from mujoco import viewer as mj_viewer

        with mj_viewer.launch_passive(m, d) as v:
            t0 = time.perf_counter()
            while v.is_running() and d.time < args.seconds:
                policy(m, d)
                mujoco.mj_step(m, d)
                record(log, d)
                if (slack := d.time - (time.perf_counter() - t0)) > 0:
                    time.sleep(slack)      # 节流到 1x 实时, 否则窗口一闪而过
                v.sync()
    else:
        while d.time < args.seconds:
            policy(m, d)
            mujoco.mj_step(m, d)
            record(log, d)

    t, th, err, u = to_arrays(log)
    umax = float(m.actuator_ctrlrange[0, 1])

    print(f"t={d.time:.1f}s  th={d.qpos[0]:+.5f}  thd={d.qvel[0]:+.5f}  ctrl={d.ctrl[0]:+.2f}")
    print(f"角度误差: 末值 {abs(err[-1]):.5f} rad, 最大 {np.abs(err).max():.4f} rad, "
          f"平均 {np.abs(err).mean():.5f} rad")
    print(f"控制量 u: 峰值 {np.abs(u).max():.1f} N*m, "
          f"超出 +-{umax:g} 的时间占比 {100.0 * np.mean(np.abs(u) > umax):.1f}%")

    if args.plot:
        show_plot(t, th, err, u, umax, m.opt.timestep)



if __name__ == "__main__":
    main()
