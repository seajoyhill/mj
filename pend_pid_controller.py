"""仿真单摆的 PD 控制，并记录和绘制运行结果。

模型与约定：
    使用 model/pendulum.xml，关节 hinge 的位置和速度分别为
    d.qpos[0]（theta）与 d.qvel[0]（theta_dot）。theta = 0 表示竖直下垂，
    theta = pi 表示竖直向上。仿真从竖直下垂位置开始，目标角度为 TH_REF。

控制律：
    u = Kp_th * (TH_REF - theta) - Kd_th * theta_dot。
    d.ctrl[0] 保存未限幅的力矩命令；模型中的执行器将实际力矩限制在
    [-5, 5] N·m，不会改写 d.ctrl[0]。控制器没有积分项。

运行与绘图：
    macOS：mjpython pend_pid_controller.py --viewer --plot
    Linux：python3 pend_pid_controller.py --viewer --plot
    可用 --seconds 指定仿真时长，默认 20 秒。每步记录时间、角度、角速度
    和控制量。--plot 显示角度、绘图误差 wrap(theta - TH_REF) 与控制量。
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
TH_REF = 3.14  # 目标角度，单位为弧度；3.14 约等于竖直向上的 pi。


def policy(m, d, th_target=TH_REF):
    """根据角度和角速度计算未限幅的力矩命令，并写入 d.ctrl[0]。"""

    d.ctrl[0] = Kd_th * (0.0 - d.qvel[0]) + Kp_th * ( th_target - d.qpos[0] )


def record(log, d):
    """将当前时间、角度、角速度和控制量追加到日志。"""

    log["t"].append(d.time)
    log["th"].append(d.qpos[0])
    log["thd"].append(d.qvel[0])
    log["u"].append(d.ctrl[0])


def wrap(a):
    """将角度折叠到 [-pi, pi)。"""

    return (a + np.pi) % (2.0 * np.pi) - np.pi


def to_arrays(log, th_ref=TH_REF):
    """将日志转换为时间、角度、绘图误差和控制量数组。

    绘图误差按 theta - th_ref 计算；控制律直接使用 th_ref - theta。
    """

    t = np.array(log["t"])
    th = np.array(log["th"])
    err = np.array([wrap(a - th_ref) for a in th])
    u = np.array(log["u"])
    return t, th, err, u


def plot(t, th, err, u, umax, timestep):
    """绘制角度、绘图误差和未限幅的控制量。"""

    import matplotlib
    import matplotlib.pyplot as plt

    # 为图表中的中文标签设置字体。
    matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Droid Sans Fallback",
                                              "WenQuanYi Micro Hei", "DejaVu Sans"]
    if sys.platform == "darwin":
        # macOS 专用：优先使用苹果苹方，并选择系统实际提供的字族。
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
    span = max(np.abs(err).max(), 1e-3) * 1.2  # 纵轴以目标角度为中心显示偏差。
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
    """显示图表；在 macOS 上由普通 Python 子进程创建窗口。"""

    if sys.platform == "darwin":
        # macOS 专用修复：mjpython 在工作线程运行脚本，而 Matplotlib 的
        # macosx 后端只能在系统主线程创建窗口，因此交给普通 Python 子进程。
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
    """解析命令行参数，运行仿真并输出统计结果。"""

    ap = argparse.ArgumentParser()
    ap.add_argument("--viewer", action="store_true", help="显示实时仿真窗口")
    ap.add_argument("--plot", action="store_true", help="仿真结束后显示角度、误差和控制量曲线")
    ap.add_argument("--seconds", type=float, default=20.0, help="仿真时长，单位为秒")
    ap.add_argument("--_plot-data", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args._plot_data is not None:
        # macOS 专用修复：子进程在系统主线程创建并显示图表。
        with np.load(args._plot_data) as data:
            plot(data["t"], data["th"], data["err"], data["u"],
                 float(data["umax"]), float(data["timestep"]))
        import matplotlib.pyplot as plt
        plt.show()
        return

    m = mujoco.MjModel.from_xml_path("model/pendulum.xml")
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)  # 重置到 down 关键帧。
    d.qpos[0] = 0  # 从竖直下垂位置开始。

    log = {"t": [], "th": [], "thd": [], "u": []}

    if args.viewer:
        # 按需导入 viewer，避免将全局 mujoco 名称变成局部变量。
        from mujoco import viewer as mj_viewer

        with mj_viewer.launch_passive(m, d) as v:
            t0 = time.perf_counter()
            while v.is_running() and d.time < args.seconds:
                policy(m, d)
                mujoco.mj_step(m, d)
                record(log, d)
                if (slack := d.time - (time.perf_counter() - t0)) > 0:
                    time.sleep(slack)  # 将可视化仿真限制为实时速度。
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
