"""打印 XML 模型里各组件跟 qpos / qvel / ctrl 数组下标的对应关系.

用法:
    python3 model_info.py                          # 默认 model/pendulum.xml
    python3 model_info.py double_pendulum.xml

为什么需要它: qpos 的顺序不是"你写在 XML 里的顺序",
而是 body 树的**深度优先遍历**顺序, 每个 body 内部再按 joint 声明顺序.
所以下标要靠 m.jnt_qposadr[jid] 查, 不要靠脑补.
"""

import sys

import mujoco

# 每种关节占用的 qpos 宽度 / qvel(dof) 宽度
JOINT_WIDTH = {
    mujoco.mjtJoint.mjJNT_FREE: (7, 6),   # qpos: x y z qw qx qy qz
    mujoco.mjtJoint.mjJNT_BALL: (4, 3),   # qpos: qw qx qy qz
    mujoco.mjtJoint.mjJNT_SLIDE: (1, 1),  # qpos: 位移 [m]
    mujoco.mjtJoint.mjJNT_HINGE: (1, 1),  # qpos: 角度 [rad]
}
JOINT_NAME = {
    mujoco.mjtJoint.mjJNT_FREE: "free",
    mujoco.mjtJoint.mjJNT_BALL: "ball",
    mujoco.mjtJoint.mjJNT_SLIDE: "slide",
    mujoco.mjtJoint.mjJNT_HINGE: "hinge",
}
TRN_NAME = {
    mujoco.mjtTrn.mjTRN_JOINT: "joint",
    mujoco.mjtTrn.mjTRN_JOINTINPARENT: "jointinparent",
    mujoco.mjtTrn.mjTRN_SLIDERCRANK: "slidercrank",
    mujoco.mjtTrn.mjTRN_TENDON: "tendon",
    mujoco.mjtTrn.mjTRN_SITE: "site",
    mujoco.mjtTrn.mjTRN_BODY: "body",
}


def name_of(m, objtype, i):
    return mujoco.mj_id2name(m, objtype, i) or "<unnamed>"


def joint_widths(m, jid):
    return JOINT_WIDTH[mujoco.mjtJoint(m.jnt_type[jid])]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "model/pendulum.xml"
    m = mujoco.MjModel.from_xml_path(path)

    print("model = %s" % path)
    print("nq=%d (qpos 长度)  nv=%d (qvel/qacc 长度)  nu=%d (ctrl 长度)  "
          "na=%d (actuator_activation)" % (m.nq, m.nv, m.nu, m.na))
    print("nbody=%d  njnt=%d  ngeom=%d  nsite=%d  nkey=%d  nsensordata=%d"
          % (m.nbody, m.njnt, m.ngeom, m.nsite, m.nkey, m.nsensordata))
    print()

    print("== joints: 谁切走了 qpos / qvel 的哪一段 ==")
    print("%3s  %-12s %-6s %8s %3s %7s %3s" %
          ("id", "name", "type", "qposadr", "nq", "dofadr", "nv"))
    for jid in range(m.njnt):
        nq, nv = joint_widths(m, jid)
        print("%3d  %-12s %-6s %8d %3d %7d %3d"
              % (jid, name_of(m, mujoco.mjtObj.mjOBJ_JOINT, jid),
                 JOINT_NAME[mujoco.mjtJoint(m.jnt_type[jid])],
                 m.jnt_qposadr[jid], nq, m.jnt_dofadr[jid], nv))
    print()

    def dump(adr_of, width_of, size, label, extra=None):
        print("== %s 逐元素归属 ==" % label)
        for i in range(size):
            for jid in range(m.njnt):
                a = adr_of(m, jid)
                if a <= i < a + width_of(m, jid):
                    jn = name_of(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
                    print("  %s[%d] = %s%s" % (label, i, jn, extra(i - a) if extra else ""))
                    break
        print()

    def qpos_extra(k):
        """free 关节的 7 个 qpos 是 [x y z qw qx qy qz]"""
        return "  <- 局部第 %d 个分量 (free: 0-2 平移, 3-6 四元数)" % k

    dump(lambda m, j: m.jnt_qposadr[j], lambda m, j: joint_widths(m, j)[0],
         m.nq, "qpos", qpos_extra)
    dump(lambda m, j: m.jnt_dofadr[j], lambda m, j: joint_widths(m, j)[1],
         m.nv, "qvel")

    print("== actuators: ctrl[i] 打到哪儿 ==")
    for aid in range(m.nu):
        trn = TRN_NAME[mujoco.mjtTrn(m.actuator_trntype[aid])]
        tgt = int(m.actuator_trnid[aid, 0])
        if trn in ("joint", "jointinparent"):
            tgt_name = name_of(m, mujoco.mjtObj.mjOBJ_JOINT, tgt)
        elif trn == "tendon":
            tgt_name = name_of(m, mujoco.mjtObj.mjOBJ_TENDON, tgt)
        elif trn == "site":
            tgt_name = name_of(m, mujoco.mjtObj.mjOBJ_SITE, tgt)
        else:
            tgt_name = name_of(m, mujoco.mjtObj.mjOBJ_BODY, tgt)
        print("  ctrl[%d] -> %-9s %-12s (id=%d)  gear=%s  ctrlrange=%s"
              % (aid, name_of(m, mujoco.mjtObj.mjOBJ_ACTUATOR, aid), trn + ":" + tgt_name,
                 tgt, m.actuator_gear[aid, 0],
                 tuple(m.actuator_ctrlrange[aid]) if m.actuator_ctrllimited[aid] else "unlimited"))
    print()

    print("== keyframes: resetDataKeyframe 会写进去什么 ==")
    for k in range(m.nkey):
        print("  key[%d] %-12s qpos=%s  ctrl=%s"
              % (k, name_of(m, mujoco.mjtObj.mjOBJ_KEY, k),
                 m.key_qpos[k], m.key_ctrl[k]))
    print()

    print("== sensors: data.sensordata 的顺序 (和 qpos 无关, 是单独数组) ==")
    for s in range(m.nsensor):
        print("  sensordata[%d..] %s" % (int(m.sensor_adr[s]),
                                         name_of(m, mujoco.mjtObj.mjOBJ_SENSOR, s)))


if __name__ == "__main__":
    main()
