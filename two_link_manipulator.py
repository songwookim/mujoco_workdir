import mujoco as mj
import mujoco.viewer
import numpy as np
import os
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
xml_path = '2D_simple_pendulum.xml' #xml file (assumes this is in the same folder as this file)
def controller(model, data):
    return
def set_torque_servo(actuator_no, flag):
    if (flag==0):
        model.actuator_gainprm[actuator_no, 0] = 0
    else:
        model.actuator_gainprm[actuator_no, 0] = 1
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname + "/" + xml_path)
xml_path = abspath
# MuJoCo data structures
# model = mujoco_py.load_model_from_path(mj_path)%
model = mj.MjModel.from_xml_path("2D_simple_pendulum.xml") # MuJoCo model
data = mj.MjData(model)                # MuJoCo data
cam = mj.MjvCamera()
opt = mj.MjvOption()                        # visualization options
spec = mj.MjSpec()
# opt = mj.mjvPerturb()                        # visualization options
cam.azimuth = -90.68741727466428 ; cam.elevation = -2.8073894766455036 ; cam.distance =  0.457557373462702
cam.lookat =np.array([ 0.0 , 0.0 , 5.0 ])
#set the controller
mj.set_mjcb_control(controller)
# # mujoco.mj_contactForce(model, data, j, forcetorque)
# # simulate and save data
# for i in range(n_steps):
#   mujoco.mj_step(model, data)
#   sim_time[i] = data.time
# #   ncon[i] = data.ncon
# #   velocity[i] = data.qvel[:]
# #   acceleration[i] = data.qacc[:]
#   # iterate over active contacts, save force and distance
#   for j,c in enumerate(data.contact):
#     mujoco.mj_contactForce(model, data, j, forcetorque)
#     force[i] += forcetorque[0:3]
#     # penetration[i] = min(penetration[i], c.dist)
#   # we could also do
#     force[i] += data.qfrc_constraint[0:3]
#   # do you see why?
qposs, qvels, forces, torques, force_real, torque_real, actuator_forces = [], [], [], [], [], [], []
t = 0
mujoco.viewer.cam = cam
with mujoco.viewer.launch_passive(model, data) as viewer:
    time_prev = data.time
    viewer._cam.distance = 6.8
    while viewer.is_running():
        t += 1
        force=data.sensor("force").data
        torque=data.sensor("torque").data
        os.system('cls')
        # https://github.com/google-deepmind/mujoco/blob/main/include/mujoco/mjdata.h
        print("-----print force--:  ", np.round(force,2))
        print("-----print torque--:  ", np.round(torque,2))
        print("-----print qpos--:  ", np.round(data.qpos,2))
        print("-----print qvel--:  ", np.round(data.qvel,2))
        print("-----print qvel--:  ", np.round(data.qacc,2))
        print("-----print qfrc_applied--:  ", np.round(data.qfrc_applied,2))
        print("-----print xfrc_applied--:  ", np.round(data.xfrc_applied,2)) # applied Cartesian force/torque                   (nbody x 6)
        print("-----print qfrc_constraint--:  ", np.round(data.qfrc_constraint,2)) # constraint force
        print("-----print qfrc_smooth--:  ", np.round(data.qfrc_smooth,2)) # net unconstrained force
        print("-----print qfrc_actuator--:  ", np.round(data.qfrc_actuator,2))
        print("-----print actuator_force--:  ", np.round(data.actuator_force,2))
        print("-----print act--:  ", np.round(data.act,2)) # actuactor -> general
        print("-----print ctrl--:  ", np.round(data.ctrl,2)) # actuactor -> motor
        print("-----print sensordata--:  ", np.round(data.sensordata,2)) # actuactor -> motor
        print("-----print t--:  ", t)
        forces.append(force.copy())
        torques.append(torque.copy())
        qposs.append(data.qpos.copy())
        qvels.append(data.qvel.copy())
        actuator_forces.append(data.qfrc_actuator.copy())
        if data.contact.dim.size > 0 :
            #   for j,c in enumerate(data.contact):
            temp_ft = np .zeros(6)
            mujoco.mj_contactForce(model, data, 0, temp_ft)
            # penetration[i] = min(penetration[i], c.dist)
            # # we could also do
            # force[i] += data.qfrc_constraint[0:3]
            # # do you see why?
            force_real.append(temp_ft[0:3])
            torque_real.append(temp_ft[3:7])
        else :
            force_real.append([0,0,0])
            torque_real.append([0,0,0])
        # data.qfrc_applied = t
        # mujoco.mj_fwdActuation(model, data)
        # data.ctrl[1] =  2
        # data.ctrl[2] =  3
        data.act = 5-force[2]
        mujoco.mj_step(model, data)
        data.ctrl =  t
        # mujoco.mj_forward(model, data)
        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 1
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
            viewer.opt.frame = mujoco.mjtFrame.mjFRAME_GEOM
        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        viewer.sync()
# plot
_, ax = plt.subplots(6, 1, sharex=True, figsize=(10, 10))
sim_time = range(t)
lines = ax[0].plot(sim_time, forces)
ax[0].set_title('force obs')
ax[0].set_ylabel('Newton')
ax[0].set_xlabel('time')
ax[0].legend(iter(lines), ('$F_x $', '$F_y $', '$F_z $'))
lines = ax[1].plot(sim_time, qposs, qvels)
ax[1].set_title('joint_position / joint_velocity')
ax[1].legend(iter(lines), ('q', 'q_dot'))
ax[1].set_ylabel('(meter,radian)')
ax[1].set_xlabel('time')
lines = ax[2].plot(sim_time, force_real)
ax[2].set_title('force')
ax[2].legend(iter(lines), ('$F_x $', '$F_y $', '$F_z $'))
ax[2].set_ylabel('Newton')
ax[2].set_xlabel('time')
lines = ax[3].plot(sim_time, torque_real)
ax[3].set_title('torque')
ax[3].legend(iter(lines), ('$ \\tau_x $', '$\\tau_y $', '$\\tau_z $'))
ax[3].set_ylabel('$p(v_s)$')
ax[3].set_xlabel('time')
lines = ax[4].plot(sim_time, actuator_forces)
ax[4].set_title('actuator force')
ax[4].legend(iter(lines), ('$ \\tau_x $', '$\\tau_y $', '$\\tau_z $'))
ax[4].set_ylabel('$p(v_s)$')
ax[4].set_xlabel('time')
lines = ax[5].plot(sim_time, torques)
ax[5].set_title('torque obs')
ax[5].legend(iter(lines), ('$ \\tau_x $', '$\\tau_y $', '$\\tau_z $'))
ax[5].set_ylabel('$p(v_s)$')
ax[5].set_xlabel('time')
plt.tight_layout()
plt.show()
# ax[2,0].plot(sim_time, velocity)
# ax[2,0].set_title('velocity')
# ax[2,0].set_ylabel('(meter,radian)/s')
# ax[2,0].set_xlabel('second')
# ax[0,1].plot(sim_time, ncon)
# ax[0,1].set_title('number of contacts')
# ax[0,1].set_yticks(range(6))
# ax[1,1].plot(sim_time, np.array(forces)[:,0])
# ax[1,1].set_yscale('log')
# ax[1,1].set_title('normal (z) force - log scale')
# ax[1,1].set_ylabel('Newton')
# z_gravity = -model.opt.gravity[2]
# mg = model.body("world").mass[0] * z_gravity
# mg_line = ax[1,1].plot(range(t), np.ones(t)*mg, label='m*g', linewidth=1)
# ax[1,1].legend()
# ax[2,1].plot(sim_time, 1000*penetration)
# ax[2,1].set_title('penetration depth')
# ax[2,1].set_ylabel('millimeter')
# ax[2,1].set_xlabel('second')