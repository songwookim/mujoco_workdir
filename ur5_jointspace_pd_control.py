import mujoco as mj
import mujoco.viewer
#from robot_d escriptions import ur5e_mj_description
import numpy as np
#import mediapy as media
#model = mj.MjModel.from_xml_path(ur5e_mj_description.MJCF_PATH) # ~/.cache/robot_descriptions/mujoco_menagerie/universal_robots_ur5e
model = mj.MjModel.from_xml_path("./universal_robots_ur5e/ur5e.xml")


# other option : model = load_robot_description("ur5e_mj_description")  [from robot_descriptions.loaders.mujoco import load_robot_description]

data = mj.MjData(model)                # MuJoCo data

cam = mj.MjvCamera() 
mujoco.mjv_defaultFreeCamera(model, cam)
cam.distance = 1
opt = mj.MjvOption()                        # visualization options
spec = mj.MjSpec()    

# for multiple scene
pi = np.pi

model2 = mj.MjModel.from_xml_path("./universal_robots_ur5e/ur5e.xml")
data2 = mujoco.MjData(model)
data2.qpos = [pi/2, -pi/2, pi/2, pi/2, pi/2, 0, 0]
vopt2 = mujoco.MjvOption()
vopt2.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True  # Transparent.

model3 = mj.MjModel.from_xml_path("./universal_robots_ur5e/sphere.xml")
data3 = mujoco.MjData(model3)
vopt3 = mujoco.MjvOption()
vopt3.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True 

pert = mujoco.MjvPerturb()  # Empty MjvPerturb object
# We only want dynamic objects (the humanoid). Static objects (the floor)
# should not be re-drawn. The mjtCatBit flag lets us do that, though we could
# equivalently use mjtVisFlag.mjVIS_STATIC
catmask = mujoco.mjtCatBit.mjCAT_DYNAMIC


#Put a position of the joints to get a test point
pi = np.pi
# data.qpos = [3*pi/2, -pi/2, pi/2, 3*pi/2, 3*pi/2, 0]

#Inititial joint position
qpos0 = data.qpos.copy()

#Step the simulation.
mujoco.mj_forward(model, data)
mujoco.mj_forward(model2, data2)

#Use the last piece as an "end effector" to get a test point in cartesian coordinates
target = data.body('eef').xpos.copy()

#Plot results
print("Results")
mujoco.mj_resetDataKeyframe(model, data, 1)
mujoco.mj_forward(model, data)
renderer = mujoco.Renderer(model)
init_point = data.body('eef').xpos.copy()
renderer.update_scene(data, cam)
target_plot = renderer.render()

data.qpos = qpos0
mujoco.mj_forward(model, data)
result_point = data.body('eef').xpos.copy()
renderer.update_scene(data, cam)
result_plot = renderer.render()

print("initial point =>", init_point)
print("Desire point =>", result_point, "\n")

images = {
    'Initial position': target_plot,
    ' Desire end effector position': result_plot,
}

# media.show_images(images)
def circle(t: float, r: float, h: float, k: float, f: float) -> np.ndarray:
    """Return the (x, y) coordinates of a circle with radius r centered at (h, k)
    as a function of time t and frequency f."""
    x = r * np.cos(2 * np.pi * f * t) + h
    y = r * np.sin(2 * np.pi * f * t) + k
    z = 0.5
    return np.array([x, y, z])

#get the name of the joints and its limits
for j in range(len(data.qpos)):
    print("name part =>", data.jnt(j).name, "\n", 
          "limit =>", model.jnt_range[j], "\n")


#Video Setup
DURATION = 4 #(seconds)
FRAMERATE = 60 #(Hz)
frames = []

#Reset state and time.
mujoco.mj_resetData(model, data)

#Init parameters
jacp = np.zeros((3, model.nv)) #translation jacobian
jacr = np.zeros((3, model.nv)) #rotational jacobian
step_size = 0.5
tol = 0.01
alpha = 0.5
damping = 0.

#Get error.
end_effector_id = model.body('eef').id #"End-effector we wish to control.
current_pose = data.body(end_effector_id).xpos #Current pose

goal = [0.50, 0.27, 1.79] #Desire position

x_error = np.subtract(goal, current_pose) #Init Error

def check_joint_limits(q):
    """Check if the joints is under or above its limits"""
    for i in range(len(q)):
        q[i] = max(model.jnt_range[i][0], min(q[i], model.jnt_range[i][1]))

#Simulate
i = 0
iter_ct = 0
x = data.xpos[7,:]
mujoco.set_mjcb_control(None)
with mujoco.viewer.launch_passive(model, data) as viewer:
    time_prev = data.time
    # data2.qpos = data.qpos
    # data2.qpos[0] -= 1.5
    # data2.qpos[1] -= 1
    # data2.xpos[0] - = 2
    mujoco.mj_forward(model2, data2) # update data2.xpos (ee_pose)
    mujoco.mjv_updateScene(model2, data2, vopt2, pert, cam, catmask, viewer.user_scn)
    # mujoco.mj_forward(model3, data3) # update data2.xpos (ee_pose)
    # mujoco.mjv_addGeoms(model3, data3, vopt3, pert, catmask, viewer.user_scn)
    
    viewer.sync()

    # mujoco.mj_step(model, data)

    while viewer.is_running():
        # while data.time < DURATION:
        
        # mujoco.mjv_addGeom(model, data2, vopt2, pert, catmask, viewer.user_scn)
        
        # goal = circle(data.time, 0.1, 0.5, 0.0, 0.5) #ENABLE to test circle.
        
        if (np.linalg.norm(x_error) >= tol):
            #Calculate jacobian
            mujoco.mj_jac(model, data, jacp, jacr, goal, end_effector_id)
            #Calculate delta of joint q
            n = jacp.shape[1]
            I = np.identity(n)
            product = jacp.T @ jacp + damping * I

            if np.isclose(np.linalg.det(product), 0):
                j_inv = np.linalg.pinv(product) @ jacp.T
            else:
                j_inv = np.linalg.inv(product) @ jacp.T
            grad = 0.01 * jacp.T @ x_error
            
            # data.qfrc_actuator[6] = 3
            # data.qpos += step_size * grad
            # delta_q = np.linalg.pinv(jacp) @ x_error
            kp = np.ones(7)*300
            # kp[3:6] = 1200
            kd = 100
            e_pos = data.qpos - data2.qpos
            e_vel = data.qvel
            M = np.zeros((model.nv, model.nv)) # 6x6
            mj.mj_fullM(model, M, data.qM)
            # data.qfrc_applied[0] = e_pos[0]*kp + e_vel[0]*kd
            # data.qfrc_applied = e_pos*kp + e_vel*kd
            data.efc_J
            c = data.qfrc_bias
            # data.qfrc_actuator  = data2.qpos
            tau = e_pos*kp + e_vel*kd 
            # data.ctrl = -tau # PD control
            data.ctrl = np.linalg.inv(M)@(-c-tau) # computed-torque PD control
            
            # data.xfrc_applied = data.xfrc_applied+3
            # mj.mj_fwdActuation(model, data)
            #Compute next step
            # q = data.qpos.copy()
            # q += step_size * delta_q
            
            #Check limits
            check_joint_limits(data.qpos)
            mj.mj_forward(model, data)
            #Set control signal
            # data.ctrl[1:7] = q[1:7] 
            # data.ctrl[8:17] = q[8:17] * np.sin(data.time)*2
            #Step the simulation.
            x_before = x.copy()
            x = data.xpos[8,:].copy()
            
            iter_ct += 1
            
            # if iter_ct < 10000:
            #     with viewer.lock():
            #         viewer.user_scn.ngeom += 1
            #         mujoco.mjv_initGeom(viewer.user_scn.geoms[viewer.user_scn.ngeom - 1],
            #                             mujoco.mjtGeom.mjGEOM_CAPSULE, np.array([0.38, 0.42, 1.0]),
            #                             np.zeros(3), np.zeros(9), np.array([1, 0., 0., 0.1]))
            #         mujoco.mjv_connector(viewer.user_scn.geoms[viewer.user_scn.ngeom - 1], mujoco.mjtGeom.mjGEOM_CAPSULE, 0.002, # radius
            #                         np.array([ x_before[0], x_before[1], x_before[2]]), # from
            #                         np.array([ x[0], x[1], x[2]])) # to 
            #         mujoco.mjv_connector(viewer.user_scn.geoms[viewer.user_scn.ngeom - 1], mujoco.mjtGeom.mjGEOM_CAPSULE, 0.002, # radius
            #                         np.array([ goal[0], goal[1], goal[2]]), # from
            #                         np.array([ x[0], x[1], x[2]])) # to 
            # else :
            #     continue
            viewer.sync()

            mujoco.mj_step(model, data)

            x_error = np.subtract(goal, data.body(end_effector_id).xpos)
        else :
            print("complete")
