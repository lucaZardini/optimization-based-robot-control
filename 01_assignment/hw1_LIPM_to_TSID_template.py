# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 10:31:22 2022

@author: Gianluigi Grandesso
"""

import numpy as np
import math
from plot_utils import plot_xy
import matplotlib.pyplot as plt
import hw1_conf as conf


def compute_3rd_order_poly_traj(x0, x1, T, dt):
    """
    Compute the third order polynomial trajectory.
    x0 and x1, the initial and final positions, can be:
    - two dimensions (for x and y axis)
    - one dimension (for z axis)

    @param x0: initial state position
    @param x1: final state position
    @param T: lipm time step
    @param dt: time step of TSID
    @return: the position, velocity and acceleration computed to move from x0 to x1
    """
    interpolation_time_step = int(T / dt)
    dex = - (x1[0] - x0[0]) / (interpolation_time_step**3)
    cx = - 3 * interpolation_time_step * dex
    bx = 3 * dex

    if x0.shape[0] == 2: # we are dealing with position (x,y)
        dy = - (x1[1] - x0[1]) / (2 * interpolation_time_step ** 3)
        cy = - 3 * interpolation_time_step * dy
        by = 3 * dy

        x = np.zeros((2, interpolation_time_step))
        dx = np.zeros((2, interpolation_time_step))
        ddx = np.zeros((2, interpolation_time_step))

        for i in range(interpolation_time_step):
            position_x = x0[0] + bx * i + cx * i * i + dex * i * i * i
            position_y = x0[1] + by * i + cy * i * i + dy * i * i * i
            derivative_x = bx + 2 * cx * i + 3 * dex * i * i
            derivative_y = by + 2 * cy * i + 3 * dy * i * i
            acceleration_x = 2 * cx + 6 * dex * i
            acceleration_y = 2 * cy + 6 * dy * i
            x[0, i] = position_x
            x[1, i] = position_y
            dx[0, i] = derivative_x
            dx[1, i] = derivative_y
            ddx[0, i] = acceleration_x
            ddx[1, i] = acceleration_y

        return x, dx, ddx

    else: # we are dealing with z axis
        x = np.zeros((1, interpolation_time_step))
        dx = np.zeros((1, interpolation_time_step))
        ddx = np.zeros((1, interpolation_time_step))

        for i in range(interpolation_time_step):
            if x0[0] == 0:
                position_x = bx * i + cx * i * i + dex * i * i * i
                derivative_x = bx + 2 * cx * i + 3 * dex * i * i
                acceleration_x = 2 * cx + 6 * dex * i
            else:
                position_x = x0[0] + bx * i + cx * i * i + dex * i * i * i
                derivative_x = + bx + 2 * cx * i + 3 * dex * i * i
                acceleration_x = + 2 * cx + 6 * dex * i

            x[0, i] = position_x
            dx[0, i] = derivative_x
            ddx[0, i] = acceleration_x

        return x, dx, ddx


def compute_foot_traj(foot_steps, N, dt, step_time, step_height, first_phase):
    """
    Compute the foot trajectory.
    The foot trajectory is computed considering only the right foot for the interpolation. Indeed, the right foot
    concides with the stance mode, the left foot concides with the swing mode.

    @param foot_steps: the steps of the foot
    @param N: number of time steps of TSID
    @param dt: time step of the TSID
    @param step_time: time needed for each step
    @param step_height: fixed step height
    @param first_phase: it can be stance or swing
    @return: foot trajectory
    """

    x = np.zeros((3, N+1))  # position
    dx = np.zeros((3, N+1))  # velocity
    ddx = np.zeros((3, N+1))  # acceleration
    N_step = int(step_time/dt)  # number of steps of the TSID. How many time steps tsid are contained in one robot step.
    offset = 0
    if first_phase == 'swing':  # left leg
        offset = N_step
        x[0, :N_step] = foot_steps[0, 0]  # initialization of the x leg foot
        x[1, :N_step] = foot_steps[0, 1]  # initialization of the y leg foot
        
    for s in range(foot_steps.shape[0]):  # number of time steps
        i = offset+s*2*N_step
        x[0, i:i+N_step] = foot_steps[s, 0]  # position x of foot at time step s
        x[1, i:i+N_step] = foot_steps[s, 1]  # position y of foot at time step s
        if s < foot_steps.shape[0]-1:  # if s time step is not the last one
            next_step = foot_steps[s+1, :]  # assign the next step of lipm
        elif first_phase == 'swing':  # if left foot, the last time step we do nothing.
            break
        else:
            # if s time step is the last step, the next step is the same for the right foot.
            # the height is set to 0, meaning that the robot does not have to move
            next_step = foot_steps[s, :]
            step_height = 0.0

        # x and y assignment
        x[:2, i+N_step: i+2*N_step], \
        dx[:2, i+N_step: i+2*N_step], \
        ddx[:2, i+N_step: i+2*N_step] = compute_3rd_order_poly_traj(foot_steps[s, :], next_step, step_time, dt)

        # height assignment for half step time to reach height 0.5 (lift leg).
        x[2, i+N_step: i+int(1.5*N_step)], \
        dx[2, i+N_step: i+int(1.5*N_step)], \
        ddx[2, i+N_step: i+int(1.5*N_step)] = compute_3rd_order_poly_traj(np.array([0.]), np.array([step_height]), 0.5*step_time, dt)

        # height assignment for the remaining half step time to return to height 0 (lower leg).
        x[2, i+int(1.5*N_step):i+2*N_step], \
        dx[2, i+int(1.5*N_step):i+2*N_step], \
        ddx[2, i+int(1.5*N_step):i+2*N_step] = compute_3rd_order_poly_traj(np.array([step_height]), np.array([0.0]), 0.5*step_time, dt)
            
    return x, dx, ddx


def discrete_LIP_dynamics(delta_t, g, h):
    w = math.sqrt(g/h)
    A_d = np.array([[math.cosh(w*delta_t), (1/w)*math.sinh(w*delta_t)], [w*math.sinh(w*delta_t), math.cosh(w*delta_t)]])

    B_d = np.array([1 - math.cosh(w*delta_t), -w*math.sinh(w*delta_t)])

    return A_d, B_d


def interpolate_lipm_traj(T_step, nb_steps, dt_mpc, dt_ctrl, com_z, g, com_state_x, com_state_y, cop_ref, cop_x, cop_y):
    """
    Interpolate lipm traj

    @param T_step: time for each step (default in conf value)
    @param nb_steps: number of desired walking steps
    @param dt_mpc: sampling time interval
    @param dt_ctrl: time step of TSID
    @param com_z: Center of Mass height
    @param g: value of the gravity
    @param com_state_x: center of mass x
    @param com_state_y: center of mass y
    @param cop_ref: center of pressure reference
    @param cop_x: center of pressure x
    @param cop_y: center of pressure y
    @return:
    """

    # INTERPOLATE WITH TIME STEP OF CONTROLLER (TSID)
    N = nb_steps * int(round(T_step/dt_mpc))  # number of time steps for traj-opt
    N_ctrl = int((N*dt_mpc)/dt_ctrl)   # number of time steps for TSID
    com = np.empty((3, N_ctrl+1))*np.nan
    dcom = np.zeros((3, N_ctrl+1))
    ddcom = np.zeros((3, N_ctrl+1))
    cop = np.empty((2, N_ctrl+1))*np.nan
    foot_steps = np.empty((2, N_ctrl+1))*np.nan
    contact_phase = (N_ctrl+1)*['right']
    com[2, :] = com_z
    
    N_inner = int(N_ctrl/N)
    for i in range(N):
        com[0, i*N_inner] = com_state_x[i,0]
        com[1, i*N_inner] = com_state_y[i,0]
        dcom[0, i*N_inner] = com_state_x[i,1]
        dcom[1, i*N_inner] = com_state_y[i,1]
        if i > 0:
            if np.linalg.norm(cop_ref[i, :] - cop_ref[i-1, :]) < 1e-10:
                contact_phase[i*N_inner] = contact_phase[i*N_inner-1]
            else:
                if contact_phase[(i-1)*N_inner] == 'right':
                    contact_phase[i*N_inner] = 'left'
                elif contact_phase[(i-1)*N_inner] == 'left':
                    contact_phase[i*N_inner] = 'right'
                    
        for j in range(N_inner):
            ii = i*N_inner + j
            (A, B) = discrete_LIP_dynamics((j+1)*dt_ctrl, g, com_z)
            foot_steps[:, ii] = cop_ref[i, :].T
            cop[0, ii] = cop_x[i]
            cop[1, ii] = cop_y[i]
            x_next = A.dot(com_state_x[i,:]) + B.dot(cop[0, ii])
            y_next = A.dot(com_state_y[i,:]) + B.dot(cop[1, ii])
            com[0, ii+1] = x_next[0]
            com[1, ii+1] = y_next[0]
            dcom[0, ii+1] = x_next[1]
            dcom[1, ii+1] = y_next[1]
            ddcom[:2, ii] = g/com_z * (com[:2, ii] - cop[:, ii])
            
            if j > 0:
                contact_phase[ii] = contact_phase[ii-1]
    return com, dcom, ddcom, cop, contact_phase, foot_steps


# READ COM-COP TRAJECTORIES COMPUTED WITH LIPM MODEL
data = np.load(conf.DATA_FILE_LIPM)
com_state_x = data['com_state_x']
com_state_y = data['com_state_y']
cop_ref = data['cop_ref']
cop_x = data['cop_x']
cop_y = data['cop_y']
foot_steps = data['foot_steps']

# INTERPOLATE WITH TIME STEP OF CONTROLLER (TSID)
dt_ctrl = conf.dt  # time step used by TSID
com, dcom, ddcom, cop, contact_phase, foot_steps_ctrl = interpolate_lipm_traj(conf.T_step, conf.nb_steps, conf.dt_mpc,
                                                                              dt_ctrl, conf.h, conf.g, com_state_x,
                                                                              com_state_y, cop_ref, cop_x, cop_y)
    
# COMPUTE TRAJECTORIES FOR FEET
N = conf.nb_steps * int(round(conf.T_step/conf.dt_mpc))  # number of time steps for traj-opt
N_ctrl = int((N*conf.dt_mpc)/dt_ctrl)  # number of time steps for TSID
foot_steps_RF = foot_steps[::2, :]  # assume first foot step corresponds to right foot
x_RF, dx_RF, ddx_RF = compute_foot_traj(foot_steps_RF, N_ctrl, dt_ctrl, conf.T_step, conf.step_height, 'stance')
foot_steps_LF = foot_steps[1::2, :]
x_LF, dx_LF, ddx_LF = compute_foot_traj(foot_steps_LF, N_ctrl, dt_ctrl, conf.T_step, conf.step_height, 'swing')

# SAVE COMPUTED TRAJECTORIES IN NPY FILE FOR TSID
np.savez(conf.DATA_FILE_TSID, com=com, dcom=dcom, ddcom=ddcom, 
         x_RF=x_RF, dx_RF=dx_RF, ddx_RF=ddx_RF,
         x_LF=x_LF, dx_LF=dx_LF, ddx_LF=ddx_LF,
         contact_phase=contact_phase, cop=cop)

# PLOT STUFF
time_ctrl = np.arange(0, round(N_ctrl*dt_ctrl, 2), dt_ctrl)

for i in range(3):
    plt.figure()
    plt.plot(time_ctrl, x_RF[i, :-1], label='x RF '+str(i))
    plt.plot(time_ctrl, x_LF[i, :-1], label='x LF '+str(i))
    plt.legend()
    
time = np.arange(0, round(N*conf.dt_mpc, 2), conf.dt_mpc)
for i in range(2):
    plt.figure()
    plt.plot(time_ctrl, cop[i,:-1], label='CoP')
#    plt.plot(time_ctrl, foot_steps_ctrl[i,:-1], label='Foot step')
    plt.plot(time_ctrl, com[i,:-1], 'g', label='CoM')
    if i == 0:
        plt.plot(time, com_state_x[:-1,0], ':', label='CoM TO')
    else:
        plt.plot(time, com_state_y[:-1,0], ':', label='CoM TO')
    plt.legend()

foot_length = conf.lxn + conf.lxp   # foot size in the x-direction
foot_width = conf.lyn + conf.lyp   # foot size in the y-direciton
plot_xy(time_ctrl, N_ctrl, foot_length, foot_width, foot_steps_ctrl.T, cop[0, :], cop[1, :], com[0, :].reshape((N_ctrl+1, 1)), com[1, :].reshape((N_ctrl+1, 1)))
plt.plot( com_state_x[:, 0], com_state_y[:, 0], 'r* ', markersize=15,)
plt.gca().set_xlim([-0.2, 0.4])
plt.gca().set_ylim([-0.3, 0.3])
