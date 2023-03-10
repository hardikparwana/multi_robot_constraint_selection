from cProfile import label
import numpy as np
import time
import cvxpy as cp
import matplotlib.pyplot as plt
from robot_models.SingleIntegrator2D import *
from robot_models.Unicycle import *
from robot_models.obstacles import *

from matplotlib.animation import FFMpegWriter

plt.rcParams.update({'font.size': 15}) #27

# Sim Parameters                  
dt = 0.05
tf = 9.0 #5.4#8#4.1 #0.2#4.1
num_steps = int(tf/dt)
t = 0
d_min_obstacles = 1.0 #0.1
d_min_agents = 0.2 #0.4
d_max = 2.0

check_for_constraints = True
bigNaN = 10000000

alpha_cbf = 0.7 

# Plot                  
plt.ion()
fig = plt.figure()
ax = plt.axes(xlim=(0,7),ylim=(-0.5,10)) 
ax.set_xlabel("X")
ax.set_ylabel("Y")


num_adversaries = 0
num_obstacles = 3
num_connectivity = 1
alpha = 0.1

save_plot = False
movie_name = 'max_distance_no_trust.mp4'

# agents
robots = []
num_robots = 6

robots.append( SingleIntegrator2D(np.array([3,1.5]), dt, ax, num_robots=num_robots, id = 0, color='g',palpha=1.0, alpha=alpha_cbf, num_adversaries=num_adversaries, num_obstacles=num_obstacles ) )
robots.append( SingleIntegrator2D(np.array([2.6,0]), dt, ax, num_robots=num_robots, id = 1, color='g',palpha=1.0, alpha=alpha_cbf, num_adversaries=num_adversaries, num_obstacles=num_obstacles ) )
robots.append( SingleIntegrator2D(np.array([3.5,0]), dt, ax, num_robots=num_robots, id = 2, color='g',palpha=1.0, alpha=alpha_cbf, num_adversaries=num_adversaries, num_obstacles=num_obstacles ) )
robots.append( SingleIntegrator2D(np.array([2.5,0.8]), dt, ax, num_robots=num_robots, id = 3, color='g',palpha=1.0, alpha=alpha_cbf, num_adversaries=num_adversaries, num_obstacles=num_obstacles ) )
robots.append( SingleIntegrator2D(np.array([3.0,0.8]), dt, ax, num_robots=num_robots, id = 4, color='g',palpha=1.0, alpha=alpha_cbf, num_adversaries=num_adversaries, num_obstacles=num_obstacles ) )
robots.append( SingleIntegrator2D(np.array([3.5,0.8]), dt, ax, num_robots=num_robots, id = 5, color='g',palpha=1.0, alpha=alpha_cbf, num_adversaries=num_adversaries, num_obstacles=num_obstacles ) )

# agent nominal version
robots_nominal = []

robots_nominal.append( SingleIntegrator2D(np.array([3,1.5]), dt, ax, num_robots=num_robots, id = 0, color='g',palpha=alpha) )
robots_nominal.append( SingleIntegrator2D(np.array([2.9,0]), dt, ax, num_robots=num_robots, id = 1, color='g',palpha=alpha ) )
robots_nominal.append( SingleIntegrator2D(np.array([3.5,0]), dt, ax, num_robots=num_robots, id = 2, color='g',palpha=alpha ) )
robots_nominal.append( SingleIntegrator2D(np.array([2.5,0.8]), dt, ax, num_robots=num_robots, id = 3, color='g',palpha=alpha ) )
robots_nominal.append( SingleIntegrator2D(np.array([3.0,0.8]), dt, ax, num_robots=num_robots, id = 4, color='g',palpha=alpha ) )
robots_nominal.append( SingleIntegrator2D(np.array([3.5,0.8]), dt, ax, num_robots=num_robots, id = 5, color='g',palpha=alpha ) )
U_nominal = np.zeros((2,num_robots))

obstacles = []
obstacles.append( circle( 1.8,2.5,1.0,ax,0 ) ) # x,y,radius, ax, id
obstacles.append( circle( 4.2,2.5,1.0,ax,1 ) )
obstacles.append( circle( 6.2,2.5,1.0,ax,2 ) )

############################## Optimization problems ######################################

###### 1: CBF Controller
## u1: control input
## A1, b1: All inequality constraints are expressed in form A1 u <= b1
## Constraints: collision avoidance with leader and every other robot + connectivity constraint

u1 = cp.Variable((2,1))
u1_ref = cp.Parameter((2,1),value = np.zeros((2,1)) )
num_constraints1  = num_robots - 1 + num_adversaries + num_obstacles + num_connectivity 
A1 = cp.Parameter((num_constraints1,2),value=np.zeros((num_constraints1,2)))
b1 = cp.Parameter((num_constraints1,1),value=np.zeros((num_constraints1,1)))
slack_constraints1 = cp.Parameter( (num_constraints1,1), value = np.zeros((num_constraints1,1)) )
const1 = [A1 @ u1 <= b1 + slack_constraints1]
objective1 = cp.Minimize( cp.sum_squares( u1 - u1_ref  ) )
cbf_controller = cp.Problem( objective1, const1 )


###### 2: A copy of CBF controller with control input bounds to check if constraints are compatible and then try removing some if it is infeasible
u2 = cp.Variable( (2,1) )
Q2 = cp.Parameter( (1,2), value = np.zeros((1,2)) )
num_constraints2 = num_robots - 1 + num_adversaries + num_obstacles + num_connectivity
A2 = cp.Parameter((num_constraints2,2),value=np.zeros((num_constraints2,2)))
b2 = cp.Parameter((num_constraints2,1),value=np.zeros((num_constraints2,1)))
slack_constraints2 = cp.Parameter( (num_constraints2,1), value = np.zeros((num_constraints1,1)) )
const2 = [A2 @ u2 <= b2 + slack_constraints2]
const2 += [ cp.abs( u2[0,0] ) <= 7.0 ]
const2 += [ cp.abs( u2[1,0] ) <= 7.0 ]
objective2 = cp.Minimize( Q2 @ u2 )
best_controller = cp.Problem( objective2, const2 )

##########################################################################################
       
metadata = dict(title='Movie Test', artist='Matplotlib',comment='Movie support!')
writer = FFMpegWriter(fps=15, metadata=metadata)

tp = []

with writer.saving(fig, movie_name, 100): 

    for i in range(num_steps):
        
        const_index = 0
            
        # Move nominal agents
        for j in range(num_robots):
            # u_nominal = np.array([1.0,0.0])
            u_nominal = np.array([0.0,1.0])
            robots_nominal[j].step( u_nominal )
            V, dV_dx = robots[j].lyapunov(robots_nominal[j].X)
            robots[j].x_dot_nominal = -1.0*dV_dx.T/np.linalg.norm(dV_dx) # 3.0
            robots[j].U_ref = robots[j].nominal_input( robots_nominal[j] )
            robots_nominal[j].render_plot()
        
        #  Get inequality constraints
        for j in range(num_robots):
            
            const_index = 0
                
            # obstacles
            for k in range(num_obstacles):
                h, dh_dxi, dh_dxk = robots[j].agent_barrier(obstacles[k], d_min_obstacles);  
                robots[j].obs_h[0,k] = h
                
                # Control QP constraint
                robots[j].A1[const_index,:] = dh_dxi @ robots[j].g()
                robots[j].b1[const_index] = -dh_dxi @ robots[j].f() - robots[j].obs_alpha[0,k] * h
            
                const_index = const_index + 1
                
            # Min distance constraint
            for k in range(num_robots):
                
                if k==j:
                    continue
                
                h, dh_dxj, dh_dxk = robots[j].agent_barrier(robots[k], d_min_agents)
                robots[j].robot_h[0,k] = h
                if h < 0:
                    robots[j].slack_constraint[const_index,0] = 0.0
                    
                # Control QP constraint
                robots[j].A1[const_index,:] = dh_dxj @ robots[j].g()
                robots[j].b1[const_index] = -dh_dxj @ robots[j].f() - dh_dxk @ ( robots[k].f() + robots[k].g() @ robots[k].U ) - robots[j].robot_alpha[0,k] * h

                const_index = const_index + 1
                
            # Max distance constraint for connectivity            
            if j!=0:                
                h, dh_dxj, dh_dxk = robots[j].connectivity_barrier(robots[0], d_max)
                robots[j].robot_connectivity_h = h
                if h < 0:
                    robots[j].slack_constraint[const_index,0] = 0.0
                
                # Control QP constraint
                robots[j].A1[const_index,:] = dh_dxj @ robots[j].g()
                robots[j].b1[const_index] = -dh_dxj @ robots[j].f() - dh_dxk @ ( robots[0].f() + robots[0].g() @ robots[0].U ) - robots[j].robot_connectivity_alpha[0,0] * h
                
                const_index = const_index + 1
                
            
            
        # Design control input and update alphas with trust
        for j in range(num_robots):
            
            const_index = 0      
            # Constraints in LP and QP are same      
            A1.value = robots[j].A1
            A2.value = robots[j].A1
            b1.value = robots[j].b1
            b2.value = robots[j].b1
            slack_constraints1.value = robots[j].slack_constraint
            slack_constraints2.value = robots[j].slack_constraint
            
            # Check if constraints are compatible and remove them by adding a huge slack if required to make them compatible         
            if check_for_constraints:
                    
                # Min distance (collision avoidance constraints)
                const_index = num_obstacles
                for k in range(num_robots):
                    if k==j:
                        continue
                    
                    if robots[j].slack_constraint[const_index,0] > bigNaN * 0.99:
                        const_index += 1
                        continue
                
                    best_controller.solve(solver=cp.GUROBI)#, verbose=True)
                    if best_controller.status!='optimal':
                        print(f"LP status:{best_controller.status}. Removing a constraint")
                        robots[j].slack_constraint[-1,0] = bigNaN
                        slack_constraints1.value = robots[j].slack_constraint
                        slack_constraints2.value = robots[j].slack_constraint
                        best_controller.solve(solver=cp.GUROBI)
                        if best_controller.status!='optimal':
                            for kk in range(num_constraints2-1):
                                if b2.value[kk,0] < 0:
                                    robots[j].slack_constraint[kk,0] = bigNaN
                            slack_constraints1.value = robots[j].slack_constraint
                            slack_constraints2.value = robots[j].slack_constraint
                            best_controller.solve(solver=cp.GUROBI)
                            if best_controller.status!='optimal':
                                print(f"serious ERROR")
                                exit()
                    
                            
                    h, dh_dxi, dh_dxk = robots[j].agent_barrier(robots[k], d_min_agents);
                    if  robots[j].slack_constraint[const_index,0] > bigNaN * 0.99:   
                        const_index += 1  
                        continue
                    assert(h<0.01)
                        
                    const_index += 1
                        
                # Max distance (connectivity constraint)
                if j!=0 and robots[j].slack_constraint[-1,0] < bigNaN * 0.99:
                    best_controller.solve(solver=cp.GUROBI)#, verbose=True)
                    if best_controller.status!='optimal':
                        print(f"LP status:{best_controller.status}. Removing a constraint")
                        robots[j].slack_constraint[-1,0] = bigNaN
                        slack_constraints1.value = robots[j].slack_constraint
                        slack_constraints2.value = robots[j].slack_constraint
                        best_controller.solve(solver=cp.GUROBI)
                        if best_controller.status!='optimal':
                            for kk in range(num_constraints2-1):
                                if b2.value[kk,0] < 0:
                                    robots[j].slack_constraint[kk,0] = bigNaN
                            slack_constraints1.value = robots[j].slack_constraint
                            slack_constraints2.value = robots[j].slack_constraint
                            best_controller.solve(solver=cp.GUROBI)
                            if best_controller.status!='optimal':
                                print(f"serious ERROR")
                                exit()
                    
                    if  robots[j].slack_constraint[-1,0] > bigNaN * 0.99:     
                        continue
                    h, dh_dxi, dh_dxk = robots[j].connectivity_barrier(robots[0], d_max);
                    assert(h<0.01) # h should be negative. if it became positive, then we violated the constraint and need to adjust dt(time step)
                
            # Solve for control input
            u1_ref.value = robots[j].U_ref
            cbf_controller.solve(solver=cp.GUROBI)#, verbose=True)
            if cbf_controller.status!='optimal':
                print(f"{j}'s input: {cbf_controller.status}")
                exit()
            robots[j].nextU = u1.value       

         
        for j in range(num_robots):
            robots[j].step( robots[j].nextU )
            robots[j].render_plot()
            # print(f"{j} state: {robots[j].X[1,0]}, input:{robots[j].nextU[0,0]}, {robots[j].nextU[1,0]}")
        
        t = t + dt
        tp.append(t)
        
        fig.canvas.draw()
        fig.canvas.flush_events()
        
        writer.grab_frame()
    
plt.ioff()   
