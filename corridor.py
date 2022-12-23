import numpy as np
import time
import cvxpy as cp
import matplotlib.pyplot as plt
from robot_models.SingleIntegrator2D import *
from robot_models.obstacles import *
from utils.utils import *
from graph_utils import *

from matplotlib.animation import FFMpegWriter

plt.rcParams.update({'font.size': 15}) #27

# Sim Parameters                  
dt = 0.05
tf = 13.0 #9.0 #5.4#8#4.1 #0.2#4.1
num_steps = int(tf/dt)
t = 0

# Simulation Parameters
d_min_obstacles = 1.0 # collision avoidance radius for obstacles
d_min_agents = 0.5 # collision avoidance radius for other agents
cbf_extra_bad = 0.0
                                                                                                                                                                

update_param = True
bigNaN = 10000000

eigen_alpha = 0.8  # CBF alpha parameter for eiganvalue constraint
alpha_cbf = 7.0 #3.0#2.0 #0.7 #0.8

# Plot                  
plt.ion()
fig = plt.figure()
ax = plt.axes(xlim=(-5,7),ylim=(-5,15)) 
ax.set_xlabel("X")
ax.set_ylabel("Y")


################# Make Obstacles ###############################
obstacles = []
index = 0
x1 = -1.0
x2 = 1.5
radius = 0.6
y_s = 0
y_increment = 0.3
for i in range(int( 10/y_increment )):
    obstacles.append( circle( x1,y_s,radius,ax,0 ) ) # x,y,radius, ax, id
    obstacles.append( circle( x2,y_s,radius,ax,1 ) )
    y_s = y_s + y_increment

y1 = obstacles[-1].X[1,0] 
y2 = y1 + 3.0
x_s = obstacles[-1].X[0,0]
###################################################################


num_obstacles = len(obstacles)
num_connectivity = 0
num_eigen_connectivity = 0
alpha = 0.1

save_plot = False
movie_name = 'long_corridor_single_leader.mp4'

# agents
robots = []
num_robots = 13

y_offset = -0.5
robots.append( SingleIntegrator2D(np.array([0,y_offset]), dt, ax, id = 0, color='r',palpha=1.0, alpha=alpha_cbf, eigen_alpha = eigen_alpha, num_obstacles=num_obstacles ) )
robots.append( SingleIntegrator2D(np.array([0,y_offset - 1.5]), dt, ax, id = 0, color='g',palpha=1.0, alpha=alpha_cbf, eigen_alpha = eigen_alpha, num_robots = num_robots, num_obstacles=num_obstacles, num_eigen_connectivity=num_eigen_connectivity ) )
robots.append( SingleIntegrator2D(np.array([0.0,y_offset - 0.8]), dt, ax, id = 1, color='g',palpha=1.0, alpha=alpha_cbf, eigen_alpha = eigen_alpha, num_robots = num_robots, num_obstacles=num_obstacles, num_eigen_connectivity=num_eigen_connectivity ) )
robots.append( SingleIntegrator2D(np.array([-0.3,y_offset - 0.7]), dt, ax, id = 3, color='g',palpha=1.0, alpha=alpha_cbf, eigen_alpha = eigen_alpha, num_robots = num_robots, num_obstacles=num_obstacles, num_eigen_connectivity=num_eigen_connectivity ) )
robots.append( SingleIntegrator2D(np.array([0.5,y_offset - 1.0]), dt, ax, id = 5, color='g',palpha=1.0, alpha=alpha_cbf, eigen_alpha = eigen_alpha, num_robots = num_robots, num_obstacles=num_obstacles, num_eigen_connectivity=num_eigen_connectivity ) )
robots.append( SingleIntegrator2D(np.array([-1,y_offset - 3.0]), dt, ax, id = 1, color='g',palpha=1.0, alpha=alpha_cbf, eigen_alpha = eigen_alpha, num_robots = num_robots, num_obstacles=num_obstacles, num_eigen_connectivity=num_eigen_connectivity ) )
robots.append( SingleIntegrator2D(np.array([1,y_offset - 3.0]), dt, ax, id = 2, color='g',palpha=1.0, alpha=alpha_cbf, eigen_alpha = eigen_alpha, num_robots = num_robots, num_obstacles=num_obstacles, num_eigen_connectivity=num_eigen_connectivity ) )
robots.append( SingleIntegrator2D(np.array([-0.5,y_offset - 2.0]), dt, ax, id = 1, color='g',palpha=1.0, alpha=alpha_cbf, eigen_alpha = eigen_alpha, num_robots = num_robots, num_obstacles=num_obstacles, num_eigen_connectivity=num_eigen_connectivity ) )
robots.append( SingleIntegrator2D(np.array([0.5,y_offset - 2.0]), dt, ax, id = 2, color='g',palpha=1.0, alpha=alpha_cbf, eigen_alpha = eigen_alpha, num_robots = num_robots, num_obstacles=num_obstacles, num_eigen_connectivity=num_eigen_connectivity ) )
robots.append( SingleIntegrator2D(np.array([-0.7,y_offset - 2.7]), dt, ax, id = 1, color='g',palpha=1.0, alpha=alpha_cbf, eigen_alpha = eigen_alpha, num_robots = num_robots, num_obstacles=num_obstacles, num_eigen_connectivity=num_eigen_connectivity ) )
robots.append( SingleIntegrator2D(np.array([0.7,y_offset - 2.7]), dt, ax, id = 2, color='g',palpha=1.0, alpha=alpha_cbf, eigen_alpha = eigen_alpha, num_robots = num_robots, num_obstacles=num_obstacles, num_eigen_connectivity=num_eigen_connectivity ) )
robots.append( SingleIntegrator2D(np.array([-0.3,y_offset - 3.0]), dt, ax, id = 1, color='g',palpha=1.0, alpha=alpha_cbf, eigen_alpha = eigen_alpha, num_robots = num_robots, num_obstacles=num_obstacles, num_eigen_connectivity=num_eigen_connectivity ) )
robots.append( SingleIntegrator2D(np.array([0.3,y_offset - 3.0]), dt, ax, id = 2, color='g',palpha=1.0, alpha=alpha_cbf, eigen_alpha = eigen_alpha, num_robots = num_robots, num_obstacles=num_obstacles, num_eigen_connectivity=num_eigen_connectivity ) )


# agent nominal version: to see how agents would have moved without any interaction
robots_nominal = []

robots_nominal.append( SingleIntegrator2D(np.array([0,y_offset]), dt, ax, id = 0, color='r',palpha=alpha, alpha=alpha_cbf, eigen_alpha = eigen_alpha, num_obstacles=num_obstacles ) )
robots_nominal.append( SingleIntegrator2D(np.array([0,y_offset - 1.5]), dt, ax, id = 0, color='g',palpha=alpha, alpha=alpha_cbf, eigen_alpha = eigen_alpha, num_robots = num_robots, num_obstacles=num_obstacles, num_eigen_connectivity=num_eigen_connectivity ) )
robots_nominal.append( SingleIntegrator2D(np.array([0.0,y_offset - 0.8]), dt, ax, id = 1, color='g',palpha=alpha, alpha=alpha_cbf, eigen_alpha = eigen_alpha, num_robots = num_robots, num_obstacles=num_obstacles, num_eigen_connectivity=num_eigen_connectivity ) )
robots_nominal.append( SingleIntegrator2D(np.array([-0.3,y_offset - 0.7]), dt, ax, id = 3, color='g',palpha=alpha, alpha=alpha_cbf, eigen_alpha = eigen_alpha, num_robots = num_robots, num_obstacles=num_obstacles, num_eigen_connectivity=num_eigen_connectivity ) )
robots_nominal.append( SingleIntegrator2D(np.array([0.5,y_offset - 1.0]), dt, ax, id = 5, color='g',palpha=alpha, alpha=alpha_cbf, eigen_alpha = eigen_alpha, num_robots = num_robots, num_obstacles=num_obstacles, num_eigen_connectivity=num_eigen_connectivity ) )
robots_nominal.append( SingleIntegrator2D(np.array([-1,y_offset - 3.0]), dt, ax, id = 1, color='g',palpha=alpha, alpha=alpha_cbf, eigen_alpha = eigen_alpha, num_robots = num_robots, num_obstacles=num_obstacles, num_eigen_connectivity=num_eigen_connectivity ) )
robots_nominal.append( SingleIntegrator2D(np.array([1,y_offset - 3.0]), dt, ax, id = 2, color='g',palpha=alpha, alpha=alpha_cbf, eigen_alpha = eigen_alpha, num_robots = num_robots, num_obstacles=num_obstacles, num_eigen_connectivity=num_eigen_connectivity ) )
robots_nominal.append( SingleIntegrator2D(np.array([-0.5,y_offset - 2.0]), dt, ax, id = 1, color='g',palpha=alpha, alpha=alpha_cbf, eigen_alpha = eigen_alpha, num_robots = num_robots, num_obstacles=num_obstacles, num_eigen_connectivity=num_eigen_connectivity ) )
robots_nominal.append( SingleIntegrator2D(np.array([0.5,y_offset - 2.0]), dt, ax, id = 2, color='g',palpha=alpha, alpha=alpha_cbf, eigen_alpha = eigen_alpha, num_robots = num_robots, num_obstacles=num_obstacles, num_eigen_connectivity=num_eigen_connectivity ) )
robots_nominal.append( SingleIntegrator2D(np.array([-0.7,y_offset - 2.7]), dt, ax, id = 1, color='g',palpha=alpha, alpha=alpha_cbf, eigen_alpha = eigen_alpha, num_robots = num_robots, num_obstacles=num_obstacles, num_eigen_connectivity=num_eigen_connectivity ) )
robots_nominal.append( SingleIntegrator2D(np.array([0.7,y_offset - 2.7]), dt, ax, id = 2, color='g',palpha=alpha, alpha=alpha_cbf, eigen_alpha = eigen_alpha, num_robots = num_robots, num_obstacles=num_obstacles, num_eigen_connectivity=num_eigen_connectivity ) )
robots_nominal.append( SingleIntegrator2D(np.array([-0.3,y_offset - 3.0]), dt, ax, id = 1, color='g',palpha=alpha, alpha=alpha_cbf, eigen_alpha = eigen_alpha, num_robots = num_robots, num_obstacles=num_obstacles, num_eigen_connectivity=num_eigen_connectivity ) )
robots_nominal.append( SingleIntegrator2D(np.array([0.3,y_offset - 3.0]), dt, ax, id = 2, color='g',palpha=alpha, alpha=alpha_cbf, eigen_alpha = eigen_alpha, num_robots = num_robots, num_obstacles=num_obstacles, num_eigen_connectivity=num_eigen_connectivity ) )



U_nominal = np.zeros((2,num_robots))


############################## Optimization problems ######################################
### Use cvxpy to make QP controller


###### 1: CBF Controller
u1 = cp.Variable((2,1))
u1_ref = cp.Parameter((2,1),value = np.zeros((2,1)) )
num_constraints1  = num_robots - 1 + num_obstacles + num_eigen_connectivity
A1 = cp.Parameter((num_constraints1,2),value=np.zeros((num_constraints1,2)))
b1 = cp.Parameter((num_constraints1,1),value=np.zeros((num_constraints1,1)))
slack_constraints1 = cp.Parameter( (num_constraints1,1), value = np.zeros((num_constraints1,1)) )
const1 = [A1 @ u1 <= b1 + slack_constraints1]
objective1 = cp.Minimize( cp.sum_squares( u1 - u1_ref  ) )
cbf_controller = cp.Problem( objective1, const1 )

###### 3: CBF Controller relaxed
# If the CBF controller has no solution, then solve the relaxed version
u3 = cp.Variable((2,1))
u3_ref = cp.Parameter((2,1),value = np.zeros((2,1)) )
num_constraints3  = num_robots - 1 + num_obstacles + num_eigen_connectivity
A3 = cp.Parameter((num_constraints3,2),value=np.zeros((num_constraints3,2)))
b3 = cp.Parameter((num_constraints3,1),value=np.zeros((num_constraints3,1)))
slack_constraints3 = cp.Variable( (num_constraints3,1) )
const3 = [A3 @ u3 <= b3 + slack_constraints3 ]
objective3 = cp.Minimize( cp.sum_squares( u3 - u3_ref  ) + 1000 * cp.sum_squares( slack_constraints3 ) )
cbf_controller_relaxed = cp.Problem( objective3, const3 )

###########################################################################################
       
metadata = dict(title='Movie Test', artist='Matplotlib',comment='Movie support!')
writer = FFMpegWriter(fps=15, metadata=metadata)

tp = []

with writer.saving(fig, movie_name, 100): 

    for i in range(num_steps):
        
        # Laplacina for connectivity
        L = leader_weighted_connectivity_undirected_laplacian(robots, max_dist = 6.0)
        Lambda, V = laplacian_eigen( L )
        print(f" Eigen value:{ Lambda[1] }")
        lambda2_dx( robots, L, Lambda[1], V[:,1].reshape(-1,1) )
        
        # weighted
        Lambda, V = laplacian_eigen( L )
        
        const_index = 0
            
        # Move nominal agents. This is how agents would move without any interaction with other agents or obstacles
        for j in range(num_robots):
            u_nominal = np.array([0.0,1.0])
            robots_nominal[j].step( u_nominal )
            V, dV_dx = robots[j].lyapunov(robots_nominal[j].X)
            robots[j].x_dot_nominal = -1.0*dV_dx.T/np.linalg.norm(dV_dx)
            robots[j].U_ref = robots[j].nominal_input( robots_nominal[j] )
            robots_nominal[j].render_plot()
        
        #  Get inequality constraints for QP controller
        for j in range(num_robots):
            
            if j==0:
                continue
            
            const_index = 0
                
            # obstacles collision avoidance CBF constraint
            for k in range(num_obstacles):
                h, dh_dxi, dh_dxk = robots[j].agent_barrier(obstacles[k], d_min_obstacles);  
                robots[j].obs_h[0,k] = h
                
                # Control QP constraint
                robots[j].A1[const_index,:] = dh_dxi @ robots[j].g()
                robots[j].b1[const_index] = -dh_dxi @ robots[j].f() - robots[j].obs_alpha[0,k] * h
        
                const_index = const_index + 1
                
                
            # Robot collision avoidance CBF constraint
            for k in range(num_robots):
                
                if k==j:
                    continue
                
              
                h, dh_dxj, dh_dxk = robots[j].agent_barrier(robots[k], d_min_agents)
                robots[j].robot_h[0,k] = h
                if h < 0:
                    robots[j].slack_constraint[const_index,0] = 0.0
                    
                # Control QP constraint
                robots[j].A1[const_index,:] = dh_dxj @ robots[j].g()
                robots[j].b1[const_index] = -dh_dxj @ robots[j].f() - dh_dxk @ ( robots[k].f() + robots[k].g() @ robots[k].U ) - cbf_extra_bad - robots[j].robot_alpha[0,k] * h
         
                const_index = const_index + 1
           
            #add connectivity constraint from eigenvalue
            #for k in range(num_robots): # need lambda2>>0  (opposite of CBF definition in this code)
            if num_eigen_connectivity>0:
                dLambda_dxj = robots[j].lambda2_dx.reshape(1,-1)   # assuming single integrator right now
                robots[j].A1[const_index,:] = -dLambda_dxj @ robots[j].g()
                robots[j].b1[const_index] = dLambda_dxj @ robots[j].f() + robots[j].eigen_alpha * Lambda[1] - lambda_thr * robots[j].eigen_alpha
                for k in range(num_robots):
                    if j==k:
                        continue
                    dLambda_dxk = robots[k].lambda2_dx.reshape(1,-1)
                    robots[j].b1[const_index] = robots[j].b1[const_index] + ( dLambda_dxk @ (robots[k].f() +  robots[k].g() @ robots[k].U ) )
                const_index = const_index + 1
                
        # Design control input and update alphas with trust
        for j in range(num_robots):
            if j==0:
                u1_ref.value = robots[j].U_ref
                robots[j].nextU = u1_ref.value      
            else:   
                const_index = 0      
                # Constraints in LP and QP are same      
                A1.value = robots[j].A1
                b1.value = robots[j].b1
                slack_constraints1.value = robots[j].slack_constraint
                
                # Solve for control input                
                u1_ref.value = 60*robots[j].lambda2_dx.reshape(-1,1) # move in direction that improves eigenvalue
                if 0:
                    u1_ref.value = robots[j].U_ref
                if 0:
                    print(f" j:{j}, u1ref:{ u1_ref.value }, lambadref:{ 2*robots[j].lambda2_dx.reshape(1,-1) } ")
                cbf_controller.solve(solver=cp.GUROBI, reoptimize=True)#, verbose=True)
                    
                robots[j].nextU = u1.value 
                if cbf_controller.status!='optimal':
                    
                    # solve relaxed problem
                    A3.value = A1.value
                    b3.value = b1.value
                    u3_ref.value = u1_ref.value
                    cbf_controller_relaxed.solve(solver=cp.GUROBI, reoptimize=True)
                    if cbf_controller_relaxed.status!='optimal':
                        print("Should not happen!")                        
                        exit()
                    
                    robots[j].nextU = u3.value 
         
        for j in range(num_robots):
            robots[j].step( robots[j].nextU )
            robots[j].render_plot()
        
        t = t + dt
        tp.append(t)
        
        fig.canvas.draw()
        fig.canvas.flush_events()
        
        writer.grab_frame()
    
plt.ioff()   
