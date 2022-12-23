import numpy as np
import cvxpy as cp

# Standard unweighted Laplacian construction (not used in simulation result. here only for reference)
def connectivity_undirected_laplacian(robots, max_dist):
    
    # Adjacency Matrix
    A = np.zeros( ( len(robots), len(robots) ) )    
    for i in range( len(robots) ):
        for j in range( i, len(robots) ):
            if np.linalg.norm( robots[i].X[0:2] - robots[j].X[0:2] ) < max_dist:
            # or any other criteria
                A[i, j] = 1
                A[j, i] = 1
                
    # Degree matrix
    D = np.diag( np.sum( A, axis = 1 ) )
    
    # Laplacian Matrix
    L = D - A
    return L

# Standard Weighted Laplacian construction (not used in simulation result. here only for reference)
# Does not give priority to leader when deciding on weights
def weighted_connectivity_undirected_laplacian(robots, max_dist = 1.0):
    
    # thresholds
    rho =  1.0 #1.0 #0.5
    gamma = 0.5
    
    # Adjacency Matrix
    A = np.zeros( ( len(robots), len(robots) ) )
    
    for i in range( len(robots) ):
        robots[i].dL_dx = np.zeros( ( len(robots), len(robots), np.shape(robots[i].X)[0] ) )
        robots[i].dA_dx = np.zeros( ( len(robots), len(robots), np.shape(robots[i].X)[0] ) )
    
    for i in range( len(robots) ):
        
        for j in range( i+1, len(robots) ):
            
            # weight
            dist = np.linalg.norm( robots[i].X[0:2] - robots[j].X[0:2] )
            
            # weight gradient
            d_dist_dxi = 1.0/dist * (robots[i].X[0:2] - robots[j].X[0:2] ).reshape(1,-1)
            d_dist_dxj = - 1.0/dist * (robots[i].X[0:2] - robots[j].X[0:2] ).reshape(1,-1)
            
            # derivative w.r.t state
            der_i = np.array([0,0]).reshape(1,-1)
            der_j = np.array([0,0]).reshape(1,-1)
                
            if dist <= rho:
                A[i , j] = 1.0                
            elif dist >= max_dist:
                A[i, j] = 0.0
            else:
                A[i, j] = np.exp( -gamma * (dist-rho) / (max_dist-rho)  )
                der_i = A[i , j] * ( -gamma/(max_dist-rho) * d_dist_dxi )
                der_j = A[i , j] * ( -gamma/(max_dist-rho) * d_dist_dxj )
            # or any other criteria
            A[j, i] = A[i, j]
            
            # i's Adjacency derivatives
            robots[i].dA_dx[i,j,:] = der_i
            robots[i].dA_dx[j,i,:] = der_i
            
            # j's Adjacency derivatives
            robots[j].dA_dx[i,j,:] = der_j
            robots[j].dA_dx[j,i,:] = der_j
            
            # Laplacian Derivatives
            robots[i].dL_dx[i,j,:] = - robots[i].dA_dx[i,j,:]
            robots[i].dL_dx[j,i,:] = - robots[i].dA_dx[j,i,:]
            robots[i].dL_dx[i,i,:] = robots[i].dL_dx[i,i,:] + robots[i].dA_dx[i,j,:]
            robots[i].dL_dx[j,j,:] = robots[i].dL_dx[j,j,:] + robots[i].dA_dx[j,i,:]
            
            robots[j].dL_dx[i,j,:] = - robots[j].dA_dx[i,j,:]
            robots[j].dL_dx[j,i,:] = - robots[j].dA_dx[j,i,:]
            robots[j].dL_dx[i,i,:] = robots[j].dL_dx[i,i,:] + robots[j].dA_dx[i,j,:]
            robots[j].dL_dx[j,j,:] = robots[j].dL_dx[j,j,:] + robots[j].dA_dx[j,i,:]
            
    # Degree matrix
    D = np.diag( np.sum( A, axis = 1 ) )
    
    # Laplacian Matrix
    L = D - A
    return L

# Weighted Laplacian construction with priority to leader. If distance to leader is more than maximum distance, all edge weights go to zero too
def leader_weighted_connectivity_undirected_laplacian(robots, max_dist = 1.0):
    
    # thresholds
    rho =  1.0 #1.0 #0.5
    gamma = 0.5
    
    # Adjacency Matrix
    A = np.zeros( ( len(robots), len(robots) ) )
    
    for i in range( len(robots) ):
        robots[i].dL_dx = np.zeros( ( len(robots), len(robots), np.shape(robots[i].X)[0] ) )
        robots[i].dA_dx = np.zeros( ( len(robots), len(robots), np.shape(robots[i].X)[0] ) )
    
    for i in range( len(robots) ):
        # i=0 is the leader
        
        dist_leader = 0
        for j in range( i+1, len(robots) ):
            
            # weight
            dist = np.linalg.norm( robots[i].X[0:2] - robots[j].X[0:2] )
    
            # weight gradient
            d_dist_dxi = 1.0/dist * (robots[i].X[0:2] - robots[j].X[0:2] ).reshape(1,-1)
            d_dist_dxj = - 1.0/dist * (robots[i].X[0:2] - robots[j].X[0:2] ).reshape(1,-1)
            
            # derivative w.r.t state
            der_i = np.array([0,0]).reshape(1,-1)
            der_j = np.array([0,0]).reshape(1,-1)
                
            if dist <= rho:
                A[i , j] = 1.0                
            elif dist >= max_dist:
                A[i, j] = 0.0
            else:
                A[i, j] = np.exp( -gamma * (dist-rho) / (max_dist-rho)  )
                der_i = A[i , j] * ( -gamma/(max_dist-rho) * d_dist_dxi )
                der_j = A[i , j] * ( -gamma/(max_dist-rho) * d_dist_dxj )
            
            # Add leader connection weight to this and see what happens!    
            if i>0: # 1,2   
                der_i = der_i * A[i, 0] * A[j, 0] + A[i, j] * robots[i].dA_dx[i,0,:] * A[j, 0]
                der_j = der_j * A[i, 0] * A[j, 0] + A[i, j] * A[i, 0] * robots[j].dA_dx[j,0,:]
                A[i, j] = A[i, j] * A[i, 0] * A[j, 0]
                
            # or any other criteria
            A[j, i] = A[i, j]
            
            # i's Adjacency derivatives
            robots[i].dA_dx[i,j,:] = der_i
            robots[i].dA_dx[j,i,:] = der_i
            
            # j's Adjacency derivatives
            robots[j].dA_dx[i,j,:] = der_j
            robots[j].dA_dx[j,i,:] = der_j
            
            # Laplacian Derivatives
            robots[i].dL_dx[i,j,:] = - robots[i].dA_dx[i,j,:]
            robots[i].dL_dx[j,i,:] = - robots[i].dA_dx[j,i,:]
            robots[i].dL_dx[i,i,:] = robots[i].dL_dx[i,i,:] + robots[i].dA_dx[i,j,:]
            robots[i].dL_dx[j,j,:] = robots[i].dL_dx[j,j,:] + robots[i].dA_dx[j,i,:]
            
            robots[j].dL_dx[i,j,:] = - robots[j].dA_dx[i,j,:]
            robots[j].dL_dx[j,i,:] = - robots[j].dA_dx[j,i,:]
            robots[j].dL_dx[i,i,:] = robots[j].dL_dx[i,i,:] + robots[j].dA_dx[i,j,:]
            robots[j].dL_dx[j,j,:] = robots[j].dL_dx[j,j,:] + robots[j].dA_dx[j,i,:]
            
    # Degree matrix
    D = np.diag( np.sum( A, axis = 1 ) )
    
    # Laplacian Matrix
    L = D - A
    return L

# Compute gradient of second smallest eigenvalue w.r.t state x
def lambda2_dx( robots, L, Lambda2, V2 ):
    dLambda2_dL = V2 @ V2.T / ( V2.T @ V2 )
    
    for i in range( len(robots) ):
        robots[i].lambda2_dx = np.zeros( (1,robots[i].X.shape[0]) )
        for j in range( np.shape( robots[i].X )[0] ):
            robots[i].lambda2_dx[0,j] =  np.trace( dLambda2_dL.T @ robots[i].dL_dx[:,:,j]  )
            # print(f" dl_dx:{ robots[i].lambda2_dx[0,j] } ")
    
  
# Eigenvalue and Eigenvectors of laplacian Matrix: 
def laplacian_eigen( L ):
   Lambda, V = np.linalg.eig(L)  # eigenvalues, right eigenvectors
   eigenvalue_order = np.argsort(Lambda)  # sort the eigenvalues
   Lambda = Lambda[eigenvalue_order]
   V = V[:, eigenvalue_order]
   return Lambda, V
