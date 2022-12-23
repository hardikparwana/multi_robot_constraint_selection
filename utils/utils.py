import numpy as np

def wrap_angle(angle):
    if angle>np.pi:
        return  angle - 2*np.pi
    elif angle<-np.pi:
        return  angle + 2*np.pi 
    else:
        return angle
    
def euler_to_rot_mat(phi,theta,psi):
    return np.array( [ [ np.cos(psi)*np.cos(theta), -np.sin(psi)*np.cos(phi)+np.cos(psi)*np.sin(theta)*np.sin(phi),  np.sin(psi)*np.sin(phi)+np.cos(psi)*np.cos(phi)*np.sin(theta) ],
                       [ np.sin(psi)*np.cos(theta),  np.cos(psi)*np.cos(phi)+np.sin(psi)*np.sin(theta)*np.sin(phi), -np.cos(psi)*np.sin(phi)+np.sin(theta)*np.sin(psi)*np.cos(phi) ],
                       [ -np.sin(theta),             np.cos(theta)*np.sin(phi)                                    ,  np.cos(theta)*np.cos(phi) ]  ] )
     
def euler_rate_matrix(phi,theta,psi):
    return np.array( [ [ 1, np.sin(phi)*np.tan(theta), np.cos(phi)*np.tan(theta) ],
                      [ 0,  np.cos(phi)              , -np.sin(phi) ],
                      [ 0,  np.sin(phi)/np.cos(theta), np.cos(phi)*np.sin(theta) ] ] )
    