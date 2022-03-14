import numpy as np
def stax_transform(x,y,z,theta):
    '''
    Rotation matrix for stereotaxic surgery.
    INPUTS:
    x, y, z coordinates (float)
    theta (polar angle) (float)
    OUTPUT: Rotated x,y,z coordinates for angle theta
    '''
    a = theta*np.pi/180
    r_matrix = np.array([[np.cos(a),-np.sin(a)],\
      [np.sin(a),np.cos(a)]])
    new_x, new_z = np.dot(r_matrix,np.array([[x],[z]]))
    return new_x[0], y, new_z[0]
