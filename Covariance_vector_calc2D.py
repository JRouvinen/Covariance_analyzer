import numpy as np
import scipy.linalg as la
import math

def covariance_vector2D(xx, xy, yy):

    #print("Covariance matrix values")
    # Set array values
    rep_xx = xx
    #print(rep_xx)
    rep_xy = xy
    #print(rep_xy)
    rep_yx = rep_xy
    #print(rep_yx)
    rep_yy = yy
    #print(rep_yy)

    A = np.array([[rep_xx,rep_xy],[rep_yx,rep_yy]])


    #Eigen calculations
    evalue, evec = la.eig(A)

    #convert list with complex into float


    evalue1 = evalue[0]
    evalue2 = evalue[1]

    compl_evalue1 = complex(evalue1)
    compl_evalue2 = complex(evalue2)

    lamda_1 = compl_evalue1
    lamda_2 = compl_evalue2

    #Change lamda complex to float
    lamda_1 = lamda_1.real
    lamda_2 = lamda_2.real

    lamda_1 = math.sqrt(lamda_1)
    lamda_2 = math.sqrt(lamda_2)
    axis_x = 2*(math.sqrt(lamda_1))
    axis_y = 2*(math.sqrt(lamda_2))
    return axis_x, axis_y