

#Imports
import threading
from datetime import datetime
from datetime import date
import ntpath
import os.path
import numpy as np
from os import path
import PySimpleGUI as sg
import math
from time import sleep
import scipy.linalg as la
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Ellipse
from numpy.core.defchararray import rsplit
from math import pi, cos, sin
from matplotlib import pyplot as plt
import matplotlib.transforms as transforms

def fileanalyzer(fileloc, file_size, window1, run):
    #print('fileanalyzer')
    hdg = ''
    time_tag = ''
    time_tag_old = ''
    data_set_read = 0
    lines_read = 0
    end_of_file = False
    file_to_process = fileloc
    # Last line determine
    lines_in_file = file_size
    xx = 0
    xy = 0
    yy = 0

    while lines_in_file != lines_read:
    #while run:

        # os.system('cls')
        covariance_data = track_data_reader(file_to_process, data_set_read, xx, xy, yy, hdg, time_tag,
                                            end_of_file, lines_in_file)
        # returns ref_set_read, track_nmb, kinematics_xx, kinematics_xy, kinematics_yy, kinematics_hdg, time_tag, end_of_file, line_read
        lines_read = covariance_data[8]
        data_set_read = int(covariance_data[0]) + 1
        #print('------------Data set ' + str(data_set_read) + "-------------")

        window1['dataset'].update(data_set_read)
        time_tag = covariance_data[6]
        update_rate = update_rate_resolver(time_tag, time_tag_old)
        track_id = covariance_data[1]
        #print(track_id)
        window1['track_id'].update(track_id)
        #print('Time tag: ' + str(time_tag))
        window1['timetag'].update(time_tag)
        time_tag_old = time_tag
        #print('Update rate: ' + str(update_rate))
        window1['update_rate'].update(update_rate)
        window1['trk_hdg'].update(covariance_data[5])
        end_of_file = covariance_data[7]
        xx = covariance_data[2]
        xy = covariance_data[3]
        yx = xy
        yy = covariance_data[4]
        xx_wr = round(float(xx),5)
        yy_wr = round(float(yy),5)
        xy_wr = round(float(xy), 5)
        window1['xx_value1'].update(xx_wr)
        window1['yy_value1'].update(yy_wr)
        window1['xy_value1'].update(xy_wr)
        vector_data = covariance_vector(xx, xy, yx, yy)
        x_result = str(vector_data[0])
        y_result = str(vector_data[1])
        angle_data = vector_angle_calc(xx, xy, yx, yy)
        #window['print1'].print('Angle data: ' + str(angle_data))
        #window['print1'].print('X = ' + x_result + ' ft , Y = ' + y_result + ' ft')
        #print('X = ' + x_result + ' ft , Y = ' + y_result + ' ft')
        #print('Ellipse Angle: ' + str(angle_data))
        #print('Track heading: ' + str(covariance_data[5]))
        #print("-----------------------------------------")
        # Create multiline data to write
        window1['print1'].print('------------Data set ' + str(data_set_read) + "-------------" + '\n' +
                               'Time tag: ' + str(time_tag) + '\n' +
                               'Track ID: ' + str(track_id) + '\n' +
                               'Update rate: ' + str(update_rate) + '\n' +
                               'Heading: ' + str(covariance_data[5]) + '\n' +
                               'XX: ' + str(xx_wr) + ' XY:'+str(xy_wr) +' YY:' + str(xx_wr)+'\n'+
                               'Ellipse angle: ' + str(angle_data) + '\n' +
                               'Bounding box (X/Y): ' + str(x_result)+' '+units+ ' ' + str(y_result)+' '+units
                               )
        #draw_ellipse(xx, xy,yy,yx)
        sleep(file_reader_sleep)
        window1.write_event_value('-THREAD-', (threading.current_thread().name))

        if end_of_file:
            return 'All data processed'
            run = False
            break
        run = False

def filereader(fileloc):
    try:
        initfile = []
        f = open(fileloc, "r")
        file = f.readlines()
        line_num = 0
        for x in file:
            x = x.strip()
            initfile.append(x)
            line_num = line_num + 1
        window['print'].print(str(line_num)+" words in loadfile read.")
        f.close()
        return initfile

    except Exception as e:
        window['print'].print("ERROR: " + str(e))

def manualanalyzer():
    print('manualanalyzer')

def covariance_vector(xx,xy,yy,yx):

    #print("Covariance matrix values")
    # Set array values
    rep_xx = xx
    #print(rep_xx)
    rep_xy = xy
    #print(rep_xy)
    rep_yx = yx
    #print(rep_yx)
    rep_yy = yy
    #print(rep_yy)

    A = np.array([[rep_xx,rep_xy],[rep_yx,rep_yy]])
    #print('xx = '+str(xx))
    #print('xy = '+str(xy))
    #print('yx = '+str(yx))
    #print('yy = '+str(yy))


    #Eigen calculations
    results = la.eig(A)
    eigenvalues_result_str = str(results[0])
    eigenvalues_list = eigenvalues_result_str.split('j')
    #Get eigenvectors
    eigenvectors_result_str = str(results[1])
    eigenvectors_result_str = eigenvectors_result_str.split(' ')
    eigenvectors_result_list = covariance_eigen_vector_cleaner(eigenvectors_result_str)
    eigenvectors_result_str_xx = eigenvectors_result_list[0]
    eigenvectors_result_str_xy = eigenvectors_result_list[1]
    eigenvectors_result_str_yx = eigenvectors_result_list[2]
    eigenvectors_result_str_yy = eigenvectors_result_list[3]
    #print('Eigenvectors')
    #print(eigenvectors_result_list)
    #Clean and cut eigenvalues into string
    lamda_1 = eigenvalues_list[0]
    first_cut = lamda_1.find('[')
    second_cut = lamda_1.find('+')
    lamda_1 = lamda_1[first_cut+1:second_cut]
    lamda_1 = lamda_1.strip(" ")
    end_check = lamda_1.endswith('.')
    if end_check:
        lamda_1 = lamda_1.strip(".")
    #print("Lamda1 "+lamda_1)
    lamda_2 = eigenvalues_list[1]
    second_cut = lamda_2.find('+')
    lamda_2 = lamda_2[:second_cut]
    lamda_2 = lamda_2.strip(" ")
    end_check = lamda_2.endswith('.')
    if end_check:
        lamda_2 = lamda_2.strip(".")

    #print("Lamda2 "+lamda_2)

    lamda_1 = float(lamda_1)
    lamda_2 = float(lamda_2)

    # Change units from m2 to m
    lamda_1 = math.sqrt(lamda_1)
    lamda_2 = math.sqrt(lamda_2)
    axis_x = 2*(math.sqrt(lamda_1))
    axis_y = 2*(math.sqrt(lamda_2))
    #print('x m: ' + str(axis_x))
    #print('y m:' + str(axis_y))

    #print("------Axis calculations 2D------")
    #print('--------------m-------------')
    #print('x = '+str(axis_x)+'m')
    #print('y = '+str(axis_y)+'m')

    #print('--------------ft-------------')
    #print('x = '+str(axis_x*3.281)+'ft')
    #print('y = '+str(axis_y*3.281)+'ft')
    # Change units and return values
    if metric_units:
        axis_x = round(axis_x, 5)
        axis_y = round(axis_y, 5)
        return axis_x, axis_y
    else:
        axis_x = axis_x*3.281
        axis_x = round(axis_x, 5)
        axis_y = axis_y*3.281
        axis_y = round(axis_y, 5)
        return axis_x, axis_y


def covariance_eigen_vector_cleaner(eigenvectors_result_str):
    eigenvector_list = []
    for i in eigenvectors_result_str:
        i = i.strip(' ')
        i = i.strip('\n')
        i = i.strip('[')
        i = i.strip('[')
        i = i.strip(']')
        i = i.strip(']')
        """ Returns True is string is a number. """
        try:
            float(i)
            numeric = True
        except ValueError:
            numeric = False

        if numeric:
            add_0 = '0'
            i_lenght = len(i)
            point_place = i.find('.')
            if i_lenght == point_place+1:
                i = i+add_0
            eigenvector_list.append(i)
    return eigenvector_list

def track_data_reader(file_to_process, data_set_read, xx, xy, yy, hdg, time_tag,
                                                end_of_file, lines_in_file):
    # Ask for file name
    kinematics_xx = xx
    kinematics_xy = xy
    kinematics_yy = yy
    file_name = file_to_process
    track_nmb = ''
    kinematics_hdg = hdg
    time_tag = ''
    ref_set_read = data_set_read
    loc_set_read = 0
    line_read = 0
    #Last line determine
    #lines_in_file = file_size_resolver(file_name)
    # Open file
    f = open(file_name, "r")

    while line_read != lines_in_file:
        for x in f:
            line_read = line_read+1
            # read last file line
            file_last_line = x[:25]
            if line_read == lines_in_file:
                end_of_file = True
                #print('End of file set')
                track_nmb = 00000
                kinematics_xx = 0
                kinematics_yy = 0
                kinematics_xy = 0
                kinematics_hdg = 0
                return ref_set_read, track_nmb, kinematics_xx, kinematics_xy, kinematics_yy, kinematics_hdg, time_tag, \
                       end_of_file, line_read
            # read empty line
            line_id_last_line = x[:22]
            #print(line_id_last_line)
            if line_id_last_line == 'SecurityClassification':
                loc_set_read = loc_set_read+1
                #print('Last line of set: ' + str(ref_set_read))
                if loc_set_read > ref_set_read:
                    f.close()
                    return ref_set_read, track_nmb, kinematics_xx, kinematics_xy, kinematics_yy, kinematics_hdg, \
                           time_tag, end_of_file, line_read
            # Read track data time tag
            line_id_Timetag = x
            line_id_Timetag_bool = line_id_Timetag.find('UPDATE')
            if line_id_Timetag_bool != -1:
                time_tag = x
                time_tag = time_tag[:25]
                time_tag = time_tag.strip('\n')
                time_tag = time_tag.strip('[')
                time_tag = time_tag.strip(']')
                time_tag = time_tag[-12:]

            # Read MST Track ID
            line_id_MSTTrack = x[:8]
            #print(line_id_MSTTrack)
            if line_id_MSTTrack == 'MSTTrack':
                track_nmb = x
                track_nmb = track_nmb.strip('\n')
                #print('Track id found: '+track_nmb)
            # Read Covariance
            line_id_Kinematics = x[:10]
            #print(line_id_Kinematics)
            if line_id_Kinematics == 'Kinematics':
                kinematics_data = x
                #print('Kinematics data found')
                #print(kinematics_data)
                xx_loc = kinematics_data.find('xx:')
                #print(xx_loc)
                xy_loc = kinematics_data.find('xy:')
                #print(xy_loc)
                yy_loc = kinematics_data.find('yy:')
                #print(yy_loc)
                hdg_loc = kinematics_data.find('Heading:')
                end_loc = kinematics_data.find('SampleTime:')
                kinematics_xx = kinematics_data[xx_loc:xy_loc]
                kinematics_xx = kinematics_xx[3:]
                kinematics_xx = kinematics_xx.strip(' ')
                kinematics_xy = kinematics_data[xy_loc:yy_loc]
                kinematics_xy = kinematics_xy[3:]
                kinematics_xy = kinematics_xy.strip(' ')
                kinematics_yy = kinematics_data[yy_loc:end_loc]
                kinematics_yy = kinematics_yy[3:]
                kinematics_yy = kinematics_yy.strip(' ')
                kinematics_yy = kinematics_yy[:-2]
                kinematics_hdg = kinematics_data[hdg_loc+8:hdg_loc+14]
                kinematics_hdg = kinematics_hdg.strip(' ')

def update_rate_resolver(time_tag, time_tag_old):

    timestamp1 = time_tag[:-4]
    #print(timestamp1)
    timestamp2 = time_tag_old[:-4]
    #print(timestamp2)
    if timestamp2 == '':
        timestamp2 = timestamp1

    t1 = datetime.strptime(timestamp1, "%H:%M:%S")
    #print(t1)
    t2 = datetime.strptime(timestamp2, "%H:%M:%S")
    #print(t2)
    difference = t1 - t2
    #print(difference.seconds)  # in this case
    return difference

def vector_angle_calc(rep_xx, rep_xy, rep_yx, rep_yy):
    # print('----------Vector calculations-----------')
    A = np.array([[rep_xx, rep_xy],[rep_xy, rep_yy]])
    angle_x = True
    results = la.eig(A)
    eigvals, eigvecs = la.eig(A)
    eigvals = eigvals.real
    #print(eigvals)
    lambda1 = eigvals[0]
    lambda2 = eigvals[1]
    rep_xx = float(rep_xx)
    rep_yy = float(rep_yy)
    rep_xy = float(rep_xy)
    lambda_arc1 = math.atan2(lambda1-rep_xx,rep_xy)
    lambda_arc2 = math.atan2(lambda2-rep_xx,rep_xy)
    angle1 = math.degrees(lambda_arc1)
    angle2 = math.degrees(lambda_arc2)
    #print(str(angle1)+", "+str(angle2))
    return angle1, angle2

def calc_ellipse(xx,xy, yy, yx):
    # Heading parameters
    hdg_x = 15000
    hdg_y = 15000
    angle = 90
    length = 7500
    # print('----------Vector calculations-----------')
    A = np.array([[xx, xy], [xy, yy]])
    eigvals, eigvecs = la.eig(A)
    eigvals = eigvals.real
    lambda1 = eigvals[0]
    lambda2 = eigvals[1]
    rep_xx = float(xx)
    rep_yy = float(yy)
    rep_xy = float(xy)
    lambda_arc1 = math.atan2(lambda1 - rep_xx, rep_xy)
    lambda_arc2 = math.atan2(lambda2 - rep_xx, rep_xy)

    u = 1.  # x-position of the center
    v = 0.5  # y-position of the center
    a = 18562.070652493334  # radius on the x-axis
    b = 23606.251428713906  # radius on the y-axis
    t_rot = pi / 4  # rotation angle

    t = np.linspace(0, 2 * pi, 100)
    Ell = np.array([a * np.cos(t), b * np.sin(t)])
    # u,v removed to keep the same center location
    R_rot = np.array([[cos(t_rot), -sin(t_rot)], [sin(t_rot), cos(t_rot)]])
    # 2-D rotation matrix

    Ell_rot = np.zeros((2, Ell.shape[1]))
    for i in range(Ell.shape[1]):
        Ell_rot[:, i] = np.dot(R_rot, Ell[:, i])

    # find the end point
    endy = hdg_x + length * math.sin(math.radians(angle))
    endx = length * math.cos(math.radians(angle))

    plt.plot([0,endx], [0, endy], color="red")
    #plt.plot(u + Ell[0, :], v + Ell[1, :])  # initial ellipse
    plt.plot(u + Ell_rot[0, :], v + Ell_rot[1, :], 'darkorange')  # rotated ellipse
    plt.grid(color='lightgray', linestyle='--')
    plt.show()
    #return plt

def get_correlated_dataset(n, dependency, mu, scale):
    latent = np.random.randn(n, 2)
    dependent = latent.dot(dependency)
    scaled = dependent * scale
    scaled_with_offset = scaled + mu
    # return x and y of the new, correlated dataset
    return scaled_with_offset[:, 0], scaled_with_offset[:, 1]

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    #if x.size != y.size:
        #raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def draw_ellipse(xx, xy, yy, yx):
    u = 1.  # x-position of the center
    v = 0.5  # y-position of the center
    a = float(xx)  # radius on the x-axis
    b = float(yy)  # radius on the y-axis
    t_rot = pi / 4  # rotation angle
    # Heading parameters
    hdg_x = 15000
    hgd_y = 15000
    angle = 90
    length = 7500

    t = np.linspace(0, 2 * pi, 100)
    Ell = np.array([a * np.cos(t), b * np.sin(t)])
    # u,v removed to keep the same center location
    R_rot = np.array([[cos(t_rot), -sin(t_rot)], [sin(t_rot), cos(t_rot)]])
    # 2-D rotation matrix

    Ell_rot = np.zeros((2, Ell.shape[1]))
    for i in range(Ell.shape[1]):
        Ell_rot[:, i] = np.dot(R_rot, Ell[:, i])

    plt.plot(u + Ell[0, :], v + Ell[1, :])  # initial ellipse
    plt.plot(u + Ell_rot[0, :], v + Ell_rot[1, :], 'darkorange')  # rotated ellipse
    plt.grid(color='lightgray', linestyle='--')
    plt.show()

def draw_figure_and_dataset(xx, xy, yy, yx):
    np.random.seed(0)
    #parameters = {
        #'XX-YY': [[70944.48045008327, -16095.794241829783],
        #                         [0, 57610.15350529482]],
        #'ZZ-YY': [[0.9, -0.4],
                #[0.1, -0.6]]}

    rep_xx = float(xx)
    rep_yy = float(yy)
    rep_xy = float(xy)
    rep_yx = rep_xy

    parameters = {
    'XX-YY': [[rep_xx, rep_xy],
                             [rep_yy, rep_yx]],
     'ZZ-YY': [[0.9, -0.4],
     [0.1, -0.6]]}

    mu = 2, 4
    scale = 3, 5

    fig, axs = plt.subplots(1, 2, figsize=(9, 3))
    for ax, (title, dependency) in zip(axs, parameters.items()):
        x, y = get_correlated_dataset(dataset_size, dependency, mu, scale)
        ax.scatter(x, y, s=0.5)

        ax.axvline(c='grey', lw=1)
        ax.axhline(c='grey', lw=1)

        confidence_ellipse(x, y, ax, edgecolor='red')

        ax.scatter(mu[0], mu[1], c='red', s=3)
        ax.set_title(title)

    plt.show()

def file_size_resolver(file_loc):
    file = file_loc
    #print(file)
    lines_in_file = 0
    # Open file
    f = open(file, "r")
    for x in f:
        lines_in_file = lines_in_file+1

    #print('lines in file: ' + str(lines_in_file))

    data_sets = 0
    line_read = 0
    data_found = 0
    # Last line determine
    # lines_in_file = file_size_resolver(file_name)
    # Open file
    f = open(file, "r")

    while line_read != lines_in_file:
        for x in f:
            if data_found == 4:
                data_found = 0
            line_read = line_read + 1
            # read last file line
            file_last_line = x[:25]
            if line_read == lines_in_file:
                return lines_in_file, data_sets

            # read empty line
            line_id_last_line = x[:22]
            # print(line_id_last_line)
            if line_id_last_line == 'SecurityClassification':
                data_found = data_found+1
                #print(data_found)
            # Read track data time tag
            line_id_Timetag = x
            line_id_Timetag_bool = line_id_Timetag.find('UPDATE')
            if line_id_Timetag_bool != -1:
                data_found = data_found+1
                #print(data_found)
            # Read MST Track ID
            line_id_MSTTrack = x[:8]
            # print(line_id_MSTTrack)
            if line_id_MSTTrack == 'MSTTrack':
                data_found = data_found+1
                #print(data_found)
            # Read Covariance
            line_id_Kinematics = x[:10]
            # print(line_id_Kinematics)
            if line_id_Kinematics == 'Kinematics':
                data_found = data_found+1
                #print(data_found)

            if data_found == 4:
                data_sets = data_sets+1
                #print(data_found)

    return lines_in_file, data_sets

def manual_input_resolver(manual_xx_xy_yy, manual_zz_zx_zy):
    xx_loc = manual_xx_xy_yy.find('XX:')
    xy_loc = manual_xx_xy_yy.find('XY:')
    yy_loc = manual_xx_xy_yy.find('YY:')
    zz_loc = manual_xx_xy_yy.find('ZZ:')
    zx_loc = manual_xx_xy_yy.find('ZX:')
    zy_loc = manual_xx_xy_yy.find('ZY:')
    end_loc = manual_xx_xy_yy
    kinematics_xx = manual_xx_xy_yy[xx_loc:xy_loc]
    kinematics_xx = kinematics_xx[3:]
    kinematics_xx = kinematics_xx.strip(' ')

    kinematics_xy = manual_xx_xy_yy[xy_loc:yy_loc]
    kinematics_xy = kinematics_xy[3:]
    kinematics_xy = kinematics_xy.strip(' ')
    kinematics_yy = manual_xx_xy_yy[yy_loc:]
    kinematics_yy = kinematics_yy[3:]
    kinematics_yy = kinematics_yy.strip(' ')
    #kinematics_yy = kinematics_yy[:-2]
    xx = kinematics_xx
    xy = kinematics_xy
    yx = xy
    yy = kinematics_yy
    #window1['print2'].print(xx,xy,yy)
    return xx,xy,yy,yx

sg.theme('Reddit')   # Add a touch of color

#Define variables
ver = "V0.1"

xx = 0

yy = 0

xx = 0

yx = 0

zz = 0

zx = 0

zy = 0
hdg = ''
time_tag = ''
time_tag_old = ''
file_loc = ""
metric_units = True
units = 'm'
draw1 = False
dataset1 = False
draw2 = True
dataset2 = True
dataset_size = 300
file_reader_sleep = 1
data_file_size = 0
thread_event = '-THREAD-'
file_analyzer_thread_run = False
manual_xx_xy_yy = ''
manual_zz_zx_zy = ''
#cprint = sg.cprint()
# All the stuff inside your window.

def make_win1():

    tab1_layout = [[sg.Text('Loadfile:'), sg.Input(key="importfile", enable_events=True), sg.FileBrowse()],
                   [sg.Button('Calculate', key="calculate1", disabled=False),
                   sg.Text('File reader sleep time:'),
                    sg.InputText(file_reader_sleep, size=(5, 1), key="sleeptime", enable_events=True)],
                    [sg.Text('Dataset:'),sg.Text('',key="dataset", size=(5,1)),
                     sg.Text('Track ID:'),sg.Text('',key="track_id"),
                     sg.Text('Timetag:'),sg.Text('',key="timetag")],
                    [sg.Text('Update rate:'), sg.Text('',key="update_rate"), sg.Text('Track hdg:'), sg.Text('',key="trk_hdg")],
                    [sg.Text('XX:'),sg.Text('',size=(10,1), key="xx_value1"),sg.Text('XY:'), sg.Text('', size=(10, 1), key="xy_value1"),
                     sg.Text('YY:'),sg.Text('',size=(10,1), key="yy_value1")],
                     [sg.Text('ZZ:'),sg.Text('',size=(10,1), key="zz_value1"),
                    sg.Text('ZX:'), sg.Text('', size=(10, 1), key="zx_value1"), sg.Text('ZY:'),
                    sg.Text('', size=(10, 1), key="zy_value1")],
                    [sg.Text('Ellipse angle:', size=(20,1)),sg.Text('', key="ellip_ang"),sg.Text('Ellipse tilt:',size=(20,1)),sg.Text('', key="ellip_tilt")],
                    [sg.Multiline("", size=(70, 10), key="print1", autoscroll=True, reroute_stdout=True,write_only=True)],
                    [sg.Button('Exit')]
                    ]
    tab2_layout = [[sg.Button('Calculate', key="calculate2")],
                   [sg.Text('XX, XY, YY:'), sg.InputText("", size=(70,1), key="xx_xy_yy_values", enable_events=True)],
                   [sg.Text('ZZ, ZX, ZY:'), sg.InputText("", size=(70,1), key="zz_zx_zy_values", enable_events=True)],
                   [sg.Checkbox('Draw covariance area', default=True,enable_events=True, key='draw2')],
                   [sg.Checkbox('Create dataset for drawing', default=True,enable_events=True, key='createdataset2'),
                    sg.Text('Dataset size:'), sg.InputText(dataset_size, size=(5,1), key="dataset_size", enable_events=True)],
                   [sg.Multiline("", size=(70, 5), key="print2")]]

    tab3_layout = [[sg.Text('Units:'), sg.Radio('Imperial', "UNITS", default=True, enable_events=True, key='imperial'),
                 sg.Radio('Metric', "UNITS", default=False, enable_events=True, key='metric')]]



    layout = [[sg.TabGroup([[sg.Tab('File', tab1_layout),
                           sg.Tab('Manual', tab2_layout),
                            sg.Tab('Settings', tab3_layout)
                             ]])]]

    return sg.Window('Main', layout, finalize=True)

def make_win2():

    layout = [[sg.Text('Covariance draw:')], sg.Canvas(size=(150,150), key='canvas')]
    return sg.Window('Draw', layout, location=(800, 600), finalize=True)


# Create the Window
window1, window2 = make_win1(), None

# Event Loop to process "events" and get the "values" of the inputs
while True:
    window, event, values = sg.read_all_windows(timeout=500)
    if event == "importfile":
        fileloc = values["importfile"]
        output = file_size_resolver(fileloc)
        data_file_size = output[0]
        data_sets_found = output[1]
        window['print1'].print(str(data_sets_found)+" datasets found in file")
        #readfile = filereader(fileloc)
    if event == "imperial":
        metric_units = not metric_units
        if metric_units:
            units = 'm'
        else:
            units = 'ft'
    if event == "metric":
        metric_units = not metric_units
        if metric_units:
            units = 'm'
        else:
            units = 'ft'
    if event == 'calculate1':
        file_analyzer_thread_run = True
        #fileanalyzer(fileloc, data_file_size, window1, file_analyzer_thread_run)
        threading.Thread(target=fileanalyzer, args=(fileloc, data_file_size, window1, file_analyzer_thread_run), daemon=True).start()
        #file_analyzer_thread_run = False

    if event == thread_event:
        #draw_ellipse(xx, xy,yy,yx)
        #queued_thread_event_read
        print('thread_event_update')

    if event == 'calculate2':
        values = manual_input_resolver(manual_xx_xy_yy, manual_zz_zx_zy)
        val_xx = values[0]
        val_xy = values[1]
        val_yy = values[2]
        val_yx = values[3]
        output = covariance_vector(val_xx,val_xy,val_yy,val_yx)
        if draw2 and dataset2:
            draw_figure_and_dataset(val_xx,val_xy,val_yy,val_yx)
        elif draw2:
            draw_ellipse(val_xx,val_xy,val_yy,val_yx)
        window['print2'].print(output)
    if event == "draw1":
        draw1 = not draw1

    if event == "createdataset1":
        dataset1 = not dataset1
    if event == "draw2":
        draw2 = not draw2
        #window1['print2'].print(draw2)
    if event == "createdataset2":
        dataset2 = not dataset2
        #window1['print2'].print(dataset2)
    if event == 'dataset_size':
        dataset_size = values["dataset_size"]
        #window1['print2'].print(dataset_size)

    if event == 'sleeptime':
        file_reader_sleep = values["sleeptime"]
    if event == 'xx_xy_yy_values':
        manual_xx_xy_yy = values["xx_xy_yy_values"]
    if event == 'zz_zx_zy_values':
        manual_zz_zx_zy = values["zz_zx_zy_values"]

    if event == sg.WIN_CLOSED or event == 'Exit': # if user closes window or clicks cancel
        break

window.close()




