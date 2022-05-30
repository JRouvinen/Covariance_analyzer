#--List of things to do
#TODO -add debugger options and print lines
#TODO -add posibility to select single track from datafile
#TODO -add popup error window and text for errors
#Imports
import File_analyzer
import Color_cycler
import Covariance_vector_calc2D
import threading
import time
from datetime import datetime
import numpy as np
import PySimpleGUI as sg
import math
from time import sleep
import scipy.linalg as la
from matplotlib.patches import Ellipse, Polygon
import matplotlib.lines as lines
from math import pi, cos, sin
from matplotlib import pyplot as plt
import matplotlib.transforms as transforms

def fileanalyzer(fileloc, file_size, window, run):
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
        covariance_data = track_data_reader(file_to_process, data_set_read, #returns six values: ref_set_read, track_nmb, covariance_dict, time_tag, end_of_file, line_read
                                            end_of_file, lines_in_file)
        # returns ref_set_read, track_nmb, kinematics_xx, kinematics_xy, kinematics_yy, kinematics_hdg, time_tag, end_of_file, line_read
        lines_read = covariance_data[5]
        data_set_read = int(covariance_data[0]) + 1
        #print('------------Data set ' + str(data_set_read) + "-------------")
        covariance_dict = covariance_data[2]

        window['dataset'].update(data_set_read)

        time_tag = covariance_data[3]
        update_rate = update_rate_resolver(time_tag, time_tag_old)
        track_id = covariance_data[1]
        #print(track_id)
        window['track_id'].update(track_id)
        #print('Time tag: ' + str(time_tag))
        window['timetag'].update(time_tag)
        time_tag_old = time_tag
        #print('Update rate: ' + str(update_rate))
        window['update_rate'].update(update_rate)

        window['trk_hdg'].update(covariance_dict["hdg"])
        end_of_file = covariance_data[4]

        #Lat Lon parameters
        xx = covariance_dict["lat_lon_xx"]
        xy = covariance_dict["lat_lon_xy"]
        yy = covariance_dict["lat_lon_yy"]
        yx = xy
        #yy = covariance_dict["lat_lon_yy"]

        xx_wr = round(float(xx),5)
        yy_wr = round(float(yy),5)
        xy_wr = round(float(xy), 5)
        window['xx_value1'].update(xx_wr)
        window['yy_value1'].update(yy_wr)
        window['xy_value1'].update(xy_wr)
        # Alt parameters
        xz = covariance_dict["alt_xz"]
        yz = covariance_dict["alt_yz"]
        zz = covariance_dict["alt_zz"]

        xz_wr = round(float(xz), 5)
        yz_wr = round(float(yz), 5)
        zz_wr = round(float(zz), 5)
        window['xz_value1'].update(xz_wr)
        window['yz_value1'].update(yz_wr)
        window['zz_value1'].update(zz_wr)

        # Vel parameters
        vel_xx = covariance_dict["vel_xx"]
        vel_xy = covariance_dict["vel_xy"]
        vel_yy = covariance_dict["vel_yy"]

        vel_xx_wr = round(float(xz), 5)
        vel_xy_wr = round(float(yz), 5)
        vel_yy_wr = round(float(zz), 5)
        window['vel_xx_value1'].update(vel_xx_wr)
        window['vel_xy_value1'].update(vel_xy_wr)
        window['vel_yy_value1'].update(vel_yy_wr)

        #Lat lon covariance vectors and angle data
        lat_lon_vector_data = Covariance_vector_calc2D.covariance_vector2D(xx, xy, yy) #returns (m)
        lat_lon_x_result = str(lat_lon_vector_data[0])
        lat_lon_y_result = str(lat_lon_vector_data[1])
        lat_lon_angle_data = vector_angle_calc(xx, xy, yy)
        #DONE - add type change between metrics and imperial units
        if metric_units:
            axis_x = float(lat_lon_x_result)
            axis_y = float(lat_lon_y_result)
            lat_lon_x_result = str(round(axis_x, 5))
            lat_lon_y_result = str(round(axis_y, 5))
        else:
            if debug_console:
                window.write_event_value('debug_write', 'Convert X/Y to imperial')
            axis_x = float(lat_lon_x_result)
            axis_y = float(lat_lon_y_result)
            axis_x = axis_x * 3.281
            lat_lon_x_result = str(round(axis_x, 5))
            axis_y = axis_y * 3.281
            lat_lon_y_result = str(round(axis_y, 5))

        window['ellip_x'].update(lat_lon_x_result)
        window['ellip_y'].update(lat_lon_y_result)
        lat_lon_angle = lat_lon_angle_data[1]
        lat_lon_angle = round(lat_lon_angle,1)
        window["ellip_ang"].update(lat_lon_angle)
        #TODO - add conversion between metric and imperial units
        # Alt covariance vectors and angle data
        alt_vector_data = Covariance_vector_calc2D.covariance_vector2D(yy, zy, zz) #returns (m)
        alt_z_result = str(alt_vector_data[0])
        alt_y_result = str(alt_vector_data[1])
        alt_angle_data = vector_angle_calc(xz, zy, yy)

        # TODO - add conversion between metric and imperial units
        # Velocity uncert vectors and angle data
        vel_vector_data = Covariance_vector_calc2D.covariance_vector2D(vel_xx, vel_xy, vel_yy) #returns (m/s)

        vel_x_result = str(vel_vector_data[0])
        vel_y_result = str(vel_vector_data[1])
        vel_angle_data = vector_angle_calc(vel_xx, vel_xy, vel_yy)

        # Create multiline data to write

        if not end_of_file:

            window['print1'].print('------------Data set ' + str(data_set_read) + "-------------" + '\n' +
                                   'Time tag: ' + str(time_tag) + '\n' +
                                   'Track ID: ' + str(track_id) + '\n' +
                                   'Update rate: ' + str(update_rate) + '\n' +
                                   'Heading: ' + str(covariance_dict["hdg"]) + '\n' +
                                   'XX: ' + str(xx_wr) + ' XY:' + str(xy_wr) +' YY:' + str(xx_wr) +'\n' +
                                    ' ZZ:' + str(zz_wr) +' XZ: ' + str(xz_wr) + ' YZ:' + str(yz_wr) +'\n' +
                                   'Lat lon ellipse angle: ' + str(lat_lon_angle_data) + '\n' +
                                    'Alt ellipse angle: ' + str(alt_angle_data) + '\n' +
                                   'Bounding box (X/Y/Z): ' + str(lat_lon_x_result) +' ' + dist_units + ' ' + str(lat_lon_y_result)
                                   + ' ' + dist_units + ' ' + str(alt_z_result) +' '+ dist_units+'\n'+
                                   'Velocity uncert. (X/Y/Z): '+str(vel_x_result)+' '+str(vel_y_result)
                                   )
            if draw_unc_area:
                window.write_event_value('draw_unc_area', 'fileanalyzer_done')
            if draw_cov_ell:
                window.write_event_value('draw_cov_ell', 'fileanalyzer_done')
            if draw_bound_box:
                window.write_event_value('draw_bound_box', 'fileanalyzer_done')

            sleep(int(file_reader_sleep))
            window.write_event_value('-THREAD-', (threading.current_thread().name))
        else:

            data_set = 0

            window['dataset'].update('----')
            window['track_id'].update('----')
            window['timetag'].update('----')
            window['update_rate'].update('----')
            window['trk_hdg'].update('----')
            window['xx_value1'].update('----')
            window['yy_value1'].update('----')
            window['xy_value1'].update('----')
            window['xz_value1'].update('----')
            window['yz_value1'].update('----')
            window['zz_value1'].update('----')
            window['vel_xx_value1'].update('----')
            window['vel_xy_value1'].update('----')
            window['vel_yy_value1'].update('----')
            window['ellip_x'].update('----')
            window['ellip_y'].update('----')
            window["ellip_ang"].update('----')
            run = False
            file_analyzer_thread_run = False
            window.write_event_value('calculate1_enable', 'fileanalyzer_done')
            #print('All data processed')
            window['print1'].print('------------All data processed-------------'+'\n')

            return file_analyzer_thread_run
            break
        run = False
        #file_analyzer_thread_run = False
        #return file_analyzer_thread_run



#This isnt used anymore -- can be cleaned away ----
def filereader(fileloc):
    try:
        initfile = []
        f = open(fileloc, "r")
        window.write_event_value('debug_write', 'file opened')
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
#---- End of to be cleaned ------

def manualanalyzer():
    print('manualanalyzer')

def print_debug(thread_name,run_freq,window2,write,file):

    if debug_console:
        output_window = window2 if window == window1 else window1
        if output_window:
            output_window['debug_con'].print(str(write))
            debug_write_file = file+write+'\n'
            time.sleep(run_freq/1000)
        else:
            sg.Popup('Main window not open')
        #window2['debug_con'].print(write)

def covariance_vector2D(xx, xy, yy): # Can be cleaned

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




def track_data_reader(file_to_process, data_set_read, end_of_file, lines_in_file):
    # Ask for file name
    # Covariance dict
    covariance_dict = {
        "lat_lon_xx": "0",
        "lat_lon_xy": "0",
        "lat_lon_yy": "0",
        "alt_xz": "0",
        "alt_yz": "0",
        "alt_zz": "0",
        "vel_xx": "0",
        "vel_xy": "0",
        "vel_yy": "0",
        "hdg": "0",
    }
    vel_covariance = []
    #old codes
    #kinematics_xx = xx
    #kinematics_xy = xy
    #kinematics_yy = yy
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
                return ref_set_read, track_nmb, covariance_dict, time_tag, end_of_file, line_read
            # read empty line
            line_id_last_line = x[:22]
            #print(line_id_last_line)
            if line_id_last_line == 'SecurityClassification':
                loc_set_read = loc_set_read+1
                #print('Last line of set: ' + str(ref_set_read))
                if loc_set_read > ref_set_read:
                    f.close()
                    return ref_set_read, track_nmb, covariance_dict, time_tag, end_of_file, line_read
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
                track_nmb = track_nmb.strip(' ')
                track_nmb = track_nmb.strip('MSTTrack')
                track_nmb = track_nmb.strip(' (')
                track_nmb = track_nmb.strip(')')
                #print('Track id found: '+track_nmb)
            # Read Covariance
            line_id_Kinematics = x[:10]
            #print(line_id_Kinematics)
            if line_id_Kinematics == 'Kinematics':
                kinematics_data = x
                #print('Kinematics data found')
                #print(kinematics_data)
                lat_lon_xx_loc = kinematics_data.find('xx:')
                #print(xx_loc)
                lat_lon_xy_loc = kinematics_data.find('xy:')
                #print(xy_loc)
                lat_lon_yy_loc = kinematics_data.find('yy:')
                #print(yy_loc)
                hdg_loc = kinematics_data.find('Heading:')
                end_loc = kinematics_data.find('SampleTime:')
                #Get position covariance data
                kinematics_xx = kinematics_data[lat_lon_xx_loc:lat_lon_xy_loc]
                kinematics_xx = kinematics_xx[3:]
                kinematics_xx = kinematics_xx.strip(' ')
                kinematics_xy = kinematics_data[lat_lon_xy_loc:lat_lon_yy_loc]
                kinematics_xy = kinematics_xy[3:]
                kinematics_xy = kinematics_xy.strip(' ')
                kinematics_yy = kinematics_data[lat_lon_yy_loc:end_loc]
                kinematics_yy = kinematics_yy[3:]
                kinematics_yy = kinematics_yy.strip(' ')
                kinematics_yy = kinematics_yy[:-2]
                xx_wr = round(float(xx), 5)
                #Set position covariance data into dictionary
                covariance_dict["lat_lon_xx"] = str(round(float(kinematics_xx), 5))
                covariance_dict["lat_lon_xy"] = str(round(float(kinematics_xy), 5))
                covariance_dict["lat_lon_yy"] = str(round(float(kinematics_yy), 5))
                #Get position covariance data
                kinematics_hdg = kinematics_data[hdg_loc+8:hdg_loc+14]
                kinematics_hdg = kinematics_hdg.strip(' ')
                #Set hdg data into dictionary
                covariance_dict["hdg"] = kinematics_hdg
                #Get altitude covariance data
                alt_cov_loc_start = kinematics_data.find('ALTITUDE / COVARIANCE:')
                if alt_cov_loc_start != -1:
                    alt_cov_data = kinematics_data[alt_cov_loc_start:]
                    alt_cov_loc_end = alt_cov_data.find('],')
                    alt_cov_data = alt_cov_data[:alt_cov_loc_end]
                    alt_cov_data = alt_cov_data.strip('ALTITUDE / COVARIANCE: [')
                    alt_cov_data = alt_cov_data.split(',')
                    alt_cov_xz = str(alt_cov_data[0])
                    alt_cov_xz = alt_cov_xz[3:]
                    alt_cov_xz = alt_cov_xz.strip(',')
                    alt_cov_xz = alt_cov_xz.strip(' ')
                    alt_cov_yz = str(alt_cov_data[1])
                    alt_cov_yz = alt_cov_yz[3:]
                    alt_cov_yz = alt_cov_yz.strip(',')
                    alt_cov_yz = alt_cov_yz.strip(': ')
                    alt_cov_zz = str(alt_cov_data[2])
                    alt_cov_zz = alt_cov_zz[3:]
                    alt_cov_zz = alt_cov_zz.strip(',')
                    alt_cov_zz = alt_cov_zz.strip(': ')
                    #Set altitude covariance data

                    covariance_dict["alt_xz"] = str(round(float(alt_cov_xz), 5))
                    covariance_dict["alt_yz"] = str(round(float(alt_cov_yz), 5))
                    covariance_dict["alt_zz"] = str(str(round(float(alt_cov_zz), 5)))
                else:
                    covariance_dict["alt_xz"] = 0
                    covariance_dict["alt_yz"] = 0
                    covariance_dict["alt_zz"] = 0

                #Get velocity covariance data

                vel_cov_loc_start = kinematics_data.find('VELOCITY MEASUREMENT / SYMMETRIC2D:')
                if vel_cov_loc_start != -1:

                    vel_cov_data = kinematics_data[vel_cov_loc_start:]
                    vel_cov_data = vel_cov_data.strip('VELOCITY MEASUREMENT / SYMMETRIC2D:')
                    vel_cov_data = vel_cov_data.split(',')
                    vel_cov_xx = str(vel_cov_data[0])
                    vel_cov_xx = vel_cov_xx[3:]
                    vel_cov_xx = vel_cov_xx.strip(',')
                    vel_cov_xx = vel_cov_xx.strip(': ')
                    vel_cov_xy = str(vel_cov_data[1])
                    vel_cov_xy = vel_cov_xy[3:]
                    vel_cov_xy = vel_cov_xy.strip(',')
                    vel_cov_xy = vel_cov_xy.strip(': ')
                    vel_cov_yy = str(vel_cov_data[2])
                    vel_cov_yy = vel_cov_yy[3:]
                    vel_cov_yy = vel_cov_yy.strip(',')
                    vel_cov_yy = vel_cov_yy.strip(': ')
                    vel_cov_yy = vel_cov_yy.strip('\n')
                    vel_cov_yy = vel_cov_yy.strip(']')
                    #Set altitude covariance data

                    covariance_dict["vel_xx"] = str(round(float(vel_cov_xx), 5))
                    covariance_dict["vel_xy"] = str(round(float(vel_cov_xy), 5))
                    covariance_dict["vel_yy"] = str(str(round(float(vel_cov_yy), 5)))

                else:
                    covariance_dict["vel_xx"] = 0
                    covariance_dict["vel_xy"] = 0
                    covariance_dict["vel_yy"] = 0

                #return covariance_dict


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


def vector_angle_calc(rep_xx, rep_xy, rep_yy):
    #('----------Vector calculations-----------')
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
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    pearson = round(pearson,5)
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


def draw_ellipse_canvas_test(draw_x, draw_y, count, draws, data_set):
    #TODO -need to get radius data, not covariance data in
    u = 1.  # x-position of the center
    v = 0.5  # y-position of the center
    a = float(draw_x)  # radius on the x-axis
    b = float(draw_y)  # radius on the y-axis
    t_rot = pi / 4  # rotation angle
    # Heading parameters
    hdg_x = 15000
    hgd_y = 15000
    angle = 90
    length = 7500
    data_set = 'Dataset '+str(data_set)
    #Draw counts


    if count == 'ALL':
        t = np.linspace(0, 2 * pi, 100)
        Ell = np.array([a * np.cos(t), b * np.sin(t)])
        # u,v removed to keep the same center location
        R_rot = np.array([[cos(t_rot), -sin(t_rot)], [sin(t_rot), cos(t_rot)]])
        # 2-D rotation matrix

        Ell_rot = np.zeros((2, Ell.shape[1]))
        for i in range(Ell.shape[1]):
            Ell_rot[:, i] = np.dot(R_rot, Ell[:, i])

        plt.plot(u + Ell[0, :], v + Ell[1, :])  # initial ellipse
        plt.plot(u + Ell_rot[0, :], v + Ell_rot[1, :], 'green')  # rotated ellipse
        plt.grid(color='lightgray', linestyle='--')
        plt.title(data_set)

        plt.show(block=False)

    elif draws < count:
        t = np.linspace(0, 2 * pi, 100)
        Ell = np.array([a * np.cos(t), b * np.sin(t)])
        # u,v removed to keep the same center location
        R_rot = np.array([[cos(t_rot), -sin(t_rot)], [sin(t_rot), cos(t_rot)]])

        # 2-D rotation matrix

        Ell_rot = np.zeros((2, Ell.shape[1]))

        for i in range(Ell.shape[1]):
            Ell_rot[:, i] = np.dot(R_rot, Ell[:, i])
        plt.plot(u + Ell[0, :], v + Ell[1, :])  # initial ellipse
        plt.plot(u + Ell_rot[0, :], v + Ell_rot[1, :], 'b')  # rotated ellipse
        plt.grid(color='lightgray', linestyle='--')
        draws = draws+1
        plt.title(data_set)
        plt.show(block=False)
        return draws


    elif draws >= count:
        plt.clf()
        draws = 0
        t = np.linspace(0, 2 * pi, 100)
        Ell = np.array([a * np.cos(t), b * np.sin(t)])
        # u,v removed to keep the same center location
        R_rot = np.array([[cos(t_rot), -sin(t_rot)], [sin(t_rot), cos(t_rot)]])

        # 2-D rotation matrix

        Ell_rot = np.zeros((2, Ell.shape[1]))
        for i in range(Ell.shape[1]):
            Ell_rot[:, i] = np.dot(R_rot, Ell[:, i])

        plt.plot(u + Ell[0, :], v + Ell[1, :])  # initial ellipse
        plt.plot(u + Ell_rot[0, :], v + Ell_rot[1, :], 'green')  # rotated ellipse
        plt.grid(color='lightgray', linestyle='--')
        draws = draws+1
        plt.title(data_set)
        plt.show(block=False)
        return draws

def on_close(event):

    if debug_console:
        window.write_event_value('debug_write', 'Canvas window closed - reopening')
    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'}, label='Uncertainty area')
    canvas = True
    new_window = True
    return fig,ax, canvas, new_window

def draw_ellipse_canvas_test2(x, y, heading, ell_angle, count, draws, data_set, fig, ax, clean, draw_bbox,draw_ell):


    #Variables
    ell_angle = round(float(angle), 1) * (-1) #deg
    #angle = round(float(-90), 1)  # deg
    draw_x = float(x)*2
    draw_y = float(y)*2
    hdg = float(heading)
    #print('draw_ell:'+str(draw_ell))
    #window.write_event_value('debug_write', 'Draw X/Y:' + str(draw_x) + ',' + str(draw_y))
    draws = draws
    window_label = 'Dataset '+str(data_set)
    new_window = False
    #TODO -needs better handling when user closes window -> new window is now opened but not populated

    # Check if window is closed
    fig.canvas.mpl_connect('close_event', on_close)

   #DONE -create own color cycler for ellipse colors cycler('color', ['blue', 'green', 'red', 'cyan', 'purple', 'yellow','black', 'dodgerblue', 'sienna', 'crimson')

    ellipse_draw_color = Color_cycler.color_cycler(data_set)
    bbox_color_adder = str(int(data_set)+2)
    bbox_draw_color = Color_cycler.color_cycler(bbox_color_adder)

    if count == 'ALL':

        if debug_console:
            window.write_event_value('debug_write', 'ALL Draws:' + str(draws))

        ellipse = Ellipse((0,0), width=draw_x, height=draw_y, angle=ell_angle, edgecolor=ellipse_draw_color, alpha=0.3, linewidth=2, facecolor='none')

        #vertice calculation
        path = ellipse.get_path()
        vertices = path.vertices.copy()
        vertices = ellipse.get_patch_transform().transform(vertices)

        if draw_bbox:

            box_c1_x = vertices[3,[0]]
            box_c1_x = box_c1_x[0]-10
            box_c1_y = vertices[3, [1]]
            box_c1_y = box_c1_y[0]-10
            box_c2_x = vertices[9, [0]]
            box_c2_x = box_c2_x[0]+10
            box_c2_y = vertices[9, [1]]
            box_c2_y = box_c2_y[0]-10
            box_c3_x = vertices[15, [0]]
            box_c3_x = box_c3_x[0]+10
            box_c3_y = vertices[15, [1]]
            box_c3_y = box_c3_y[0]+10
            box_c4_x = vertices[21, [0]]
            box_c4_x = box_c4_x[0]-10
            box_c4_y = vertices[21, [1]]
            box_c4_y = box_c4_y[0]+10
            if debug_console:
                window.write_event_value('debug_write', 'Box X/Y and angle:' + str(box_c1_x)+','+str(box_c1_y))
            #DONE - needs to calculate correct position from unc ell radius

            box =Polygon([(box_c1_x,box_c1_y),(box_c2_x,box_c2_y),(box_c3_x,box_c3_y),(box_c4_x,box_c4_y)],closed=True, alpha=0.2, facecolor='none', edgecolor=bbox_draw_color, linewidth=1)
        #plot y-axis
        y_arrow_x = [0,vertices[12,[0]]]
        y_arrow_y = [0,vertices[12,[1]]]
        #ax.plot(y_arrow_x, y_arrow_y, marker='1', label='Y-axis')
        y_axis = lines.Line2D(y_arrow_x, y_arrow_y, marker='1', label='Y-axis', color='r')
        ax.add_line(y_axis)
        # plot x-axis
        x_arrow_x = [0, vertices[6, [0]]]
        x_arrow_y = [0, vertices[6, [1]]]
        #ax.plot(x_arrow_x, x_arrow_y, marker='1', label='X-axis')
        x_axis = lines.Line2D(x_arrow_x, x_arrow_y, marker='1', label='X-axis', color='b')
        ax.add_line(x_axis)

        #heading
        hdg_draw_x = (draw_x/2)/45
        if hdg == 0 or hdg == 360:
            hdg_x = 0
            hdg_y = draw_x / 2
            hdg_arrow_x = [0, hdg_x]
            hdg_arrow_y = [0, hdg_y]
            ax.plot(hdg_arrow_x, hdg_arrow_y, marker='3', label='HDG', color='g')
        elif 0 < hdg < 45:
            hdg_x = 0+(hdg*hdg_draw_x)
            hdg_y = draw_x/2
            hdg_arrow_x = [0, hdg_x]
            hdg_arrow_y = [0, hdg_y]
            ax.plot(hdg_arrow_x, hdg_arrow_y, marker='3', label='HDG', color='g')
        elif hdg == 45:
            hdg_x = draw_x/2
            hdg_y = draw_x/2
            hdg_arrow_x = [0, hdg_x]
            hdg_arrow_y = [0, hdg_y]
            ax.plot(hdg_arrow_x, hdg_arrow_y, marker='3', label='HDG '+str(hdg), color='g')
        elif 45 < hdg < 90:
            hdg_x = draw_x/2
            hdg_y = 0+(hdg-45)*hdg_draw_x
            hdg_arrow_x = [0, hdg_x]
            hdg_arrow_y = [0, hdg_y]
            ax.plot(hdg_arrow_x, hdg_arrow_y, marker='3', label='HDG', color='g')
        elif hdg == 90:
            hdg_x = draw_x / 2
            hdg_y = 0
            hdg_arrow_x = [0, hdg_x]
            hdg_arrow_y = [0, hdg_y]
            ax.plot(hdg_arrow_x, hdg_arrow_y, marker='3', label='HDG', color='g')
        elif 90 < hdg < 135:
            hdg_x = draw_x/2
            hdg_y = 0+((hdg-90)*hdg_draw_x*(-1))
            hdg_arrow_x = [0, hdg_x]
            hdg_arrow_y = [0, hdg_y]
            ax.plot(hdg_arrow_x, hdg_arrow_y, marker='3', label='HDG', color='g')
        elif hdg == 135:
            hdg_x = (draw_x/2)
            hdg_y = (draw_x/2)*(-1)
            hdg_arrow_x = [0, hdg_x]
            hdg_arrow_y = [0, hdg_y]
            ax.plot(hdg_arrow_x, hdg_arrow_y, marker='3', label='HDG', color='g')
        elif 135 < hdg < 180:
            hdg_x = ((draw_x/2)-(hdg-135)*hdg_draw_x)
            hdg_y = (draw_x/2)*(-1)
            hdg_arrow_x = [0, hdg_x]
            hdg_arrow_y = [0, hdg_y]
            ax.plot(hdg_arrow_x, hdg_arrow_y, marker='3', label='HDG', color='g')
        elif hdg == 180:
            hdg_x = 0
            hdg_y = (draw_x/2)*(-1)
            hdg_arrow_x = [0, hdg_x]
            hdg_arrow_y = [0, hdg_y]
            ax.plot(hdg_arrow_x, hdg_arrow_y, marker='3', label='HDG', color='g')
        elif 180 < hdg < 225:
            hdg_x = 0-(hdg-180)*hdg_draw_x
            hdg_y = (draw_x/2)*(-1)
            hdg_arrow_x = [0, hdg_x]
            hdg_arrow_y = [0, hdg_y]
            ax.plot(hdg_arrow_x, hdg_arrow_y, marker='3', label='HDG', color='g')
        elif hdg == 225:
            hdg_x = (draw_x/2)*(-1)
            hdg_y = (draw_x/2)*(-1)
            hdg_arrow_x = [0, hdg_x]
            hdg_arrow_y = [0, hdg_y]
            ax.plot(hdg_arrow_x, hdg_arrow_y, marker='3', label='HDG', color='g')
        elif 225 < hdg < 270:
            hdg_x = (draw_x/2)*(-1)
            hdg_y = (draw_x/2)*(-1)+((hdg-225)*hdg_draw_x)
            hdg_arrow_x = [0, hdg_x]
            hdg_arrow_y = [0, hdg_y]
            ax.plot(hdg_arrow_x, hdg_arrow_y, marker='3', label='HDG', color='g')
        elif hdg == 270:
            hdg_x = (draw_x/2)*(-1)
            hdg_y = 0
            hdg_arrow_x = [0, hdg_x]
            hdg_arrow_y = [0, hdg_y]
            ax.plot(hdg_arrow_x, hdg_arrow_y, marker='3', label='HDG', color='g')
        elif 270 < hdg < 315:
            hdg_x = (draw_x/2)*(-1)
            hdg_y = 0+((hdg-270)*hdg_draw_x)
            hdg_arrow_x = [0, hdg_x]
            hdg_arrow_y = [0, hdg_y]
            ax.plot(hdg_arrow_x, hdg_arrow_y, marker='3', label='HDG', color='g')
        elif hdg == 315:
            hdg_x = (draw_x/2)*(-1)
            hdg_y = (draw_x/2)
            hdg_arrow_x = [0, hdg_x]
            hdg_arrow_y = [0, hdg_y]
            ax.plot(hdg_arrow_x, hdg_arrow_y, marker='3', label='HDG', color='g')
        elif 315 < hdg < 360:
            hdg_x = (draw_x/2)*(-1)+((hdg-315)*hdg_draw_x)
            hdg_y = (draw_x/2)
            hdg_arrow_x = [0, hdg_x]
            hdg_arrow_y = [0, hdg_y]
            ax.plot(hdg_arrow_x, hdg_arrow_y, marker='3', label='HDG', color='g')

        #add ellipse
        if draw_ell:
            if debug_console:
                window.write_event_value('debug_write', 'Add ellipse')
            ax.add_artist(ellipse)

        #add bbox
        if draw_bbox:
            if debug_console:
                window.write_event_value('debug_write', 'Add box')
            ax.add_artist(box)

        #add vectors

        if data_set == '1':
            # set legend
            ax.legend()
            # add point to zero
            ax.plot(0, 0, marker='x')
            # set axes to the same scale
            ax.axis('equal')

            # plot grid lines
            ax.grid(visible=True, which='major')

        # set the x and y axis limits
        ax.set_xlim(-1.2*(draw_x), 1.2*draw_x)
        ax.set_ylim(-1.2*(draw_y), 1.2*draw_y)

        ax.set_title('Dataset - '+data_set)

        draws = draws+1
        plt.show(block=False)

        return draws

    elif draws < count:
        if debug_console:
            window.write_event_value('debug_write', 'ALL Draws:' + str(draws))

        ellipse = Ellipse((0, 0), width=draw_x, height=draw_y, angle=ell_angle, edgecolor=ellipse_draw_color, alpha=0.3,
                          linewidth=2, facecolor='none')

        # vertice calculation
        path = ellipse.get_path()
        vertices = path.vertices.copy()
        vertices = ellipse.get_patch_transform().transform(vertices)

        if draw_bbox:

            box_c1_x = vertices[3, [0]]
            box_c1_x = box_c1_x[0] - 10
            box_c1_y = vertices[3, [1]]
            box_c1_y = box_c1_y[0] - 10
            box_c2_x = vertices[9, [0]]
            box_c2_x = box_c2_x[0] + 10
            box_c2_y = vertices[9, [1]]
            box_c2_y = box_c2_y[0] - 10
            box_c3_x = vertices[15, [0]]
            box_c3_x = box_c3_x[0] + 10
            box_c3_y = vertices[15, [1]]
            box_c3_y = box_c3_y[0] + 10
            box_c4_x = vertices[21, [0]]
            box_c4_x = box_c4_x[0] - 10
            box_c4_y = vertices[21, [1]]
            box_c4_y = box_c4_y[0] + 10
            if debug_console:
                window.write_event_value('debug_write', 'Box X/Y and angle:' + str(box_c1_x) + ',' + str(box_c1_y))
            # DONE - needs to calculate correct position from unc ell radius

            box = Polygon([(box_c1_x, box_c1_y), (box_c2_x, box_c2_y), (box_c3_x, box_c3_y), (box_c4_x, box_c4_y)],
                          closed=True, alpha=0.2, facecolor='none', edgecolor=bbox_draw_color, linewidth=1)
        # plot y-axis
        y_arrow_x = [0, vertices[12, [0]]]
        y_arrow_y = [0, vertices[12, [1]]]
        # ax.plot(y_arrow_x, y_arrow_y, marker='1', label='Y-axis')
        y_axis = lines.Line2D(y_arrow_x, y_arrow_y, marker='1', label='Y-axis', color='r')
        ax.add_line(y_axis)
        # plot x-axis
        x_arrow_x = [0, vertices[6, [0]]]
        x_arrow_y = [0, vertices[6, [1]]]
        # ax.plot(x_arrow_x, x_arrow_y, marker='1', label='X-axis')
        x_axis = lines.Line2D(x_arrow_x, x_arrow_y, marker='1', label='X-axis', color='b')
        ax.add_line(x_axis)

        # heading
        hdg_draw_x = (draw_x / 2) / 45
        if hdg == 0 or hdg == 360:
            hdg_x = 0
            hdg_y = draw_x / 2
            hdg_arrow_x = [0, hdg_x]
            hdg_arrow_y = [0, hdg_y]
            ax.plot(hdg_arrow_x, hdg_arrow_y, marker='3', label='HDG', color='g')
        elif 0 < hdg < 45:
            hdg_x = 0 + (hdg * hdg_draw_x)
            hdg_y = draw_x / 2
            hdg_arrow_x = [0, hdg_x]
            hdg_arrow_y = [0, hdg_y]
            ax.plot(hdg_arrow_x, hdg_arrow_y, marker='3', label='HDG', color='g')
        elif hdg == 45:
            hdg_x = draw_x / 2
            hdg_y = draw_x / 2
            hdg_arrow_x = [0, hdg_x]
            hdg_arrow_y = [0, hdg_y]
            ax.plot(hdg_arrow_x, hdg_arrow_y, marker='3', label='HDG ' + str(hdg), color='g')
        elif 45 < hdg < 90:
            hdg_x = draw_x / 2
            hdg_y = 0 + (hdg - 45) * hdg_draw_x
            hdg_arrow_x = [0, hdg_x]
            hdg_arrow_y = [0, hdg_y]
            ax.plot(hdg_arrow_x, hdg_arrow_y, marker='3', label='HDG', color='g')
        elif hdg == 90:
            hdg_x = draw_x / 2
            hdg_y = 0
            hdg_arrow_x = [0, hdg_x]
            hdg_arrow_y = [0, hdg_y]
            ax.plot(hdg_arrow_x, hdg_arrow_y, marker='3', label='HDG', color='g')
        elif 90 < hdg < 135:
            hdg_x = draw_x / 2
            hdg_y = 0 + ((hdg - 90) * hdg_draw_x * (-1))
            hdg_arrow_x = [0, hdg_x]
            hdg_arrow_y = [0, hdg_y]
            ax.plot(hdg_arrow_x, hdg_arrow_y, marker='3', label='HDG', color='g')
        elif hdg == 135:
            hdg_x = (draw_x / 2)
            hdg_y = (draw_x / 2) * (-1)
            hdg_arrow_x = [0, hdg_x]
            hdg_arrow_y = [0, hdg_y]
            ax.plot(hdg_arrow_x, hdg_arrow_y, marker='3', label='HDG', color='g')
        elif 135 < hdg < 180:
            hdg_x = ((draw_x / 2) - (hdg - 135) * hdg_draw_x)
            hdg_y = (draw_x / 2) * (-1)
            hdg_arrow_x = [0, hdg_x]
            hdg_arrow_y = [0, hdg_y]
            ax.plot(hdg_arrow_x, hdg_arrow_y, marker='3', label='HDG', color='g')
        elif hdg == 180:
            hdg_x = 0
            hdg_y = (draw_x / 2) * (-1)
            hdg_arrow_x = [0, hdg_x]
            hdg_arrow_y = [0, hdg_y]
            ax.plot(hdg_arrow_x, hdg_arrow_y, marker='3', label='HDG', color='g')
        elif 180 < hdg < 225:
            hdg_x = 0 - (hdg - 180) * hdg_draw_x
            hdg_y = (draw_x / 2) * (-1)
            hdg_arrow_x = [0, hdg_x]
            hdg_arrow_y = [0, hdg_y]
            ax.plot(hdg_arrow_x, hdg_arrow_y, marker='3', label='HDG', color='g')
        elif hdg == 225:
            hdg_x = (draw_x / 2) * (-1)
            hdg_y = (draw_x / 2) * (-1)
            hdg_arrow_x = [0, hdg_x]
            hdg_arrow_y = [0, hdg_y]
            ax.plot(hdg_arrow_x, hdg_arrow_y, marker='3', label='HDG', color='g')
        elif 225 < hdg < 270:
            hdg_x = (draw_x / 2) * (-1)
            hdg_y = (draw_x / 2) * (-1) + ((hdg - 225) * hdg_draw_x)
            hdg_arrow_x = [0, hdg_x]
            hdg_arrow_y = [0, hdg_y]
            ax.plot(hdg_arrow_x, hdg_arrow_y, marker='3', label='HDG', color='g')
        elif hdg == 270:
            hdg_x = (draw_x / 2) * (-1)
            hdg_y = 0
            hdg_arrow_x = [0, hdg_x]
            hdg_arrow_y = [0, hdg_y]
            ax.plot(hdg_arrow_x, hdg_arrow_y, marker='3', label='HDG', color='g')
        elif 270 < hdg < 315:
            hdg_x = (draw_x / 2) * (-1)
            hdg_y = 0 + ((hdg - 270) * hdg_draw_x)
            hdg_arrow_x = [0, hdg_x]
            hdg_arrow_y = [0, hdg_y]
            ax.plot(hdg_arrow_x, hdg_arrow_y, marker='3', label='HDG', color='g')
        elif hdg == 315:
            hdg_x = (draw_x / 2) * (-1)
            hdg_y = (draw_x / 2)
            hdg_arrow_x = [0, hdg_x]
            hdg_arrow_y = [0, hdg_y]
            ax.plot(hdg_arrow_x, hdg_arrow_y, marker='3', label='HDG', color='g')
        elif 315 < hdg < 360:
            hdg_x = (draw_x / 2) * (-1) + ((hdg - 315) * hdg_draw_x)
            hdg_y = (draw_x / 2)
            hdg_arrow_x = [0, hdg_x]
            hdg_arrow_y = [0, hdg_y]
            ax.plot(hdg_arrow_x, hdg_arrow_y, marker='3', label='HDG', color='g')

        # add ellipse
        if draw_ell:
            if debug_console:
                window.write_event_value('debug_write', 'Add ellipse')
            ax.add_artist(ellipse)

        # add bbox
        if draw_bbox:
            if debug_console:
                window.write_event_value('debug_write', 'Add box')
            ax.add_artist(box)

        # add vectors

        if data_set == '1':
            # set legend
            ax.legend()
            # add point to zero
            ax.plot(0, 0, marker='x')
            # set axes to the same scale
            ax.axis('equal')

            # plot grid lines
            ax.grid(visible=True, which='major')

        # set the x and y axis limits
        ax.set_xlim(-1.2 * (draw_x), 1.2 * draw_x)
        ax.set_ylim(-1.2 * (draw_y), 1.2 * draw_y)

        ax.set_title('Dataset - ' + data_set)

        draws = draws + 1
        plt.show(block=False)

        return draws

    elif draws >= count or clean == True:
        ax.clear()
        #plt.clf()
        draws = 0
        if debug_console:
            window.write_event_value('debug_write', 'ALL Draws:' + str(draws))

        ellipse = Ellipse((0, 0), width=draw_x, height=draw_y, angle=ell_angle, edgecolor=ellipse_draw_color, alpha=0.3,
                          linewidth=2, facecolor='none')

        # vertice calculation
        path = ellipse.get_path()
        vertices = path.vertices.copy()
        vertices = ellipse.get_patch_transform().transform(vertices)

        if draw_bbox:

            box_c1_x = vertices[3, [0]]
            box_c1_x = box_c1_x[0] - 10
            box_c1_y = vertices[3, [1]]
            box_c1_y = box_c1_y[0] - 10
            box_c2_x = vertices[9, [0]]
            box_c2_x = box_c2_x[0] + 10
            box_c2_y = vertices[9, [1]]
            box_c2_y = box_c2_y[0] - 10
            box_c3_x = vertices[15, [0]]
            box_c3_x = box_c3_x[0] + 10
            box_c3_y = vertices[15, [1]]
            box_c3_y = box_c3_y[0] + 10
            box_c4_x = vertices[21, [0]]
            box_c4_x = box_c4_x[0] - 10
            box_c4_y = vertices[21, [1]]
            box_c4_y = box_c4_y[0] + 10
            if debug_console:
                window.write_event_value('debug_write', 'Box X/Y and angle:' + str(box_c1_x) + ',' + str(box_c1_y))
            # DONE - needs to calculate correct position from unc ell radius

            box = Polygon([(box_c1_x, box_c1_y), (box_c2_x, box_c2_y), (box_c3_x, box_c3_y), (box_c4_x, box_c4_y)],
                          closed=True, alpha=0.2, facecolor='none', edgecolor=bbox_draw_color, linewidth=1)
        # plot y-axis
        y_arrow_x = [0, vertices[12, [0]]]
        y_arrow_y = [0, vertices[12, [1]]]
        # ax.plot(y_arrow_x, y_arrow_y, marker='1', label='Y-axis')
        y_axis = lines.Line2D(y_arrow_x, y_arrow_y, marker='1', label='Y-axis', color='r')
        ax.add_line(y_axis)
        # plot x-axis
        x_arrow_x = [0, vertices[6, [0]]]
        x_arrow_y = [0, vertices[6, [1]]]
        # ax.plot(x_arrow_x, x_arrow_y, marker='1', label='X-axis')
        x_axis = lines.Line2D(x_arrow_x, x_arrow_y, marker='1', label='X-axis', color='b')
        ax.add_line(x_axis)

        # heading
        hdg_draw_x = (draw_x / 2) / 45
        if hdg == 0 or hdg == 360:
            hdg_x = 0
            hdg_y = draw_x / 2
            hdg_arrow_x = [0, hdg_x]
            hdg_arrow_y = [0, hdg_y]
            ax.plot(hdg_arrow_x, hdg_arrow_y, marker='3', label='HDG', color='g')
        elif 0 < hdg < 45:
            hdg_x = 0 + (hdg * hdg_draw_x)
            hdg_y = draw_x / 2
            hdg_arrow_x = [0, hdg_x]
            hdg_arrow_y = [0, hdg_y]
            ax.plot(hdg_arrow_x, hdg_arrow_y, marker='3', label='HDG', color='g')
        elif hdg == 45:
            hdg_x = draw_x / 2
            hdg_y = draw_x / 2
            hdg_arrow_x = [0, hdg_x]
            hdg_arrow_y = [0, hdg_y]
            ax.plot(hdg_arrow_x, hdg_arrow_y, marker='3', label='HDG ' + str(hdg), color='g')
        elif 45 < hdg < 90:
            hdg_x = draw_x / 2
            hdg_y = 0 + (hdg - 45) * hdg_draw_x
            hdg_arrow_x = [0, hdg_x]
            hdg_arrow_y = [0, hdg_y]
            ax.plot(hdg_arrow_x, hdg_arrow_y, marker='3', label='HDG', color='g')
        elif hdg == 90:
            hdg_x = draw_x / 2
            hdg_y = 0
            hdg_arrow_x = [0, hdg_x]
            hdg_arrow_y = [0, hdg_y]
            ax.plot(hdg_arrow_x, hdg_arrow_y, marker='3', label='HDG', color='g')
        elif 90 < hdg < 135:
            hdg_x = draw_x / 2
            hdg_y = 0 + ((hdg - 90) * hdg_draw_x * (-1))
            hdg_arrow_x = [0, hdg_x]
            hdg_arrow_y = [0, hdg_y]
            ax.plot(hdg_arrow_x, hdg_arrow_y, marker='3', label='HDG', color='g')
        elif hdg == 135:
            hdg_x = (draw_x / 2)
            hdg_y = (draw_x / 2) * (-1)
            hdg_arrow_x = [0, hdg_x]
            hdg_arrow_y = [0, hdg_y]
            ax.plot(hdg_arrow_x, hdg_arrow_y, marker='3', label='HDG', color='g')
        elif 135 < hdg < 180:
            hdg_x = ((draw_x / 2) - (hdg - 135) * hdg_draw_x)
            hdg_y = (draw_x / 2) * (-1)
            hdg_arrow_x = [0, hdg_x]
            hdg_arrow_y = [0, hdg_y]
            ax.plot(hdg_arrow_x, hdg_arrow_y, marker='3', label='HDG', color='g')
        elif hdg == 180:
            hdg_x = 0
            hdg_y = (draw_x / 2) * (-1)
            hdg_arrow_x = [0, hdg_x]
            hdg_arrow_y = [0, hdg_y]
            ax.plot(hdg_arrow_x, hdg_arrow_y, marker='3', label='HDG', color='g')
        elif 180 < hdg < 225:
            hdg_x = 0 - (hdg - 180) * hdg_draw_x
            hdg_y = (draw_x / 2) * (-1)
            hdg_arrow_x = [0, hdg_x]
            hdg_arrow_y = [0, hdg_y]
            ax.plot(hdg_arrow_x, hdg_arrow_y, marker='3', label='HDG', color='g')
        elif hdg == 225:
            hdg_x = (draw_x / 2) * (-1)
            hdg_y = (draw_x / 2) * (-1)
            hdg_arrow_x = [0, hdg_x]
            hdg_arrow_y = [0, hdg_y]
            ax.plot(hdg_arrow_x, hdg_arrow_y, marker='3', label='HDG', color='g')
        elif 225 < hdg < 270:
            hdg_x = (draw_x / 2) * (-1)
            hdg_y = (draw_x / 2) * (-1) + ((hdg - 225) * hdg_draw_x)
            hdg_arrow_x = [0, hdg_x]
            hdg_arrow_y = [0, hdg_y]
            ax.plot(hdg_arrow_x, hdg_arrow_y, marker='3', label='HDG', color='g')
        elif hdg == 270:
            hdg_x = (draw_x / 2) * (-1)
            hdg_y = 0
            hdg_arrow_x = [0, hdg_x]
            hdg_arrow_y = [0, hdg_y]
            ax.plot(hdg_arrow_x, hdg_arrow_y, marker='3', label='HDG', color='g')
        elif 270 < hdg < 315:
            hdg_x = (draw_x / 2) * (-1)
            hdg_y = 0 + ((hdg - 270) * hdg_draw_x)
            hdg_arrow_x = [0, hdg_x]
            hdg_arrow_y = [0, hdg_y]
            ax.plot(hdg_arrow_x, hdg_arrow_y, marker='3', label='HDG', color='g')
        elif hdg == 315:
            hdg_x = (draw_x / 2) * (-1)
            hdg_y = (draw_x / 2)
            hdg_arrow_x = [0, hdg_x]
            hdg_arrow_y = [0, hdg_y]
            ax.plot(hdg_arrow_x, hdg_arrow_y, marker='3', label='HDG', color='g')
        elif 315 < hdg < 360:
            hdg_x = (draw_x / 2) * (-1) + ((hdg - 315) * hdg_draw_x)
            hdg_y = (draw_x / 2)
            hdg_arrow_x = [0, hdg_x]
            hdg_arrow_y = [0, hdg_y]
            ax.plot(hdg_arrow_x, hdg_arrow_y, marker='3', label='HDG', color='g')

        # add ellipse
        if draw_ell:
            if debug_console:
                window.write_event_value('debug_write', 'Add ellipse')
            ax.add_artist(ellipse)

        # add bbox
        if draw_bbox:
            if debug_console:
                window.write_event_value('debug_write', 'Add box')
            ax.add_artist(box)

        # add vectors


        # set legend
        ax.legend()
        # add point to zero
        ax.plot(0, 0, marker='x')
        # set axes to the same scale
        ax.axis('equal')

        # plot grid lines
        ax.grid(visible=True, which='major')

        # set the x and y axis limits
        ax.set_xlim(-1.2 * (draw_x), 1.2 * draw_x)
        ax.set_ylim(-1.2 * (draw_y), 1.2 * draw_y)

        ax.set_title('Dataset - ' + data_set)

        draws = draws + 1
        plt.show(block=False)

        return draws


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

    plt.show(block=False)

def file_size_resolver(file_loc):
    file = file_loc
    #print(file)
    lines_in_file = 0
    # Open file
    try:
        f = open(file, "r")
        if debug_console:
            window.write_event_value('debug_write - ', 'file opened')
        #print('file opened')
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

    except Exception as e:
        sg.Popup('ERROR'+'\n'+str(e),non_blocking=True)


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
    #window['print2'].print(xx,xy,yy)
    return xx,xy,yy,yx


sg.theme('Reddit')   # Add a touch of color

#Define variables
ver = "V2"

#Covariance dict
covariance_dict = {
        "lat_lon_xx": "",
        "lat_lon_xy": "",
        "lat_lon_yy": "",
        "alt_xx": "",
        "alt_xy": "",
        "alt_yy": "",
        "vel_xx": "",
        "vel_xy": "",
        "vel_yy": "",
        "hdg": "",
    }

xx = 0

yy = 0

xy = 0

yx = 0

zz = 0

zx = 0

zy = 0

lat_lon_x_result = 0
lat_lon_y_result = 0

hdg = ''
time_tag = ''
time_tag_old = ''
file_loc = ""
metric_units = True
dist_units = 'm'
spd_units = 'm/s'
draw_cov_ell = False
draw_unc_area = True
dataset1 = False
draw_unc_ell = True
dataset2 = True
draw_bound_box = True
dataset_size = 300
file_reader_sleep = 1
data_file_size = 0
thread_event = '-THREAD-'
file_analyzer_thread_run = False
manual_xx_xy_yy = ''
manual_zz_zx_zy = ''
debug_console = False
ellipse_draw_count = 5
ellipse_draws = 0
#TODO -create def that inserts number of datasets into combo_list
data_set_combo_list = ['ALL', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50]
track_combo_list = ['ALL']
fileanalyzer_done = False
canvas = False
new_window = False
debug_write = ''
debug_write_file = ''
debug_win_open = False
# All the stuff inside your window.
column_layout = []

def make_win1():

    tab1_layout = [[sg.Text('Loadfile:'), sg.Input(key="importfile", size=(70,1),enable_events=True), sg.FileBrowse()],
                   [sg.Button('Calculate', key="calculate1", disabled=False),
                    sg.Text('Selected track:'),
                    sg.Combo(track_combo_list, default_value=track_combo_list[0], enable_events=True, readonly=True, key='track_combo'),
                    sg.Text('Units:'), sg.Radio('Imperial', "UNITS", default=True, enable_events=True, key='imperial'),
                    sg.Radio('Metric', "UNITS", default=False, enable_events=True, key='metric')
                    ],
                   [sg.Text('File reader sleep time (s):'),
                    sg.InputText(file_reader_sleep, size=(3, 1), key="sleeptime", enable_events=True),

                    ],
                   [sg.Text('Draw covariance ellipse:'), sg.Checkbox('',key="bool_cov_ell", enable_events=True),
                    sg.Text('Draw uncertainty ellipse:'), sg.Checkbox('', key="bool_unc_ell", default=True,enable_events=True),
                    sg.Text('Draw bounding box:'), sg.Checkbox('', key="bool_bound_box", default=True, enable_events=True)
                    ],
                    [sg.Text('Max number of figures to draw:'),
                     sg.Combo(data_set_combo_list, default_value=data_set_combo_list[0], enable_events=True, readonly=True, key='draw_combo')],
                    [sg.Button('Draw cov ellipse', key="draw_cov_ell", disabled=False, enable_events=True, visible=False),
                     sg.Button('Draw unc ellipse', key="draw_unc_ell", disabled=False, enable_events=True,
                               visible=False),
                     sg.Button('Draw bounding box', key="draw_bound_box", disabled=False, enable_events=True,
                               visible=False),
                    ],
                   [sg.HSep()],
                    [sg.Text('Dataset:'),sg.Input('',key="dataset", size=(4,1), enable_events=True),
                     sg.Text('Track ID:'),sg.Text('',key="track_id", size=(5,1)),
                     sg.Text('Timetag:'),sg.Text('',key="timetag",size=(10,1))],
                    [sg.Text('Update rate:'), sg.Text('',key="update_rate",size=(4,1)), sg.Text('Track hdg:'),
                     sg.Input('',key="trk_hdg", size=(5,1),enable_events=True, readonly=True)],
                   [sg.Text('Latitude, Longitude and Altitude uncertainty values:')],
                    [sg.Text('XX: '),sg.Text('',size=(10,1), key="xx_value1"),sg.Text('XY:'), sg.Text('', size=(10, 1), key="xy_value1"),
                     sg.Text('YY: '),sg.Text('',size=(10,1), key="yy_value1")],
                     [sg.Text('ZZ: '),sg.Text('',size=(10,1), key="zz_value1"),
                    sg.Text('XZ: '), sg.Text('', size=(10, 1), key="xz_value1"), sg.Text('YZ:'),
                    sg.Text('', size=(10, 1), key="yz_value1")],
                    [sg.HSep()],
                    [sg.Text('Velocity uncertainty values:')],
                    [sg.Text('XX: '),sg.Text('',size=(10,1), key="vel_xx_value1"),
                     sg.Text('XY: '), sg.Text('', size=(10, 1), key="vel_xy_value1"),
                     sg.Text('YY: '),sg.Text('',size=(10,1), key="vel_yy_value1")],
                    [sg.HSep()],
                   [sg.Text('Ellipse X:'), sg.Input('',size=(10,1), key="ellip_x", enable_events=True, readonly=True),sg.Text(dist_units),
                    sg.Text('Ellipse Y:'),sg.Input('',size=(10,1), key="ellip_y", enable_events=True, readonly=True),sg.Text(dist_units)],
                    [sg.Text('Ellipse angle:'),sg.Input('', key="ellip_ang",size=(5,1),enable_events=True, readonly=True),
                     sg.Text('Ellipse tilt:'),sg.Input('', key="ellip_tilt",size=(5,1),enable_events=True, readonly=True)],
                    [sg.Multiline("", size=(80, 12), key="print1", autoscroll=True, reroute_stdout=False,write_only=True)],
                    ]

    tab2_layout = [[sg.Button('Calculate', key="calc2_test")],
                   [sg.Text('XX, XY, YY:'), sg.InputText("", size=(70,1), key="xx_xy_yy_values", enable_events=True)],
                   [sg.Text('ZZ, ZX, ZY:'), sg.InputText("", size=(70,1), key="zz_zx_zy_values", enable_events=True)],
                   [sg.Checkbox('Draw covariance area', default=True,enable_events=True, key='draw2')],
                   [sg.Checkbox('Create dataset for drawing', default=True,enable_events=True, key='createdataset2'),
                    sg.Text('Dataset size:'), sg.InputText(dataset_size, size=(5,1), key="dataset_size", enable_events=True)],
                   [sg.Multiline("", size=(70, 5), key="print2")]]

    layout1 = [[sg.TabGroup([[sg.Tab('File', tab1_layout),sg.Tab('Manual',
                tab2_layout)
                              ]
                             ])],[sg.Button('Exit'),
                sg.Text('                                                                                              '
                        '                                        '),
                sg.Image(data=sg.DEFAULT_BASE64_ICON, enable_events=True, key='debug_win', expand_x=True)
                                                                  ]]

    return sg.Window('Covariance analyzer '+ver, layout1, finalize=True, debugger_enabled=False)

def make_win2():
    layout2 = [[sg.Text('Debug console:',key='debug_con_txt',visible=True)],
              [sg.Multiline("", size=(70, 10), key="debug_con", autoscroll=True, reroute_stdout=False,write_only=True, visible=True,expand_y=True)],
                [sg.Button('Close', key='debug_hide', visible=True),sg.Button('Save', key='debug_save', visible=True)]]
    return sg.Window('Debug window ', layout2, finalize=True, debugger_enabled=False,grab_anywhere=True,resizable=True)




# Create the Window
window1, window2 = make_win1(), None
#window = sg.Window('Covariance analyzer '+ver, layout, finalize=True, debugger_enabled=False)

# Event Loop to process "events" and get the "values" of the inputs
while True:
    window, event, values = sg.read_all_windows(timeout=500)
    if event == "importfile":
        # DONE - add handling for canceled fileloc selection
        fileloc = values["importfile"]
        #print(fileloc)
        if fileloc != '':
            output = File_analyzer.file_analyzer(fileloc)
            print(output)
            #output = file_size_resolver(fileloc)
            data_file_size = output[0]
            data_sets_found = output[1]
            window['print1'].print(str(data_sets_found)+" datasets found in file")
            #readfile = filereader(fileloc)
        else:
            window['print1'].print("---- No file selected ----")
    if event == "imperial":
        metric_units = not metric_units
        if metric_units:
            dist_units = 'm'
            spd_units = 'm/s'
        else:
            dist_units = 'ft'
            spd_units = 'ft/s'
    if event == 'debug_pop':
        # DONE -- write more code to prevent opening debug when thread is running -> moved debugger into own thread
        sg.show_debugger_popout_window()
    if event == 'debug_win':
        #DONE -write more code to prevent opening debug when thread is running -> moved debug to its own thread
        debug_console = not debug_console
        window2 = make_win2()
        sg.cprint_set_output_destination(window2, 'debug_con')
        threading.Thread(target=print_debug, name='debug_threat',
                         args=('Thread - debug',1000,window2,debug_write,debug_write_file), daemon=True).start()

        #sg.show_debugger_window()

    if event == 'debug_write':
        sg.cprint(event, values[event])
    if event == 'debug_save':
        time_date = time.strftime("%Y/%m/%d_%H:%M")
        file = time_date+"_debug_log.txt"
        f = open(file, "x")
        f.write(debug_write_file)
        f.close()
    if event == 'close_event':
        print('Reopen window')
        fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'}, label='Uncertainty area')
        canvas = True

    if event == 'debug_hide':
        window2.close()
        debug_console = not debug_console
    if event == "metric":
        metric_units = not metric_units
        if metric_units:
            dist_units = 'm'
            spd_units = 'm/s'
        else:
            dist_units = 'ft'
            spd_units = 'ft/s'
    if event == 'calculate1':
        window['calculate1'].update(disabled=True)
        file_analyzer_thread_run = True
        threading.Thread(target=fileanalyzer,name='fileanalyzer_threat', args=(fileloc, data_file_size, window, file_analyzer_thread_run), daemon=True).start()
        #window['print1'].print('Calculate 1 - Done')
        #file_analyzer_thread_run = False
    if event == 'calculate1_enable':
        window['calculate1'].update(disabled=False)
    if event == "draw_unc_area":
        if debug_console:
            window.write_event_value('debug_write', 'Draw uncertainty area')
        clean = False
        draw_x = values["ellip_x"]
        draw_y = values["ellip_y"]
        data_set = values['dataset']
        if debug_console:
            window.write_event_value('debug_write', 'Dataset:'+str(data_set))
        ellipse_draw_count = values["draw_combo"]
        if debug_console:
            window.write_event_value('debug_write', 'Data for drawer: '+str(ellipse_draw_count) + ' ' + str(ellipse_draws))
        heading = values["trk_hdg"]
        angle = values["ellip_ang"]
        if not canvas:
            fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'}, label='Uncertainty area')
            canvas = True
            if debug_console:
                window.write_event_value('debug_write',
                                     'Canvas created:' + str(canvas))

        else:
            if canvas and data_set == '1':
                clean = True
            else:
                clean = False

        output = draw_ellipse_canvas_test2(draw_x, draw_y, heading,angle ,ellipse_draw_count, ellipse_draws, data_set, fig, ax, clean,draw_bound_box,draw_unc_ell)
        ellipse_draws = output
        if debug_console:
            window.write_event_value('debug_write',
                                 'Output - Unc area draws:'+str(ellipse_draws))
            #TODO -add handling for canvas settings -> user can start new canvas when opening new file or restart olr file.


    if event == "draw_cov_ell":
        if debug_console:
            window.write_event_value('debug_write',
                                 'Draw covariance ellipse event')
        #TODO -add code to perform covariance ellipse draw

    if event == thread_event:
        window.write_event_value('draw_unc_ell', 'fileanalyzer_done')
        window.write_event_value('draw_cov_ell', 'fileanalyzer_done')
        window.write_event_value('draw_bound_box', 'fileanalyzer_done')
        if debug_console:
            window.write_event_value('debug_write',
                                     'Thread_event_update')

    if event == 'calc2_test':
        vec_x = 68.1212154
        vec_y = 76.821629
        val_xx = 18562.070652493334
        val_xy = -1416.075721082226
        val_yy = 23606.251428713906
        val_yx = val_xy
        heading = 120
        data_set = values['dataset']
        type = 'test calc'
        output = covariance_vector2D(val_xx, val_xy, val_yy, type)
        ellipse_draw_count = 0
        ellipse_draws = 0
        draw_ellipse_canvas_test2(vec_x, vec_y, heading,ellipse_draw_count, ellipse_draws, data_set)
        #draw_figure_and_dataset(val_xx, val_xy, val_yy, val_yx)
    if event == 'calculate2':
        values = manual_input_resolver(manual_xx_xy_yy, manual_zz_zx_zy)
        val_xx = values[0]
        val_xy = values[1]
        val_yy = values[2]
        val_yx = values[3]
        output = covariance_vector2D(val_xx, val_xy, val_yy, val_yx)
        if draw_unc_ell and dataset2:
            draw_figure_and_dataset(val_xx,val_xy,val_yy,val_yx)
        elif draw_unc_ell:
            draw_ellipse(val_xx,val_xy,val_yy,val_yx)
        window['print2'].print(output)
    if event == "bool_cov_ell":
        draw_cov_ell = not draw_cov_ell
        if debug_console:
            window.write_event_value('debug_write', 'bool_cov_ell:'+str(draw_cov_ell))
    if event == "bool_unc_ell":
        draw_unc_ell = not draw_unc_ell
        if debug_console:
            window.write_event_value('debug_write', 'bool_unc_ell:'+str(draw_unc_ell))
    if event == "bool_bound_box":
        draw_bound_box = not draw_bound_box
        if debug_console:
            window.write_event_value('debug_write', 'bool_bound_box:'+str(draw_bound_box))
    if event == "draw_combo":
        ellipse_draw_count = values["draw_combo"]

    if event == "createdataset1":
        dataset1 = not dataset1

    if event == "createdataset2":
        dataset2 = not dataset2
        #window['print2'].print(dataset2)
    if event == 'dataset_size':
        dataset_size = values["dataset_size"]
        #window['print2'].print(dataset_size)

    if event == 'sleeptime':
        file_reader_sleep = values["sleeptime"]
    if event == 'xx_xy_yy_values':
        manual_xx_xy_yy = values["xx_xy_yy_values"]
    if event == 'zz_zx_zy_values':
        manual_zz_zx_zy = values["zz_zx_zy_values"]



    if event == sg.WIN_CLOSED or event == 'Exit': # if user closes window or clicks cancel
        break
    #sg.cprint(event, values[event])

window.close()




