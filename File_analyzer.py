def file_analyzer(file_loc):
    file = file_loc
    #print(file)
    lines_in_file = 0
    # Open file
    try:
        f = open(file, "r")
        #window.write_event_value('debug_write - ', 'file opened')
        #print('file opened')
        for x in f:
            lines_in_file = lines_in_file + 1

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
                    data_found = data_found + 1
                    # print(data_found)
                # Read track data time tag
                line_id_Timetag = x
                line_id_Timetag_bool = line_id_Timetag.find('UPDATE')
                if line_id_Timetag_bool != -1:
                    data_found = data_found + 1
                    # print(data_found)
                # Read MST Track ID
                line_id_MSTTrack = x[:8]
                # print(line_id_MSTTrack)
                if line_id_MSTTrack == 'MSTTrack':
                    data_found = data_found + 1
                    # print(data_found)
                # Read Covariance
                line_id_Kinematics = x[:10]
                # print(line_id_Kinematics)
                if line_id_Kinematics == 'Kinematics':
                    data_found = data_found + 1
                    # print(data_found)

                if data_found == 4:
                    data_sets = data_sets + 1
                    # print(data_found)

        return lines_in_file, data_sets

    except Exception as e:
        error_msg = str(e)
        return error_msg
        #sg.Popup('ERROR' + '\n' + str(e), non_blocking=True)