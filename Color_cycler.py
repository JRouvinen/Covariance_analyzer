def color_cycler(data_set):
    color = 'crimson'
    number_str = str(data_set)
    str_lenght = len(number_str)
    if str_lenght == 1:
        if data_set == '1':
            color = 'blue'
        if data_set == '2':
            color = 'green'
        if data_set == '3':
            color = 'red'
        if data_set == '4':
            color = 'cyan'
        if data_set == '5':
            color = 'purple'
        if data_set == '6':
            color = 'yellow'
        if data_set == '7':
            color = 'black'
        if data_set == '8':
            color = 'dodgerblue'
        if data_set == '9':
            color = 'sienna'
        if data_set == '0':
            color = 'crimson'

    if str_lenght > 1:
        last_number = number_str[:-1]
        if last_number == '1':
            color = 'blue'
        if last_number == '2':
            color = 'green'
        if last_number == '3':
            color = 'red'
        if last_number == '4':
            color = 'cyan'
        if last_number == '5':
            color = 'purple'
        if last_number == '6':
            color = 'yellow'
        if last_number == '7':
            color = 'black'
        if last_number == '8':
            color = 'dodgerblue'
        if last_number == '9':
            color = 'sienna'
        if last_number == '0':
            color = 'crimson'

    return color


