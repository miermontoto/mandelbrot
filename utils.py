def alias(string):
    splitted = string.split('_')
    if len(splitted) == 1:
        return string
    alias = splitted[1]
    if len(splitted) == 3:
        alias += f" ({splitted[2]})"
    return alias


def print_header(options):
    csv = options['csv']
    separator = ';' if csv else ' '
    objectives = {
        'Function': '',
        'Mode': '',
        'Size': '',
        'Time': '',
    }
    if not options['onlytimes']:
        objectives.append('Error')
        objectives.append('Average Function')
        objectives.append('Average')
        if options['times']:
            objectives.append('Average Time')
        if options['binarizar']:
            objectives.append('Binary (err)')
            if options['times']:
                objectives.append('Binary Time')


