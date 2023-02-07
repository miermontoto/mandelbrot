import os


def alias(string):
    splitted = string.split('_')
    if len(splitted) == 1:
        return string
    alias = splitted[1]
    if len(splitted) == 3:
        alias += f" ({splitted[2]})"
    return alias


def print_header(calls, sizes, options, params):
    csv = options['csv']

    if not csv:
        if options['cuda']:
            print("\u001b[32;1mGPU/CUDA mode:\u001b[0m{paramgs['tpb']} threads per block", flush=True)
            print(f"Threads per block: {params['tpb']}")
        else:
            mode = "SEQUENTIAL" if os.environ.get('OMP_NUM_THREADS') == 1 else "PARALLEL"
            print(f"\u001b[36;1mCPU mode: \u001b[0m{mode} ({os.cpu_count()} cores)", flush=True)

    objectives = {
        'Function': max(max(map(lambda x: len(x['name']), calls)), 8),
        'Size': max(max(map(lambda x: len(str(x)), sizes)), 4),
        'Time': 11,
    }

    if not options['onlytimes']:
        objectives['Error'] = 18
        objectives['Average Function'] = 18
        objectives['Average'] = 18
        if options['times']:
            objectives['Average Time'] = 11
        if options['binarizar']:
            objectives['Binary Error'] = 18
            if options['times']:
                objectives['Binary Time'] = 11

    if csv:
        print(";".join(objectives.keys()), flush=True)
        return
    header = ""
    for name, spaces in objectives.items():
        header += f"{name:{spaces}} "
    print("\033[1m" + header[:-1] + "\033[0m", flush=True)
    return objectives


def print_execution(objectives, results, csv):
    if csv:
        print(';'.join(map(str, results.values())), flush=True)
        return
    for key, value in results.items():
        output = f"{value:1.5E}" if "Time" in key else value
        try: output = f"\u001b[31;1m{output}\u001b[0m" if "Error" in key and float(value) > 0 else output
        except ValueError: pass
        print(f"{str(output):{objectives[key]}}", end=' ', flush=True)
    print()
