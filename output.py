import os
import sys


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
            print(f"\u001b[32;1mGPU/CUDA mode:\u001b[0m {params['tpb']} threads per block", flush=True)
        else:
            mode = "SEQUENTIAL" if os.environ.get('OMP_NUM_THREADS') == 1 else "PARALLEL"
            print(f"\u001b[36;1mCPU mode: \u001b[0m{mode} ({os.cpu_count()} cores)", flush=True)

    objectives = {
        'Function': max(max(map(lambda x: len(alias(x['name'])), calls)), 8)
    }

    if not csv and options['onlytimes']:
        for size in sizes:
            objectives[str(size)] = 11
    else:
        objectives['Size'] = max(max(map(lambda x: len(str(x)), sizes)), 4)
        objectives['Time'] = 11

        if not options['onlytimes']:
            objectives['Error'] = 7
            objectives['Average Function'] = 16
            objectives['Average'] = 18
            if options['times']:
                objectives['Average Time'] = 11
            if options['binarizar']:
                objectives['Binary Error'] = 7
                if options['times']:
                    objectives['Binary Time'] = 11
            if options['times']:
                objectives['Total Time'] = 11

    header = ""
    for name, spaces in objectives.items():
        header += f"{name:{spaces}} "
    print("\033[1m" + header[:-1] + "\033[0m", flush=True)
    print('⎯' * len(header), flush=True)
    return objectives


def print_execution(objectives, results, options, prev, flush):
    if options['csv']:
        print(';'.join(map(str, results.values())), flush=True)
        return
    if options['onlytimes']:
        if results['Function'] not in prev:
            prev[results['Function']] = f"{results['Function']:{objectives['Function']}}"
        prev[results['Function']] = prev[results['Function']] + f" {results['Time']:1.5E}"

        if list(objectives)[-1] == str(results['Size']):
            print(f"{prev[results['Function']]}", flush=True)
        return prev
    for key, value in results.items():
        try: output = f"{value:1.5E}" if "Time" in key else value
        except ValueError: output = value
        try:
            if "Error" in key and float(value[:-1]) > 0:
                color = "\u001b[31;1m" if float(value[:-1]) > 5 else "\u001b[33;1m"
                output = f"{color}{output}\u001b[0m  "
        except ValueError: pass
        print(f"{str(output):{objectives[key]}}", end=' ', flush=True)
    print("", end="\r" if flush else "\n")
