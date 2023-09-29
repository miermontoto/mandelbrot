import os
import sys
import time
import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer
from numpy import linalg as LA
import mandel
import output
from mandel import mandelPy, mediaPy, binarizaPy


def read_options(argv):
    options = {
        'debug': "debug" in argv,
        'binarizar': "bin" in argv,
        'diffs': "diffs" in argv,
        'times': "times" in argv,
        'onlytimes': "onlytimes" in argv,
        'mode': 'cuda' if 'tpb' in argv else 'omp',
        'cuda': 'tpb' in argv,
        'noheader': "noheader" in argv,
        'csv': "csv" in argv,
    }

    params = {
        'xmin': float(argv[1]),
        'xmax': float(argv[2]),
        'ymin': float(argv[3]),
        'maxiter': int(argv[4])
    }

    if os.environ.get('OMP_NUM_THREADS') is not None: options['threads'] = int(os.environ.get('OMP_NUM_THREADS'))
    else: options['threads'] = os.cpu_count()

    if options['cuda']:  # Detección de modo CUDA
        try: tpb = int(argv[argv.index("tpb") + 1])
        except Exception:
            tpb = 32
            print("Error al obtener el número de hilos por bloque, se utiliza valor por defecto (32)")
        params['tpb'] = tpb

    params['ymax'] = params['xmax'] - params['xmin'] + params['ymin']

    return options, params


def read_calls(argv, mode):
    valid_functions = {   # Diccionario de funciones válidas y sus alias en parámetros.
        'omp': {
            'mandel': {
                'normal': 'mandel_normal',
                'collapse': 'mandel_collapse',
                'tasks': 'mandel_tasks',
                'schedule_auto': 'mandel_schedule_auto',
                'schedule_static': 'mandel_schedule_static',
                'schedule_guided': 'mandel_schedule_guided',
                'schedule_dynamic': 'mandel_schedule_dynamic'
            },
            'promedio': {
                'normal': 'promedio_normal',
                'int': 'promedio_int',
                'schedule': 'promedio_schedule',
                'atomic': 'promedio_atomic',
                'critical': 'promedio_critical',
                'vect': 'promedio_vect'
            }
        },
        'cuda': {
            'mandel': {
                'normal': 'mandelGPU_normal',
                'heter': 'mandelGPU_heter',
                'unified': 'mandelGPU_unified',
                'pinned': 'mandelGPU_pinned',
                '1D': 'mandelGPU_1D'
            },
            'promedio': {
                'api': 'promedioGPU_api',
                'shared': 'promedioGPU_shared',
                'param': 'promedioGPU_param',
                'atomic': 'promedioGPU_atomic',
            }
        }
    }

    valid_calls = {
        'prof': {
            'function': 'mandelProf',
            'name': 'fractalProf',
            'average': 'mediaProf',
            'binary': 'binarizaProf'
        },
        'py': {
            'function': 'mandelPy',
            'name': 'fractalPy',
            'average': 'mediaPy',
            'binary': 'binarizaPy'
        },
        'own': {}
    }

    calls = []
    sizes = []

    for key in list(valid_calls.keys()):
        if key in argv:
            if key == 'own':
                averages = [next(iter(valid_functions[mode]['promedio'].values()))]

                if "averages" in sys.argv:
                    if "all" == sys.argv[sys.argv.index("averages") + 1]:
                        averages = list(valid_functions[mode]['promedio'].values())
                    else:
                        averages = []
                        for i in range(sys.argv.index("averages") + 1, len(sys.argv)):
                            if sys.argv[i] in valid_functions[mode]['promedio']:
                                averages.append(valid_functions[mode]['promedio'][sys.argv[i]])
                            else: break

                if "methods" in sys.argv:
                    if "all" == sys.argv[sys.argv.index("methods") + 1]:
                        for key, value in valid_functions[mode]['mandel'].items():
                            for average in averages:
                                calls.append({
                                    'function': value,
                                    'name': f'fractalAlumnx{key.capitalize()}',
                                    'average': average,
                                    'binary': 'binarizaAlumnx'
                                })
                    else:
                        for i in range(sys.argv.index("methods") + 1, len(sys.argv)):
                            if sys.argv[i] in valid_functions[mode]['mandel']:
                                for average in averages:
                                    calls.append({
                                        'function': valid_functions[mode]['mandel'][sys.argv[i]],
                                        'name': f'fractalAlumnx{sys.argv[i].capitalize()}',
                                        'average': average,
                                        'binary': 'binarizaAlumnx'
                                    })
                            else: break
                else:
                    for average in averages:
                        calls.append({
                            'function': next(iter(valid_functions[mode]['mandel'].values())),
                            'name': 'fractalAlumnx',
                            'average': average,
                            'binary': 'binarizaAlumnx'
                        })
            else: calls.append(valid_calls.get(key))
        elif f"-{key}" in sys.argv and key in valid_calls:
            calls.remove(valid_calls.get(key))

    if "sizes" in sys.argv:
        for i in range(sys.argv.index("sizes") + 1, len(sys.argv)):
            try: sizes.append(int(sys.argv[i]))
            except Exception: break

    if len(sizes) == 0: sizes.append(4)  # marcar error si no se detectan tamaños
    return calls, sizes


def load_libraries(calls, cuda):
    functions = {
        'mandel': {
            'name': 'mandel',
            'restype': None,
            'argtypes': [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.c_int, ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]
        },
        'media': {
            'name': 'promedio',
            'restype': ctypes.c_double,
            'argtypes': [ctypes.c_int, ctypes.c_int, ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]
        },
        'binariza': {
            'name': 'binariza',
            'restype': None,
            'argtypes': [ctypes.c_int, ctypes.c_int, ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ctypes.c_double]
        }
    }

    owners = ['Prof', 'Alumnx']
    for owner in owners:
        lib = ctypes.cdll.LoadLibrary(f"./{'cuda' if cuda else 'openmp'}/mandel{owner}{'GPU' if cuda else ''}.so")
        for key, value in functions.items():
            if owner == 'Alumnx' and (key == 'mandel' or key == 'media'): continue
            globals()[f"{key}{owner}"] = getattr(lib, f"{value['name']}{'GPU' if cuda else ''}")
            globals()[f"{key}{owner}"].restype = value['restype']
            globals()[f"{key}{owner}"].argtypes = value['argtypes']

        if owner == "Alumnx":
            for call in calls:
                if "Prof" in call['function'] or "Py" in call['function']: continue
                globals()[f"{call['function']}"] = getattr(lib, call['function'])
                globals()[f"{call['function']}"].restype = functions['mandel']['restype']
                globals()[f"{call['function']}"].argtypes = functions['mandel']['argtypes']
                globals()[f"{call['average']}"] = getattr(lib, call['average'])
                globals()[f"{call['average']}"].restype = functions['media']['restype']
                globals()[f"{call['average']}"].argtypes = functions['media']['argtypes']


def execute(calls, sizes, options, params):
    times = options['times']
    onlytimes = options['onlytimes']
    cuda = options['cuda']
    binarizar = options['binarizar']
    mode = options['mode']
    debug = options['debug']
    diffs = options['diffs']
    csv = options['csv']

    if cuda: tpb = params['tpb']
    xmin = params['xmin']
    ymin = params['ymin']
    xmax = params['xmax']
    ymax = params['ymax']
    maxiter = params['maxiter']

    objectives = output.print_header(calls, sizes, options, params) if not csv else []
    prev = {}

    if options['cuda']:  # heat up cache
        for i in range(0, 3):
            size = next(iter(sizes))
            for call in calls:
                locals()[call['name']] = np.zeros(size * size).astype(np.double)
                globals()[call['function']](xmin, ymin, xmax, ymax, maxiter, size, size, locals()[call['name']], tpb)
                globals()[call['average']](size, size, locals()[call['name']], tpb)

    for size in sizes:
        yres = size
        xres = size

        for call in calls:
            function = call['function']
            name = call['name']
            average_function = call['average']
            binary_function = call['binary']
            original = calls[0]['name']

            check_cuda = cuda and "Py" not in function

            # Como indicado en clase, tamaños superiores a 2048 suponen un calculo
            # demasiado largo y no son útiles para la práctica.
            # Para poder enviar todos los tamaños en una sola ejecución, se comprueba
            # el tamaño aquí.
            if "Py" in function and size > 2048: continue

            locals()[name] = np.zeros(yres * xres).astype(np.double)  # reservar memoria

            results = {
                'Function': output.alias(function),
                'Size': size,
                'Time': '...',
                'Error': '...'
            }

            if not onlytimes and not csv:
                results['Average Function'] = output.alias(average_function)
                results['Average'] = '...'
                if times: results['Average Time'] = '...'
                if binarizar:
                    results['Binary Function'] = output.alias(binary_function)
                    results['Binary Error'] = '...'
                    if times: results['Binary Time'] = '...'
                if times: results['Total Time'] = '...'
                output.print_execution(objectives, results, options, prev, True)

            # ejecutar función
            calc_time = time.time()
            if check_cuda: globals()[function](xmin, ymin, xmax, ymax, maxiter, xres, yres, locals()[name], tpb)
            else: globals()[function](xmin, ymin, xmax, ymax, maxiter, xres, yres, locals()[name])
            calc_time = time.time() - calc_time
            results['Time'] = calc_time

            # calcular error
            try: error = "-" if original == name else '{:.2f}%'.format(sum(abs(locals()[name] - locals()[original])) / (xres * yres))  # calcular error
            except Exception: error = "NaN"
            results['Error'] = error

            if not onlytimes:  # calcular promedio
                results['Average Function'] = output.alias(average_function)
                average_time = time.time()
                if check_cuda: average = globals()[average_function](xres, yres, locals()[name], tpb)
                else: average = globals()[average_function](xres, yres, locals()[name])  # calcular promedio
                average_time = time.time() - average_time
                results['Average'] = average
                if times: results['Average Time'] = average_time

            # guardar imágenes
            if debug:
                mandel.grabar(locals()[name], xres, yres, f"{name}_{size}.bmp")  # guardar archivo
                if diffs and i > 0: mandel.grabar(mandel.diffImage(locals()[name], globals()[original]), xres, yres, f"diff_{name}_{size}.bmp")

            # binarizar
            if binarizar and not onlytimes:
                binary_name = f"bin_{name}"
                binary_original = f"bin_{original}"
                globals()[binary_name] = np.copy(locals()[name])  # copiar imagen para evitar sobreescribirla

                # calcular binarización
                binary_time = time.time()
                if check_cuda: globals()[binary_function](xres, yres, globals()[binary_name], average, tpb)
                else: globals()[binary_function](yres, xres, globals()[binary_name], average)
                binary_time = time.time() - binary_time

                # calcular e imprimir error
                error = "-" if binary_name == binary_original else LA.norm(globals()[binary_name] - globals()[binary_original])
                results['Binary Error'] = error
                if times: results['Binary Time'] = binary_time

                # guardar binarizado
                if debug: mandel.grabar(globals()[binary_name], xres, yres, f"{binary_name}_{size}.bmp")

            if times and not onlytimes:
                total = calc_time + average_time
                if binarizar: total += binary_time
                results['Total Time'] = total

            prev = output.print_execution(objectives, results, options, prev, False)


if __name__ == '__main__':
    options, params = read_options(sys.argv)   # Leer parámetros de entrada
    os.system(f"make {options['mode']} >/dev/null")  # Compilar librerías necesarias
    calls, sizes = read_calls(sys.argv, options['mode'])  # Leer funciones y tamaños a ejecutar
    load_libraries(calls, options['cuda'])  # Cargar librerías mediante ctypes
    execute(calls, sizes, options, params)  # Ejecutar funciones y mostrar resultados
