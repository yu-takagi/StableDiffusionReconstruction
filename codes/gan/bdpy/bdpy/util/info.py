import datetime
import hashlib
import os
import sys
import time
import uuid
import warnings
import yaml


def dump_info(output_dir, script=None, parameters=None, info_file='info.yaml'):
    '''Dump runtime information.'''

    if script is not None:
        script_path = os.path.abspath(script)
        with open(script_path, 'r') as f:
            script_txt = f.read()
        if sys.version_info.major == 2:
            script_md5 = hashlib.md5(script_txt).hexdigest()
        else:
            script_md5 = hashlib.md5(script_txt.encode()).hexdigest()
    else:
        script_path = None
        script_txt = None
        script_md5 = None

    run_id = str(uuid.uuid1())
    run_time = time.time()
    run_info = {
        'run_time'   : run_time,
        'time_stamp' : datetime.datetime.fromtimestamp(run_time).strftime('%Y-%m-%d %H:%M:%S'),
        'host'       : os.uname()[1],
        'hardware'   : os.uname()[4],
        'os'         : os.uname()[0],
        'os_release' : os.uname()[2],
        'os_version' : os.uname()[3],
        'user'       : os.getlogin(),
        'script_path': script_path,
        'script_txt' : script_txt,
        'script_md5' : script_md5,
        }

    if parameters is not None:
        parameters_fixed = {}
        for k, v in parameters.items():
            if isinstance(v, type({}.keys())):
                v = list(v)
            parameters_fixed.update({k: v})
        run_info.update({'parameters': parameters_fixed})

    run_info_file = os.path.join(output_dir, info_file)

    if os.path.isfile(run_info_file):
        with open(run_info_file, 'r') as f:
            info_yaml = yaml.load(f, Loader=yaml.SafeLoader)
        while info_yaml is None:
            warnings.warn('Failed to load info from %s. Retrying...'
                          % run_info_file)
            with open(run_info_file, 'r') as f:
                info_yaml = yaml.load(f, Loader=yaml.SafeLoader)

    else:
        info_yaml = {}

    if '_runtime_info' in info_yaml:
        pass
    else:
        info_yaml.update({'_runtime_info' : {}})

    info_yaml['_runtime_info'].update({run_id: run_info})

    with open(run_info_file, 'w') as f:
        f.write(yaml.dump(info_yaml, default_flow_style=False))

    return run_info
