import sys
import os
import json
import glob

def query_yes_no(question, default="no", allow_skip=False):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if allow_skip:
        valid["skip"] = "skip"
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)
    if allow_skip:
        prompt += " or skip"
    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")


def load_exps_data(exp_path, gap=1, max=None):
    exp_folder_paths = [os.path.abspath(x) for x in glob.iglob(exp_path)]
    exps = []
    for exp_folder_path in exp_folder_paths:
        exps += [x[0] for x in os.walk(exp_folder_path)]
    exps_data = []
    for exp in exps:
        try:
            exp_path = exp
            params_json = load_json(os.path.join(exp_path, "params.json"))
            progress_csv_path = os.path.join(exp_path, "progress.csv")
            pkl_paths = []
            for pkl_path in glob.iglob(os.path.join(exp_path, '*.pkl')):
                pkl_paths.append(pkl_path)
            pkl_paths.sort(key=lambda x: int(x.split('_')[-1][:-4]))
            pkl_paths = pkl_paths[:max:gap]
            exps_data.append(dict(csv=progress_csv_path, json=params_json, pkl=pkl_paths))
        except IOError as e:
            print(e)
    return exps_data


def load_json(params_json_path):
    with open(params_json_path, 'r') as f:
        data = json.loads(f.read())
        if "args_data" in data:
            del data["args_data"]
        if "exp_name" not in data:
            data["exp_name"] = params_json_path.split("/")[-2]
    return data