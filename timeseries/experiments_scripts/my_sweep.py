from itertools import product

dx_dict = {'sunspots': 1,
           'energy': 4,
           'synthetic': 7, 
           'airfoil': 11}

def simple_sweep(grid, prefix=""):
    permutations = product(*grid.values())
    result = []

    for permutation in permutations:
        string = prefix + " "
        for i, key in enumerate(grid.keys()):
            string += "--{} {} ".format(key, permutation[i])
        result.append(string)
        print(string)

    return result


def sweep():
    parameters = {
        "dataset": ['sunspots'],
        "init_w_pca": [0],
        "exp_num": ["1",],
        "seed" : [0],
        "method": {
            "lin_ssm": {
                "dz": list(range(1,4)),
            },
            "linear_hsk_ssm": {
                "dz": list(range(1,4)),
                "du": list(range(1,))
            },
            
        }
    }

    commands = []

    for dataset in parameters["dataset"]:
        for method in parameters["method"]:
            for exp_num in parameters["exp_num"]:
                for init_pca in parameters["init_w_pca"]:
                    prefix = "python run_experiment.py --dataset {} --exp_num {} --model_name {}  --init_w_pca {} --results_file {}".format(dataset, exp_num, method, init_pca, 'sunspots')
                    if len(parameters["method"][method]) == 0:
                        commands.append(prefix)
                    else:
                        commands += simple_sweep(parameters["method"][method], prefix)

    return commands

sweep()