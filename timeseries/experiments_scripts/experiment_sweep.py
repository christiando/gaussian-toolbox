from itertools import product


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
        "dataset": ['sunspots', 'energy'],
        "init_w_pca": [1],
        "exp_num": ["1"],
        "seed" : [0],
        "method": {
            "lin_ssm_hsk": {
                "du": [1, 2, 3],
                "dz": [1, 2, 3],
            },
            "lin_ssm": {
                "dz": [1, 2, 3]
            },
            "hmm": {
                "num_states": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                "obs_model": ["gaussian", "studentst"],
            },
            "arimax": {
                "p_arimax":[0, 1, 2, 3],
                "q_arimax":[0, 1, 2, 3],
            },
        }
    }

    commands = []
    
    for dataset in parameters["dataset"]:
        for method in parameters["method"]:
            for exp_num in parameters["exp_num"]:
                for init_pca in parameters["init_w_pca"]:
                    prefix = "python run_experiment.py --dataset {} --exp_num {} --model_name {}  --init_w_pca {}".format(dataset, exp_num, method, init_pca)
                    if len(parameters["method"][method]) == 0:
                        commands.append(prefix)
                    else:
                        commands += simple_sweep(parameters["method"][method], prefix)

    return commands

sweep()
