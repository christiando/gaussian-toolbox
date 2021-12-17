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
        "dataset": ['synthetic'],
        "init_w_pca": [0],
        "init_lin_model": [1],
        "exp_num": ["1",],
        "seed" : [0],
        "train_ratio": [.75],
        "method": {
            "lin_ssm": {
                "dz": list(range(1,7)),
            },
            "linear_hsk_ssm": {
                "dz": list(range(1,6)),
                "du": list(range(1,6))
            },
            "arimax" : {
                "p_arimax": list(range(1,8)),
                "q_arimax": list(range(1,8))
            },
            "hmm": {
                "num_states": list(range(1,40))
            }
            
        }
    }

    commands = []

    for dataset in parameters["dataset"]:
        for method in parameters["method"]:
            for exp_num in parameters["exp_num"]:
                for init_lin_model in parameters["init_lin_model"]:
                    for train_ratio in parameters["train_ratio"]:
                        prefix = "python run_experiment.py --dataset {} --exp_num {} --model_name {}  --init_lin_model {} --train_ratio {} --results_file {}".format(dataset, exp_num, method, init_lin_model, train_ratio, dataset)
                        if len(parameters["method"][method]) == 0:
                            commands.append(prefix)
                        else:
                            commands += simple_sweep(parameters["method"][method], prefix)

    return commands

sweep()