import argparse
import torch
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--file', type=str, default='./results/results_exp1_0.75_06102021_renamed.txt')
    args = parser.parse_args()

    optimize_picp = False

    f = open(args.file, "r")
    lines = f.readlines()
    f.close()

    results = {}
    all_nets = set()

    for line in lines:
        line_fields = [sj.split(" ") for sj in [si.rstrip().lstrip()
                                                for si in line.split("|")]]
        line_fields = [item for sublist in line_fields for item in sublist]

        net = line_fields[0]
        dataset = line_fields[1]
        objective_va = round(float(line_fields[4]), 2)
        captured_va = round(float(line_fields[10]), 2)
        width_va = round(float(line_fields[13]), 2)
        objective_te = round(float(line_fields[5]), 2)
        captured_te = round(float(line_fields[11]), 2)
        width_te = round(float(line_fields[14]), 2)

        seed = 0 #int(line_fields[11])

        if dataset not in results:
            results[dataset] = {}

        if net not in results[dataset]:
            results[dataset][net] = {}

        all_nets.add(net)

        if seed not in results[dataset][net]:
            results[dataset][net][seed] = []

        results[dataset][net][seed].append([objective_va,
                                            captured_va,
                                            width_va,
                                            objective_te,
                                            captured_te,
                                            width_te])

    all_nets = list(all_nets)
    print(results)
    
    for dataset in results.keys():
            
        for net in results[dataset].keys():
                    all_results_va = []
                    all_results_te = []
                    best_results_te = []

                    # for each seed, collect all results and annotate the best one
                    # according to the validation set, since this result is what we
                    # would get if optimizing hyperparameters
                    for seed in results[dataset][net].keys():
                        seed_results = np.array(results[dataset][net][seed])

                        all_results_va.append(seed_results[:, :3])
                        all_results_te.append(seed_results[:, 3:])

                        index_best_va = np.argmax(seed_results[:, 0])
                        best_results_te.append(seed_results[index_best_va, 3:])

                    all_results_va = np.vstack(all_results_va)
                    all_results_te = np.vstack(all_results_te)

                    if len(best_results_te):
                        best_results_te = np.asarray(best_results_te).reshape(-1)
                        
                    results[dataset][net]["summary"] = "${:.2f} & {:.2f} & {:.2f}$ ".format(
                     best_results_te[0],
                     best_results_te[1],
                     best_results_te[2])
    
    
    print("\\resizebox{\\textwidth}{!}{")
    print("\\begin{tabular}{l|ccc|ccc|ccc|ccc}")
    print("\\toprule")
    row_str = "{:<30} & ".format("\\textbf{dataset ($d_x$)}")

    for method in all_nets:
        row_str += "{:<32}".format("\multicolumn{3}{c}{" + method + "} & ")
    row_str = row_str[:-3]
    row_str += "\\\\"
    print(row_str)
    print("\\midrule")
    print("       & \multicolumn{1}{c}{LL} &       \multicolumn{1}{c}{PICP} &       \multicolumn{1}{c}{MPIW} &        \multicolumn{1}{c}{LL} &       \multicolumn{1}{c}{PICP} &       \multicolumn{1}{c}{MPIW} & \multicolumn{1}{c}{LL} &       \multicolumn{1}{c}{PICP} &       \multicolumn{1}{c}{MPIW} & \multicolumn{1}{c}{LL} &       \multicolumn{1}{c}{PICP} &       \multicolumn{1}{c}{MPIW} &  ")
    print("\\midrule")
    for dataset in results:
        dname = dataset

        row_str = "{:<30}".format(dname) + " & "
        for method in all_nets:
            if method in results[dataset]:
                row_str += results[dataset][method]["summary"] + " & "
            else:
                row_str += "                                 & "
        print(row_str[:-2] + "\\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("}")
    
    #print(results)
