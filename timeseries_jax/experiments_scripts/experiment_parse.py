import sys
import torch
import numpy as np

def tensor_to_latex_table(tensor,
                          row_names,
                          column_names,
                          standalone=True):

    if standalone:
        print("\\documentclass{article}")
        print("\\usepackage{booktabs}")
        print("\\begin{document}")
        print("")

    alignments = "l" + "r" * len(column_names) 

    max_row_len = 12#max(*[len(name) for name in row_names])

    print("\\begin{table}")
    print("\\begin{center}")
    print("\\begin{tabular}{" + alignments + "}")
    print("\\toprule")

    print(" & ".join(column_names) + " \\\\")
    print("\\midrule")

    for row_name, row_values in zip(row_names, tensor):
        row_format = [row_name.ljust(max_row_len)]
        row_format += [v for v in row_values]
        print(" & ".join(row_format) + " \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{center}")
    print("\\caption{}")
    print("\\label{tab:}")
    print("\\end{table}")

    if standalone:
        print("")
        print("\\end{document}")

    return None


if __name__ == "__main__":
    results = np.genfromtxt(sys.argv[1], dtype=np.str)
    datasets = np.unique(results[:, 2])
    methods = np.unique(results[:, 0])
    experiment = np.unique(results[:, 4])
    errors = torch.zeros(len(methods), len(datasets))
    errors = []
    
    print(methods)
    print(datasets)
    
    for row, method in enumerate(methods):
        errors.append([])
        for column, dataset in enumerate(datasets):
            errors[row].append([])
            idx_dataset = np.where(results[:, 2] == dataset)[0]
            idx_method = np.where(results[:, 0] == method)[0]
            idx = np.intersect1d(idx_dataset, idx_method)
            if len(idx):
                nll = results[idx, 7].astype(np.float)
                #nll = results[idx, 14].astype(np.float) #picp
                #nll = results[idx, 17].astype(np.float)
                errors[row][column] = "${:.2f} \\pm {:.2f}$".format(nll.mean(), nll.std())
            else:
                errors[row][column] = "empty"

    #errors = torch.cat((errors, errors.mean(1).view(-1, 1)), 1)
    #print(errors)
    #print(methods)
    #print(datasets)
    tensor_to_latex_table(errors, methods, [""] + list(datasets)) #+ ["all"]
