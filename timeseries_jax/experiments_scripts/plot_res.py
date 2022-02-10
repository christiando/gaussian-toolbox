import argparse
import torch

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update(matplotlib.rcParamsDefault)

from matplotlib import pyplot as plt
#plt.rc('text', usetex=True)
#plt.rc('text.latex', preamble=r'\usepackage{times}')
#plt.rc('font', family='serif')
#plt.rc('font', size=16)
plt.rc('font', size = 14)

import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Continuum learning')
    parser.add_argument('--file', type=str, default='./results/results_exp1_0.75_06102021_renamed.txt')
    args = parser.parse_args()

    f = open(args.file, "r")
    lines = f.readlines()
    f.close()

    results = {}

    for line in lines:
        line_fields = [sj.split(" ") for sj in [si.rstrip().lstrip() for si in line.split("|")]]
        line_fields = [item for sublist in line_fields for item in sublist]
        net = line_fields[0]
        dataset = line_fields[1]
        objective_va = float(line_fields[4])
        captured_va = float(line_fields[10])
        width_va = float(line_fields[13])

        captured_te = float(line_fields[11])
        width_te = float(line_fields[14])
        
        seed = 0#int(line_fields[11])

        if dataset not in results:
            results[dataset] = {}
        
        if net not in results[dataset]:
            results[dataset][net] = {}

        
        if seed not in results[dataset][net]:
            results[dataset][net][seed] = []

        results[dataset][net][seed].append([objective_va,
                                            captured_va,
                                            width_va,
                                            captured_te,
                                            width_te])
        
    plt.figure(figsize=(9,4.5), dpi=100)
    plt.subplots_adjust(hspace = .51)
    i = 1
    #for n_ens in ["1", "10"]:
    # assign one different marker and color to each algorithm
    for dataset in results.keys():
            marker_list = [".", "+", "x", "d"]
            marker_idx = 0
            markers = {}
            for net in results[dataset].keys():
                #if net.split("-")[1] == n_ens:
                    markers[net] = (marker_list[marker_idx], marker_idx)
                    marker_idx += 1

            
            
            ax = plt.subplot(1, 2, i)
            ax.set_xlabel("PICP")
            ax.set_xlim(0, 1)

           

            title = ""
            

            ax.set_title(title)

            for net in results[dataset].keys():
                #if net.split("-")[1] == n_ens:
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
                        best_results_te = np.vstack(best_results_te)
                    
                        ax.plot(best_results_te[:, 0],
                                 best_results_te[:, 1],
                                 markers[net][0],
                                 ms=15,
                                 c="C" + str(markers[net][1]),
                                 alpha=1)

                    ax.plot(all_results_te[:, 0],
                             all_results_te[:, 1],
                             markers[net][0],
                             ms=12,
                             c="C" + str(markers[net][1]),
                             label=net.split("-")[0],
                             alpha=0.45)

            ax.axvline(x=0.9, ls='--', c='gray')
            
            
            #ax.show()
            
            ax.locator_params(axis="y", tight=True, nbins=1) 
            ax.locator_params(axis="x", tight=True, nbins=1) 
            ax.set_ylim(0, 1)
            ax.set_yticks([1])
            if i == 1:
                ax.set_title('(a)', loc='left', fontweight='bold', x=-.1, y=1.05)
                ax.set_ylabel("MPIW")
                
                ax.legend()
            else: 
                ax.set_title('(b)', loc='left', fontweight='bold', x=-.1, y=1.05)
            i+=1
    
    
    plt.savefig('results_2in1.pdf'.format(dataset))
