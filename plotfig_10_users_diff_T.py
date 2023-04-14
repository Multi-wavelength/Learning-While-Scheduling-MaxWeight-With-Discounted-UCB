import numpy as np
import argparse
import os
import inspect
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

CUR_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PARENT_DIR = os.path.dirname(CUR_DIR)

parser = argparse.ArgumentParser()

parser.add_argument("--setting", default="stationary", help="stationary, aperiodic, periodic, periodic_larger_period")

args = parser.parse_args()

foldername = "data_10_users_" + args.setting + "/max_weight"

# MaxWeight with discounted UCB
RESULT_FILE_PATH = []
T_list = None
if args.setting == "stationary":
    T_list = [4096, 8192, 33554432, 137438953472]
    for T in T_list:
        RESULT_FILE_PATH.append(os.path.join(CUR_DIR, foldername, "discount_" + "yes"
                                             + "_ucb_" + "yes" + "_T_" + str(T) + ".npy"))
elif args.setting == "aperiodic":
    T_list = [2048, 4096, 8192, 16384]
    for T in T_list:
        RESULT_FILE_PATH.append(os.path.join(CUR_DIR, foldername, "discount_" + "yes"
                                             + "_ucb_" + "yes" + "_T_" + str(T) + ".npy"))
elif args.setting == "periodic":
    T_list = [2048, 4096, 8192, 16384]
    for T in T_list:
        RESULT_FILE_PATH.append(os.path.join(CUR_DIR, foldername, "discount_" + "yes"
                                             + "_ucb_" + "yes" + "_T_" + str(T) + ".npy"))
elif args.setting == "periodic_larger_period":
    T_list = [2048, 4096, 8192, 16384]
    for T in T_list:
        RESULT_FILE_PATH.append(os.path.join(CUR_DIR, foldername, "discount_" + "yes"
                                             + "_ucb_" + "yes" + "_T_" + str(T) + ".npy"))
else:
    print("Error Input of Setting!")
    exit(1)

FIGURE_FILE_PATH = os.path.join(CUR_DIR, foldername, args.setting + "_diff_T" + ".pdf")

queue_length = []
for FILE in RESULT_FILE_PATH:
    queue_length.append(np.load(FILE))

num_sims = queue_length[0].shape[0]

if args.setting == "stationary":
    time_horizon = 100000
elif args.setting == "aperiodic":
    time_horizon = 30000
elif args.setting == "periodic":
    time_horizon = 4000
else:
    time_horizon = 40000

total_queue_length = []
for queue_length_i in queue_length:
    total_queue_length.append(np.sum(queue_length_i[:, :, 0:time_horizon], axis=1).reshape((-1, 1)))

del queue_length

for i in range(len(total_queue_length)):
    total_queue_length[i] = np.concatenate((np.tile(np.arange(time_horizon).reshape((-1, 1)), (num_sims, 1)),
                                            total_queue_length[i]), axis=1)
    total_queue_length[i] = total_queue_length[i][::10, :]

pd_data = []
for total_queue_length_i in total_queue_length:
    pd_data.append(pd.DataFrame(total_queue_length_i, columns=["Time", "Total Queue Length"]))

del total_queue_length

for i in range(len(pd_data)):
    pd_data[i]["T"] = str(T_list[i])

pd_data = pd.concat(pd_data, axis=0, ignore_index=True)

plt.figure(figsize=(55, 40), dpi=400)
with sns.plotting_context("notebook"):
    fig = sns.relplot(x="Time", y="Total Queue Length", markers=False, data=pd_data, kind='line',
                      # ci=None,
                      # ci="sd",
                      hue="T",
                      style="T",
                      height=4,
                      aspect=1.5
                      )
    # replace legend title
    new_title = '$g(\gamma)$'
    fig._legend.set_title(new_title)
    fig.set_axis_labels('Time', 'Total Queue Length', labelpad=10)
    if args.setting == "stationary":
        sns.move_legend(fig, loc="upper left", bbox_to_anchor=(0.128, 0.97), frameon=True)
        fig.set(ylim=(0, 400))
        fig.set(xlim=(0, time_horizon))
        plt.xticks([0, 20000, 40000, 60000, 80000, 100000],
                   ['0', '20k', '40k', '60k', '80k', '100k'])
    elif args.setting == "aperiodic":
        sns.move_legend(fig, loc="upper left", bbox_to_anchor=(0.115, 0.97), frameon=True)
        fig.set(ylim=(0, 80))
        fig.set(xlim=(0, time_horizon))
        plt.yticks([0, 20, 40, 60, 80],
                   ['0', '20', '40', '60', '80'])
        plt.xticks([0, 5000, 10000, 15000, 20000, 25000, 30000],
                   ['0', '5k', '10k', '15k', '20k', '25k', '30k'])
    elif args.setting == "periodic":
        sns.move_legend(fig, loc="upper left", bbox_to_anchor=(0.115, 0.97), frameon=True)
        fig.set(ylim=(0, 60))
        fig.set(xlim=(0, time_horizon))
        plt.yticks([0, 10, 20, 30, 40, 50, 60],
                   ['0', '10', '20', '30', '40', '50', '60'])
        plt.xticks([0, 800, 1600, 2400, 3200, 4000],
                   ['0', '800', '1600', '2400', '3200', '4000'])
    else:
        sns.move_legend(fig, loc="upper left", bbox_to_anchor=(0.115, 0.97), frameon=True)
        fig.set(ylim=(0, 60))
        fig.set(xlim=(0, time_horizon))
        plt.yticks([0, 10, 20, 30, 40, 50, 60],
                   ['0', '10', '20', '30', '40', '50', '60'])
        plt.xticks([0, 8000, 16000, 24000, 32000, 40000],
                   ['0', '8k', '16k', '24k', '32k', '40k'])

    plt.grid()
    fig.savefig(FIGURE_FILE_PATH)
    # plt.show()

print("finished!")
