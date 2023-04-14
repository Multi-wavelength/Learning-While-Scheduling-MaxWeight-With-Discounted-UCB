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

foldername = "data_10_users_" + args.setting

# MaxWeight with discounted UCB
RESULT_FILE_PATH_1 = os.path.join(CUR_DIR, foldername, "max_weight", "discount_" + "yes"
                                  + "_ucb_" + "yes" + "_T_6071209191677812" + ".npy")
# Frame based
RESULT_FILE_PATH_2 = os.path.join(CUR_DIR, foldername, "frame_based", "framesize_120" + ".npy")
# DAM.UCB
RESULT_FILE_PATH_3 = os.path.join(CUR_DIR, foldername, "dam_ucb", "epochsize_161" + ".npy")

FIGURE_FILE_PATH_1 = os.path.join(CUR_DIR, foldername, args.setting + "_3_curves" + "_show_all.pdf")
FIGURE_FILE_PATH_2 = os.path.join(CUR_DIR, foldername, args.setting + "_3_curves" + ".pdf")

args = parser.parse_args()

queue_length_1 = np.load(RESULT_FILE_PATH_1)
queue_length_2 = np.load(RESULT_FILE_PATH_2)
queue_length_3 = np.load(RESULT_FILE_PATH_3)

num_sims = queue_length_1.shape[0]
time_horizon = 100000
total_queue_length_1 = np.sum(queue_length_1[:, :, 0:time_horizon], axis=1).reshape((-1, 1))
total_queue_length_2 = np.sum(queue_length_2[:, :, 0:time_horizon], axis=1).reshape((-1, 1))
total_queue_length_3 = np.sum(queue_length_3[:, :, 0:time_horizon], axis=1).reshape((-1, 1))

del queue_length_1
del queue_length_2
del queue_length_3

total_queue_length_1 = np.concatenate((np.tile(np.arange(time_horizon).reshape((-1, 1)), (num_sims, 1)),
                                       total_queue_length_1), axis=1)
total_queue_length_2 = np.concatenate((np.tile(np.arange(time_horizon).reshape((-1, 1)), (num_sims, 1)),
                                       total_queue_length_2), axis=1)
total_queue_length_3 = np.concatenate((np.tile(np.arange(time_horizon).reshape((-1, 1)), (num_sims, 1)),
                                       total_queue_length_3), axis=1)

total_queue_length_1 = total_queue_length_1[::10, :]
total_queue_length_2 = total_queue_length_2[::10, :]
total_queue_length_3 = total_queue_length_3[::10, :]

pd_data_1 = pd.DataFrame(total_queue_length_1, columns=["Time", "Total Queue Length"])
pd_data_2 = pd.DataFrame(total_queue_length_2, columns=["Time", "Total Queue Length"])
pd_data_3 = pd.DataFrame(total_queue_length_3, columns=["Time", "Total Queue Length"])

del total_queue_length_1
del total_queue_length_2
del total_queue_length_3

pd_data_1["Algorithm"] = "MaxWeight with discounted UCB"
pd_data_2["Algorithm"] = "Frame-based MaxWeight"
pd_data_3["Algorithm"] = "DAM.UCB"

pd_data = pd.concat([pd_data_1, pd_data_2, pd_data_3], axis=0, ignore_index=True)
del pd_data_1, pd_data_2, pd_data_3


plt.figure(figsize=(55, 40), dpi=400)
with sns.plotting_context("notebook"):
    fig = sns.relplot(x="Time", y="Total Queue Length", markers=False, data=pd_data, kind='line',
                      # ci=None,
                      # ci="sd",
                      hue="Algorithm", style="Algorithm",
                      height=4,
                      aspect=1.5
                      )
    sns.move_legend(fig, loc="upper left", bbox_to_anchor=(0.21, 0.97), title=None, frameon=True)
    fig.set_axis_labels('Time', 'Total Queue Length', labelpad=10)
    fig.set(ylim=(0, 6000))
    fig.set(xlim=(0, 10000))
    plt.yticks([0, 1000, 2000, 3000, 4000, 5000, 6000],
               ['0', '1k', '2k', '3k', '4k', '5k', '6k'])
    plt.xticks([0, 20000, 40000, 60000, 80000, 100000],
               ['0', '20k', '40k', '60k', '80k', '100k'])
    plt.grid()
    fig.savefig(FIGURE_FILE_PATH_1)
    # plt.show()

print("finished!")
