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
parser.add_argument("--show_all", default="False", help="whether to use a larger Y-axis limit to show all the curves")

args = parser.parse_args()

foldername = "data_10_users_" + args.setting

# MaxWeight with discounted UCB
RESULT_FILE_PATH_1 = None
# MaxWeight with EM
RESULT_FILE_PATH_2 = None
# MaxWeight with discounted EM
RESULT_FILE_PATH_3 = None
# Frame based
RESULT_FILE_PATH_4 = None
# DAM.UCB
RESULT_FILE_PATH_5 = None

if args.setting == "stationary":
    # MaxWeight with discounted UCB
    RESULT_FILE_PATH_1 = os.path.join(CUR_DIR, foldername, "max_weight", "discount_" + "yes"
                                      + "_ucb_" + "yes" + "_T_6071209191677812" + ".npy")
    # MaxWeight with EM
    RESULT_FILE_PATH_2 = os.path.join(CUR_DIR, foldername, "max_weight", "discount_" + "no"
                                      + "_ucb_" + "no" + "_T_6071209191677812" + ".npy")
    # MaxWeight with discounted EM
    RESULT_FILE_PATH_3 = os.path.join(CUR_DIR, foldername, "max_weight", "discount_" + "yes"
                                      + "_ucb_" + "no" + "_T_6071209191677812" + ".npy")
    # Frame based
    RESULT_FILE_PATH_4 = os.path.join(CUR_DIR, foldername, "frame_based", "framesize_120" + ".npy")
    # DAM.UCB
    RESULT_FILE_PATH_5 = os.path.join(CUR_DIR, foldername, "dam_ucb", "epochsize_161" + ".npy")
elif args.setting == "aperiodic":
    # MaxWeight with discounted UCB
    RESULT_FILE_PATH_1 = os.path.join(CUR_DIR, foldername, "max_weight", "discount_" + "yes"
                                      + "_ucb_" + "yes" + "_T_8192" + ".npy")
    # MaxWeight with EM
    RESULT_FILE_PATH_2 = os.path.join(CUR_DIR, foldername, "max_weight", "discount_" + "no"
                                      + "_ucb_" + "no" + "_T_8192" + ".npy")
    # MaxWeight with discounted EM
    RESULT_FILE_PATH_3 = os.path.join(CUR_DIR, foldername, "max_weight", "discount_" + "yes"
                                      + "_ucb_" + "no" + "_T_8192" + ".npy")
    # Frame based
    RESULT_FILE_PATH_4 = os.path.join(CUR_DIR, foldername, "frame_based", "framesize_30" + ".npy")
    # DAM.UCB
    RESULT_FILE_PATH_5 = os.path.join(CUR_DIR, foldername, "dam_ucb", "epochsize_113" + ".npy")
elif args.setting == "periodic":
    # MaxWeight with discounted UCB
    RESULT_FILE_PATH_1 = os.path.join(CUR_DIR, foldername, "max_weight", "discount_" + "yes"
                                      + "_ucb_" + "yes" + "_T_8192" + ".npy")
    # MaxWeight with EM
    RESULT_FILE_PATH_2 = os.path.join(CUR_DIR, foldername, "max_weight", "discount_" + "no"
                                      + "_ucb_" + "no" + "_T_8192" + ".npy")
    # MaxWeight with discounted EM
    RESULT_FILE_PATH_3 = os.path.join(CUR_DIR, foldername, "max_weight", "discount_" + "yes"
                                      + "_ucb_" + "no" + "_T_8192" + ".npy")
    # Frame based
    RESULT_FILE_PATH_4 = os.path.join(CUR_DIR, foldername, "frame_based", "framesize_25" + ".npy")
    # DAM.UCB
    RESULT_FILE_PATH_5 = os.path.join(CUR_DIR, foldername, "dam_ucb", "epochsize_161" + ".npy")
elif args.setting == "periodic_larger_period":
    # MaxWeight with discounted UCB
    RESULT_FILE_PATH_1 = os.path.join(CUR_DIR, foldername, "max_weight", "discount_" + "yes"
                                      + "_ucb_" + "yes" + "_T_8192" + ".npy")
    # MaxWeight with EM
    RESULT_FILE_PATH_2 = os.path.join(CUR_DIR, foldername, "max_weight", "discount_" + "no"
                                      + "_ucb_" + "no" + "_T_8192" + ".npy")
    # MaxWeight with discounted EM
    RESULT_FILE_PATH_3 = os.path.join(CUR_DIR, foldername, "max_weight", "discount_" + "yes"
                                      + "_ucb_" + "no" + "_T_8192" + ".npy")
    # Frame based
    RESULT_FILE_PATH_4 = os.path.join(CUR_DIR, foldername, "frame_based", "framesize_20" + ".npy")
    # DAM.UCB
    RESULT_FILE_PATH_5 = os.path.join(CUR_DIR, foldername, "dam_ucb", "epochsize_161" + ".npy")
else:
    print("Error Input of Setting!")
    exit(1)

FIGURE_FILE_PATH_1 = os.path.join(CUR_DIR, foldername, args.setting + "_5_curves" + ".pdf")
FIGURE_FILE_PATH_2 = os.path.join(CUR_DIR, foldername, args.setting + "_5_curves" + "_show_all.pdf")

queue_length_1 = np.load(RESULT_FILE_PATH_1)
queue_length_2 = np.load(RESULT_FILE_PATH_2)
queue_length_3 = np.load(RESULT_FILE_PATH_3)
queue_length_4 = np.load(RESULT_FILE_PATH_4)
queue_length_5 = np.load(RESULT_FILE_PATH_5)

num_sims = queue_length_1.shape[0]

if args.setting == "stationary":
    time_horizon = 100000
elif args.setting == "aperiodic":
    time_horizon = 30000
elif args.setting == "periodic":
    time_horizon = 4000
else:
    time_horizon = 40000

total_queue_length_1 = np.sum(queue_length_1[:, :, 0:time_horizon], axis=1).reshape((-1, 1))
total_queue_length_2 = np.sum(queue_length_2[:, :, 0:time_horizon], axis=1).reshape((-1, 1))
total_queue_length_3 = np.sum(queue_length_3[:, :, 0:time_horizon], axis=1).reshape((-1, 1))
total_queue_length_4 = np.sum(queue_length_4[:, :, 0:time_horizon], axis=1).reshape((-1, 1))
total_queue_length_5 = np.sum(queue_length_5[:, :, 0:time_horizon], axis=1).reshape((-1, 1))

del queue_length_1
del queue_length_2
del queue_length_3
del queue_length_4
del queue_length_5

total_queue_length_1 = np.concatenate((np.tile(np.arange(time_horizon).reshape((-1, 1)), (num_sims, 1)),
                                       total_queue_length_1), axis=1)
total_queue_length_2 = np.concatenate((np.tile(np.arange(time_horizon).reshape((-1, 1)), (num_sims, 1)),
                                       total_queue_length_2), axis=1)
total_queue_length_3 = np.concatenate((np.tile(np.arange(time_horizon).reshape((-1, 1)), (num_sims, 1)),
                                       total_queue_length_3), axis=1)
total_queue_length_4 = np.concatenate((np.tile(np.arange(time_horizon).reshape((-1, 1)), (num_sims, 1)),
                                       total_queue_length_4), axis=1)
total_queue_length_5 = np.concatenate((np.tile(np.arange(time_horizon).reshape((-1, 1)), (num_sims, 1)),
                                       total_queue_length_5), axis=1)

total_queue_length_1 = total_queue_length_1[::10, :]
total_queue_length_2 = total_queue_length_2[::10, :]
total_queue_length_3 = total_queue_length_3[::10, :]
total_queue_length_4 = total_queue_length_4[::10, :]
total_queue_length_5 = total_queue_length_5[::10, :]

pd_data_1 = pd.DataFrame(total_queue_length_1, columns=["Time", "Total Queue Length"])
pd_data_2 = pd.DataFrame(total_queue_length_2, columns=["Time", "Total Queue Length"])
pd_data_3 = pd.DataFrame(total_queue_length_3, columns=["Time", "Total Queue Length"])
pd_data_4 = pd.DataFrame(total_queue_length_4, columns=["Time", "Total Queue Length"])
pd_data_5 = pd.DataFrame(total_queue_length_5, columns=["Time", "Total Queue Length"])

del total_queue_length_1
del total_queue_length_2
del total_queue_length_3
del total_queue_length_4
del total_queue_length_5

pd_data_1["Algorithm"] = "MaxWeight with discounted UCB"
pd_data_2["Algorithm"] = "MaxWeight with EM"
pd_data_3["Algorithm"] = "MaxWeight with discounted EM"
pd_data_4["Algorithm"] = "Frame-based MaxWeight"
pd_data_5["Algorithm"] = "DAM.UCB"

pd_data = pd.concat([pd_data_1, pd_data_4, pd_data_5, pd_data_2, pd_data_3], axis=0, ignore_index=True)
del pd_data_1, pd_data_2, pd_data_3, pd_data_4, pd_data_5

if args.show_all == "True":
    plt.figure(figsize=(55, 40), dpi=400)
    with sns.plotting_context("notebook"):
        fig = sns.relplot(x="Time", y="Total Queue Length", markers=False, data=pd_data, kind='line',
                          # ci=None,
                          # ci="sd",
                          hue="Algorithm", style="Algorithm",
                          height=4,
                          aspect=1.5
                          )
        fig.set_axis_labels('Time', 'Total Queue Length', labelpad=10)
        if args.setting == "stationary":
            sns.move_legend(fig, loc="upper left", bbox_to_anchor=(0.095, 0.97), title=None, frameon=True)
            fig.set(ylim=(0, 400000))
            fig.set(xlim=(0, time_horizon))
            plt.yticks([0, 50000, 100000, 150000, 200000, 250000, 300000, 350000, 400000],
                       ['0', '50k', '100k', '150k', '200k', '250k', '300k', '350k', '400k'])
            plt.xticks([0, 20000, 40000, 60000, 80000, 100000],
                       ['0', '20k', '40k', '60k', '80k', '100k'])
        elif args.setting == "aperiodic":
            sns.move_legend(fig, loc="upper left", bbox_to_anchor=(0.095, 0.97), title=None, frameon=True)
            fig.set(ylim=(0, 100000))
            fig.set(xlim=(0, time_horizon))
            plt.yticks([0, 20000, 40000, 60000, 80000, 100000],
                       ['0', '20k', '40k', '60k', '80k', '100k'])
            plt.xticks([0, 5000, 10000, 15000, 20000, 25000, 30000],
                       ['0', '5k', '10k', '15k', '20k', '25k', '30k'])
        elif args.setting == "periodic":
            sns.move_legend(fig, loc="upper left", bbox_to_anchor=(0.085, 0.97), title=None, frameon=True)
            fig.set(ylim=(0, 12000))
            fig.set(xlim=(0, time_horizon))
            plt.yticks([0, 2000, 4000, 6000, 8000, 10000, 12000],
                       ['0', '2k', '4k', '6k', '8k', '10k', '12k'])
            plt.xticks([0, 800, 1600, 2400, 3200, 4000],
                       ['0', '800', '1600', '2400', '3200', '4000'])
        else:
            sns.move_legend(fig, loc="upper left", bbox_to_anchor=(0.095, 0.97), title=None, frameon=True)
            fig.set(ylim=(0, 120000))
            fig.set(xlim=(0, time_horizon))
            plt.yticks([0, 20000, 40000, 60000, 80000, 100000, 120000],
                       ['0', '20k', '40k', '60k', '80k', '100k', '120k'])
            plt.xticks([0, 8000, 16000, 24000, 32000, 40000],
                       ['0', '8k', '16k', '24k', '32k', '40k'])

        plt.grid()
        fig.savefig(FIGURE_FILE_PATH_2)
        # plt.show()
else:
    plt.figure(figsize=(55, 40), dpi=400)
    with sns.plotting_context("notebook"):
        fig = sns.relplot(x="Time", y="Total Queue Length", markers=False, data=pd_data, kind='line',
                          # ci=None,
                          # ci="sd",
                          hue="Algorithm", style="Algorithm",
                          height=4,
                          aspect=1.3
                          )
        fig.set_axis_labels('Time', 'Total Queue Length', labelpad=10)
        if args.setting == "stationary":
            sns.move_legend(fig, loc="upper left", bbox_to_anchor=(0.1212, 0.97), title=None, frameon=True)
            fig.set(ylim=(0, 6000))
            fig.set(xlim=(0, time_horizon))
            plt.yticks([0, 1000, 2000, 3000, 4000, 5000, 6000],
                       ['0', '1k', '2k', '3k', '4k', '5k', '6k'])
            plt.xticks([0, 20000, 40000, 60000, 80000, 100000],
                       ['0', '20k', '40k', '60k', '80k', '100k'])
        elif args.setting == "aperiodic":
            sns.move_legend(fig, loc="upper left", bbox_to_anchor=(0.125, 0.97), title=None, frameon=True)
            fig.set(ylim=(0, 160))
            fig.set(xlim=(0, time_horizon))
            plt.yticks([0, 40, 80, 120, 160],
                       ['0', '40', '80', '120', '160'])
            plt.xticks([0, 5000, 10000, 15000, 20000, 25000, 30000],
                       ['0', '5k', '10k', '15k', '20k', '25k', '30k'])
        elif args.setting == "periodic":
            sns.move_legend(fig, loc="upper left", bbox_to_anchor=(0.147, 0.97), title=None, frameon=True)
            fig.set(ylim=(0, 240))
            fig.set(xlim=(0, time_horizon))
            plt.yticks([0, 40, 80, 120, 160, 200, 240],
                       ['0', '40', '80', '120', '160', '200', '240'])
            plt.xticks([0, 800, 1600, 2400, 3200, 4000],
                       ['0', '800', '1600', '2400', '3200', '4000'])
        else:
            sns.move_legend(fig, loc="upper left", bbox_to_anchor=(0.128, 0.97), title=None, frameon=True)
            fig.set(ylim=(0, 240))
            fig.set(xlim=(0, time_horizon))
            plt.yticks([0, 40, 80, 120, 160, 200, 240],
                       ['0', '40', '80', '120', '160', '200', '240'])
            plt.xticks([0, 8000, 16000, 24000, 32000, 40000],
                       ['0', '8k', '16k', '24k', '32k', '40k'])

        plt.grid()
        fig.savefig(FIGURE_FILE_PATH_1)
        # plt.show()

print("finished!")
