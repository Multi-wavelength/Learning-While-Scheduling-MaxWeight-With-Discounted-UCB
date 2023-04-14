import numpy as np
import argparse
import os
import inspect
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import math

CUR_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PARENT_DIR = os.path.dirname(CUR_DIR)

parser = argparse.ArgumentParser()

parser.add_argument("--setting", default="stationary", help="stationary, aperiodic, periodic, periodic_larger_period")
parser.add_argument("--arrival_rate_bound",
                    default=1.0, type=float,
                    help='the upper bound of the arrival rate')
parser.add_argument("--service_time_bound",
                    default=2.0, type=float,
                    help='the upper bound of the service time')
parser.add_argument("--num_users",
                    default=10, type=int,
                    help='number of users')
parser.add_argument("--num_servers",
                    default=10, type=int,
                    help='number of servers')
parser.add_argument("--simulation_horizon",
                    default=100000, type=int,
                    help='simulation horizon, the number of time slots')
parser.add_argument("--num_simulations",
                    default=100, type=int,
                    help='number of simulations in parallel')
parser.add_argument("--epoch_size",
                    default=161, type=int,
                    help='frame size')
# slackness = 0.9/0.75-1=0.2 for the stationary and periodic settings
# L_epoch = ceil(32/slackness+1)=161, where we let L_conv=1
# for the aperiodic setting, slackness = 0.9/0.7 - 1
# L_epoch = ceil(32/slackness+1)=113, where we let L_conv=1
parser.add_argument("--min_service_rate",
                    default=0.5, type=float)

args = parser.parse_args()

foldername = "data_10_users_" + args.setting + "/dam_ucb"

# check the directory does not exist
if not (os.path.exists(foldername)):
    # create the directory
    os.makedirs(foldername)

RESULT_FILE_PATH = os.path.join(CUR_DIR, foldername, "epochsize_" + str(args.epoch_size) + ".npy")
FIGURE_FILE_PATH = os.path.join(CUR_DIR, foldername, "epochsize_" + str(args.epoch_size) + ".png")

rng = np.random.default_rng()


def dam_ucb_max_weight(arrival, mu, service_time_bound=args.service_time_bound,
                       num_users=args.num_users, num_servers=args.num_servers,
                       simulation_horizon=args.simulation_horizon, num_simulations=args.num_simulations,
                       epoch_size=args.epoch_size, min_service_rate=args.min_service_rate):
    """
    DAM.UCB Algorithm adapted to the centralized setting
    https://arxiv.org/pdf/2206.03324.pdf

    :param arrival: a numpy array with shape (num_users, N)
                    the periodic mean for the arrival rate with a Bernoulli distribution.
                    Examples:
                    Suppose N=1, arrival[0, 0]=0.85,
                    then the arrival rate of user 0 follows a time-invariant Bernoulli distribution with mean 0.85.
                    Suppose N=8, mu[0, :]=[0.9, 0.8, 0.7, 0.6, 0.5, 0.6, 0.7, 0.8],
                    then the arrival rate of user 0 follows a Bernoulli distribution whose mean changes periodically as:
                    1st time slot: mean=0.9;
                    ...
                    8th time slot: mean=0.8;
                    9th time slot: mean=0.9;
                    ...
    :param mu: a numpy array with shape (num_users, num_servers, N)
               the periodic mean for the service rate
               the service time = 1 + a Bernoulli random variable.
               the service rate = 1/the service time
               Examples:
               Suppose N=1, mu[1, 0, 0]=0.9,
               then the service rate of (user 1, server 0) follows a time-invariant distribution with mean 0.9.
               Suppose N=8, mu[1, 0, :]=[0.6, 0.7, 0.8, 0.9, 1.0, 0.9, 0.8, 0.7],
               then the mean of the service rate of (user 1, server 0) changes periodically as:
               1st time slot: mean=0.6;
               ...
               8th time slot: mean=0.7
               9th time slot: mean=0.6;
               ...
    :param service_time_bound: the upper bound of the service time
    :param num_users: number of users
    :param num_servers: number of servers
    :param simulation_horizon: an integer, the simulation horizon
    :param num_simulations: number of simulations in parallel
    :param epoch_size: epoch size L_epoch in the paper
    :param min_service_rate: lower bound of service rates, \delta in the paper

    :return:
        queue_length, a numpy array with shape (num_simulations, num_users, simulation_horizon)
    """

    queue_length = np.zeros((num_simulations, num_users, simulation_horizon), dtype=int)
    unscheduled_queue_length = np.zeros((num_simulations, num_users), dtype=int)
    N = np.zeros((num_simulations, num_users, num_servers), dtype=float)
    phi = np.zeros((num_simulations, num_users, num_servers), dtype=float)
    schedule = np.zeros((num_simulations, num_servers), dtype=int)
    schedule_decision = np.zeros((num_simulations, num_servers), dtype=int)
    M = np.zeros((num_simulations, num_servers), dtype=int)
    random_arrival = np.zeros((num_simulations, num_users), dtype=int)
    random_service = np.zeros((num_simulations, num_servers), dtype=int)
    finished = np.zeros((num_simulations, num_servers), dtype=bool)
    idling = np.zeros((num_simulations, num_servers), dtype=bool)
    period_arrival = arrival.shape[1]
    period_service = mu.shape[2]
    pre_epoch = -1

    for t in range(simulation_horizon):
        # the arrival rate and service rate at the current time slot
        arrival_temp = arrival[:, t % period_arrival].reshape((1, num_users))
        mu_temp = mu[:, :, t % period_service]

        # Updating queue length
        if t != 0:
            for i in range(num_users):
                queue_length[:, i, t] = queue_length[:, i, t - 1] + random_arrival[:, i] - np.sum((schedule == i)
                                                                                                  & finished, axis=1)

        # random arrival
        random_arrival = (rng.random(size=(num_simulations, num_users)) < arrival_temp)
        unscheduled_queue_length = unscheduled_queue_length + random_arrival

        # Updating memories
        if t != 0:
            M = M + 1
            for i in range(num_users):
                N[:, i, :] = N[:, i, :] + finished * (schedule == i)
                phi[:, i, :] = phi[:, i, :] + finished * (schedule == i) * M
            M[finished | idling] = 0

        # DAM.UCB Algorithm
        available = (M == 0)
        # which epoch
        epoch = t // epoch_size
        if epoch != pre_epoch:
            # if new epoch begins
            for j in range(num_servers):
                if t == 0:
                    schedule[:, j] = rng.integers(low=0, high=num_users, size=num_simulations)
                else:
                    temp_N = N[:, :, j]
                    temp_phi = phi[:, :, j]
                    temp_queue_length = queue_length[:, :, t]
                    zero_N_idx = (temp_N < 1e-20)
                    zero_phi_idx = (temp_phi < 1e-20)
                    b = np.zeros((num_simulations, num_users))
                    b[zero_N_idx] = 1
                    b[~zero_N_idx] = np.sqrt(2 * np.log(t+num_servers) / temp_N[~zero_N_idx])
                    b = np.minimum(b, 1.0)
                    pre_max = np.zeros((num_simulations, num_users))
                    temp_ucb = np.zeros((num_simulations, num_users), dtype=float)
                    temp_ucb[~zero_phi_idx] = temp_N[~zero_phi_idx] / temp_phi[~zero_phi_idx] + b[~zero_phi_idx]
                    temp_ucb[zero_phi_idx] = b[zero_phi_idx]
                    temp_ucb = np.maximum(min_service_rate, np.minimum(temp_ucb, 1))
                    pre_max = temp_queue_length * temp_ucb
                    schedule_decision[:, j] = np.argmax(pre_max, axis=1)
                    schedule[available[:, j], j] = schedule_decision[available[:, j], j]
            pre_epoch = epoch
        else:
            for j in range(num_servers):
                schedule[available[:, j], j] = schedule_decision[available[:, j], j]
        for j in range(num_servers):
            idling[:, j] = (unscheduled_queue_length[np.arange(num_simulations), schedule[:, j]] <= 0)
            for i in range(num_users):
                unscheduled_queue_length[:, i] = unscheduled_queue_length[:, i] - ((schedule[:, j] == i)
                                                                                   & available[:, j])

        unscheduled_queue_length[unscheduled_queue_length < 0] = 0
        idling = (idling & available)

        # setting the service time for the newly started jobs
        indices = schedule * num_servers + np.arange(num_servers).reshape((1, -1))
        temp = 1 / (mu_temp.flatten()[indices]) - 1
        random_service[available] = (rng.random(size=(num_simulations, num_servers))[available] < temp[available]) + 1

        # Departure
        finished = (M + 1 == random_service) & (~idling)

        if t % 1000 == 0:
            print("Time slot ", t, " finished;")

    return queue_length


def main():
    arrival = None
    mu = None
    if args.num_users > 1:
        if args.setting == "stationary":
            arrival = np.array([[0.75], [0.75]])
            mu = [[[0.9], [0.6]],
                  [[0.5], [1.0]]]
        elif args.setting == "aperiodic":
            arrival = np.array([[0.70], [0.70]])
            temp11 = np.linspace(0.90000, 0.60000, num=30000, endpoint=False)
            temp21 = np.linspace(0.50000, 0.80000, num=30000, endpoint=False)
            temp12 = np.linspace(0.60000, 0.90000, num=30000, endpoint=False)
            temp22 = np.linspace(1.00000, 0.70000, num=30000, endpoint=False)
            mu = [[temp11, temp12],
                  [temp21, temp22]]
            args.simulation_horizon = 30000
        elif args.setting == "periodic":
            temp = np.concatenate((np.arange(0.75, 0.55, -0.001), np.arange(0.55, 0.75, 0.001)))
            arrival = [temp, temp]
            temp11 = np.concatenate((np.arange(0.90, 0.50, -0.001), np.arange(0.50, 0.90, 0.001)))
            temp21 = np.concatenate((np.arange(0.50, 0.90, 0.001), np.arange(0.90, 0.50, -0.001)))
            temp12 = np.concatenate((np.arange(0.60, 1.00, 0.001), np.arange(1.00, 0.60, -0.001)))
            temp22 = np.concatenate((np.arange(1.00, 0.60, -0.001), np.arange(0.60, 1.00, 0.001)))
            mu = [[temp11, temp12],
                  [temp21, temp22]]
        elif args.setting == "periodic_larger_period":
            temp = np.concatenate((np.arange(0.75, 0.55, -0.0001), np.arange(0.55, 0.75, 0.0001)))
            arrival = [temp, temp]
            temp11 = np.concatenate((np.arange(0.90, 0.50, -0.0001), np.arange(0.50, 0.90, 0.0001)))
            temp21 = np.concatenate((np.arange(0.50, 0.90, 0.0001), np.arange(0.90, 0.50, -0.0001)))
            temp12 = np.concatenate((np.arange(0.60, 1.00, 0.0001), np.arange(1.00, 0.60, -0.0001)))
            temp22 = np.concatenate((np.arange(1.00, 0.60, -0.0001), np.arange(0.60, 1.00, 0.0001)))
            mu = [[temp11, temp12],
                  [temp21, temp22]]
        else:
            print("Error input of setting!")
            exit(1)
    else:
        arrival = np.array([[0.7]])
        mu = [[[0.8], [0.5]]]

    if args.num_users > 1:
        arrival = np.tile(np.array(arrival), (args.num_users // 2, 1))
        mu = np.tile(np.array(mu), (args.num_users // 2, args.num_servers // 2, 1))
    else:
        mu = np.tile(np.array(mu), (1, args.num_servers // 2, 1))

    queue_length = dam_ucb_max_weight(arrival=np.array(arrival), mu=np.array(mu), simulation_horizon=args.simulation_horizon)

    queue_length_mean = np.sum(np.mean(queue_length, axis=0), axis=0).reshape((queue_length.shape[2], 1))

    pd_data = pd.DataFrame(np.concatenate((np.arange(queue_length.shape[2]).reshape((queue_length.shape[2], 1)),
                                           queue_length_mean), axis=1), columns=["time", "queue length"])
    plt.figure(figsize=(10, 8), dpi=80)
    fig = sns.relplot(x="time", y='queue length', markers=True, kind="line", data=pd_data)
    fig.set_ylabels('Average Total Queue Length')
    plt.grid()
    plt.show()
    fig.savefig(FIGURE_FILE_PATH)

    np.save(RESULT_FILE_PATH, queue_length)
    print("finished!")
    return


if __name__ == "__main__":
    main()
