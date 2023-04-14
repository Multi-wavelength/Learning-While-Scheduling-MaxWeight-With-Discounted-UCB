import numpy as np
import argparse
import os
import inspect
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

CUR_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PARENT_DIR = os.path.dirname(CUR_DIR)

parser = argparse.ArgumentParser()

parser.add_argument("--setting", default="stationary", help="stationary, aperiodic, periodic, periodic_larger_period")
parser.add_argument("--discount",
                    default="yes",
                    help='yes or no, whether or not use a discount factor.')
parser.add_argument("--ucb",
                    default="yes",
                    help='yes or no, whether or not use a UCB bonus.')
parser.add_argument("--arrival_rate_bound",
                    default=1.0, type=float,
                    help='the upper bound of the arrival rate')
parser.add_argument("--service_time_bound",
                    default=2.0, type=float,
                    help='the upper bound of the service time')
parser.add_argument("--T",
                    default=6071209191677812, type=int,
                    help='the constant integer T in the algorithm, should be sufficiently large.')
# Use T=6071209191677812, theoretical value for stationary setting
# for <=10 users, use T=8192 for nonstationary setting
# for >=12 users, use T=16384 for nonstationary setting
# for >=18 users, use T=32768 for nonstationary setting
parser.add_argument("--ucb_constant",
                    default=4.0, type=float,  # 4.0 works
                    help='the constant in the UCB bonus, 4.0 for proposed algorithm')
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

args = parser.parse_args()

foldername = "data_10_users_" + args.setting + "/max_weight"

# check the directory does not exist
if not (os.path.exists(foldername)):
    # create the directory
    os.makedirs(foldername)

RESULT_FILE_PATH = os.path.join(CUR_DIR, foldername, "discount_" + args.discount
                                + "_ucb_" + args.ucb + "_T_" + str(args.T) + ".npy")
FIGURE_FILE_PATH = os.path.join(CUR_DIR, foldername, "discount_" + args.discount
                                + "_ucb_" + args.ucb + "_T_" + str(args.T) + ".png")

rng = np.random.default_rng()


def max_weight(arrival, mu, discount=args.discount, ucb=args.ucb, service_time_bound=args.service_time_bound, T=args.T,
               ucb_constant=args.ucb_constant, num_users=args.num_users, num_servers=args.num_servers,
               simulation_horizon=args.simulation_horizon, num_simulations=args.num_simulations):
    """
    MaxWeight Algorithm

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
    :param discount: "yes" means we use a discounted average (exponential moving average)
                     "no" means we use a simple average
    :param ucb: "yes" means we use UCB
                "no" means we do not use UCB
    :param service_time_bound: the upper bound of the service time
    :param T: the constant integer T in the proposed algorithm
    :param ucb_constant: the constant in the UCB bonus term
    :param num_users: number of users
    :param num_servers: number of servers
    :param simulation_horizon: an integer, the simulation horizon
    :param num_simulations: number of simulations in parallel

    :return:
        queue_length, a numpy array with shape (num_simulations, num_users, simulation_horizon)
    """

    if discount == "yes":
        gamma = 1 - 8 * np.log(T) / T
    else:
        gamma = 1
    if ucb == "yes":
        ucb_c = ucb_constant
    else:
        ucb_c = 0
    queue_length = np.zeros((num_simulations, num_users, simulation_horizon), dtype=int)
    unscheduled_queue_length = np.zeros((num_simulations, num_users), dtype=int)
    N = np.zeros((num_simulations, num_users, num_servers), dtype=float)
    phi = np.zeros((num_simulations, num_users, num_servers), dtype=float)
    schedule = np.zeros((num_simulations, num_servers), dtype=int)
    M = np.zeros((num_simulations, num_servers), dtype=int)
    random_arrival = np.zeros((num_simulations, num_users), dtype=int)
    random_service = np.zeros((num_simulations, num_servers), dtype=int)
    finished = np.zeros((num_simulations, num_servers), dtype=bool)
    idling = np.zeros((num_simulations, num_servers), dtype=bool)
    period_arrival = arrival.shape[1]
    period_service = mu.shape[2]
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
                N[:, i, :] = gamma * N[:, i, :] + (gamma ** (M - 1)) * finished * (schedule == i)
                phi[:, i, :] = gamma * phi[:, i, :] + (gamma ** (M - 1)) * finished * (schedule == i) * M
            M[finished | idling] = 0

        # MaxWeight algorithm
        available = (M == 0)

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
                if ucb == "yes":
                    b[zero_N_idx] = 1
                    if discount == "yes":
                        b[~zero_N_idx] = ucb_c * service_time_bound * np.sqrt(np.log(T) / temp_N[~zero_N_idx])
                    else:
                        b[~zero_N_idx] = ucb_c * service_time_bound * np.sqrt(np.log(t) / temp_N[~zero_N_idx])
                    b = np.minimum(b, 1.0)
                pre_max = np.zeros((num_simulations, num_users))
                pre_max[~zero_phi_idx] = temp_queue_length[~zero_phi_idx] * (temp_N[~zero_phi_idx]
                                                                             / temp_phi[~zero_phi_idx]
                                                                             + b[~zero_phi_idx])
                pre_max[zero_phi_idx] = temp_queue_length[zero_phi_idx] * b[zero_phi_idx]
                schedule_temp = np.argmax(pre_max, axis=1)
                schedule[available[:, j], j] = schedule_temp[available[:, j]]

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

    queue_length = max_weight(arrival=np.array(arrival), mu=np.array(mu), simulation_horizon=args.simulation_horizon)

    queue_length_mean = np.sum(np.mean(queue_length, axis=0), axis=0).reshape((queue_length.shape[2], 1))

    pd_data = pd.DataFrame(np.concatenate((np.arange(queue_length.shape[2]).reshape((queue_length.shape[2], 1)),
                                           queue_length_mean), axis=1), columns=["time", "queue length"])
    plt.figure(figsize=(10, 8), dpi=80)
    fig = sns.relplot(x="time", y='queue length', markers=False, kind="line", data=pd_data)
    fig.set_ylabels('Average Total Queue Length')
    plt.grid()
    plt.show()
    fig.savefig(FIGURE_FILE_PATH)

    np.save(RESULT_FILE_PATH, queue_length)
    print("finished!")
    return


if __name__ == "__main__":
    main()
