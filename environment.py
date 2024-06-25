# give a template of the arguments
from argparse import Namespace
import numpy as np
from torch import nn
from resnet import resnet18

class Environment(object):
    """
        This class is used to define the environment of the simulation
        
        Parameters:
            n_users: number of users in the simulation
            bandwidth: bandwidth of the communication channel (1 to 20 MHz)
            signal_power: signal power of the transmission (-67 to -50 dBm)
            noise_power: noise power of the transmission -90 dBm
            communication_rate: bandwidth * log2(1 + signal_power / noise_power)
            data_size: size of the dataset (bit)
            model_size: size of the model (bit)
            bit_per_cycle: number of bits per cycle
            cycle_frequency: cycle frequency of computation (10 to 15 MHz)
            Ex, Ey, Er: position and radius of the edge server 
            l_tol: tolerance distance of an async group
            T_com: communication time of model transmission
            T_itr: iteration time of model training in one epoch
    """

    def __init__(self, model: nn.Module, opts: Namespace):
        self.__dict__.update(vars(opts))

        self.bandwidth = np.random.uniform(10, 20, self.n_users) * 1000000
        # self.signal_power = dBm_to_mW(np.random.uniform(-67, -50, n_users))
        self.signal_power = dBm_to_mW(np.random.uniform(43, 46, self.n_users))
        self.noise_power = dBm_to_mW(-90)
        self.communication_rate = self.bandwidth * np.log2(1 + self.signal_power / self.noise_power)
        

        self.dataset_size = get_dataset_size(self.dataset, self.n_sample)
        self.model_size = 32 * sum(p.numel() for p in model.parameters() if p.requires_grad)
        # self.cycle_per_bit = 3 # 取址，译码，执行
        self.cycle_per_bit = 1 # 三级流水线

        self.cycle_frequency = np.random.uniform(80, 100, self.n_users) * 1000000

        self.Ex, self.Ey, self.Er = 40, 400, 50

        self.T_com = self.model_size / self.communication_rate
        self.T_itr = self.dataset_size * self.cycle_per_bit / self.cycle_frequency

        self.show()

    def show(self):
        print("==================================================")
        print("Number of users: ", self.n_users)
        print("Bandwidth (head 3): ", self.bandwidth[:3])
        print("Signal power (head 3): ", self.signal_power[:3])
        print("Noise power: ", self.noise_power)
        print("Communication rate (head 3): ", self.communication_rate[:3])
        print("Dataset size: ", self.dataset_size)
        print("Model size: ", self.model_size)
        print("Cycles per bit: ", self.cycle_per_bit)
        print("Cycle frequency (head 3): ", self.cycle_frequency[:3])
        print("Edge server position (Ex, Ey): ", (self.Ex, self.Ey))
        print("Edge server radius (Er): ", self.Er)
        print("Tolerance distance: ", self.l_tol)
        print("T_com (head 3): ", self.T_com[:3])
        print("T_itr (head 3): ", self.T_itr[:3])
        print("==================================================")


def get_dataset_size(dataset, n_sample):
    dataset_size_map = {
        'mnist': 28 * 28 * 8 * n_sample,
        'cifar10': 32 * 32 * 24 * n_sample,
        'cifar100': 32 * 32 * 24 * n_sample,
    }
    return dataset_size_map[dataset]

def dBm_to_mW(dbm):
    return np.power(10, dbm / 10.0)

if __name__ == "__main__":
    env = Environment(10, 'mnist', resnet18(num_classes=10))
    env.show()

