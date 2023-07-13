import argparse
import mindspore
from mindspore import nn
from mindspore.train import Model
from .dataset import create_dataset
from .moe import MoE

parser = argparse.ArgumentParser()

parser.add_argument('-n', '--n-experts', type = int, metavar = 'N', 
        help = 'Number of experts')
parser.add_argument('-d', '--dataset', type = str, metavar = 'D', default = 'MNIST',
        help = 'Dataset to be used. Must be either MNIST or CIFAR10.')
parser.add_argument('-e', '--epochs', type = int, metavar = 'E', default = 10,
        help = 'Number of epochs to train for.')
parser.add_argument('-w', '--use-wandb', action = 'store_true',
        help = 'If set, log metrics to Weights and Biases')
args = parser.parse_args()

network = MoE(args.n_experts)
loss = nn.CrossEntropyLoss()
optimizer = nn.Adam(network.trainable_params(), 1e-3)

trainer = Model(network, loss, optimizer, metrics=['accuracy'])
trainer.fit(args.epochs, train_dataset, valid_dataset)