""" This file is the main starting point for all experiments """
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import os
import argparse
import sys
from inflation import BBI
import numpy as np

# Import all supported experiments
from experiments.cifar import cifar
from experiments.PDE_PoissonD import PDE_PoissonD

import json 


parser = argparse.ArgumentParser(
    description="Perform various experiments in AI/Physics"
)
parser.add_argument(
    "experiment",
    type=str,
    help="Name of the experiment to run. Can be everything in the experiments directory",
    choices=["cifar", "PDE_PoissonD"]
)

args_main = parser.parse_args(sys.argv[1:2])
experiment = eval(args_main.experiment + "()")
parser = argparse.ArgumentParser(
    parents=[experiment.parser()], prog=sys.argv[0] + " " + sys.argv[1]
)

parser.add_argument(
    "-n",
    "--name",
    type=str,
    default=None,
    help="Name to give the stats, plot, and checkpoint. If omitted, the name experiment-optimizer will be used.",
)
parser.add_argument(
    "-r", "--resume", action="store_true", help="resume from checkpoint"
)
parser.add_argument(
    "-d",
    "--device",
    default="cuda",
    type=str,
    help="The device where to run the experiment. Can be: cuda or cpu. It defaults to cuda if it is available.",
    choices=["cuda", "cpu"],
)

parser.add_argument("--lr", default=0.1, type=float, help="Learning rate")
parser.add_argument(
    "-e", "--epochs", default=10, type=int, help="Total number of training epochs."
)
parser.add_argument(
    "-s",
    "--seed",
    default=None,
    type=int,
    help="Seed for reproducible results. If omitted, no seed will be used.",
)
parser.add_argument(
    "-o",
    "--optimizer",
    default="sgd",
    type=str,
    help="Optimizer to use. Can be: {sgd, BBI}",
    choices=["sgd", "BBI"],
)

parser.add_argument("--threshold0", default=1000, type=int, help="Threshold0 for the BBI optimizer.")
parser.add_argument("--threshold", default=2000, type=int, help="Threshold for the BBI optimizer")
parser.add_argument("--nFixedBounces", default=1, type=int, help="Number of fixed bounces, regardless of progress. Default 1.")

parser.add_argument("--deltaEn", default=0.0, type=float, help="Extra energy for the BBI optimizer")
parser.add_argument("--consEn", default="true", type=str, help="Force energy conservation for the BBI optimizer")

# Set to zero by default because for BBI weight decays is different than L^2
parser.add_argument("--alpha", default=0.0, type=float, help="Weight decay.")

parser.add_argument(
    "--l2",
    default=0.0,
    type=float,
    help="L2-regularization.",
)

parser.add_argument("--rho", default=0.9, type=float, help="Momentum.")

parser.add_argument("--v0", default=0, type=float, help="V_0 for the DBI optimizer.")
args = parser.parse_args(sys.argv[2:])

if (args.consEn is None) or (args.consEn == "true") or (args.consEn == "True"):
    args.consEn = True
else: args.consEn = False

if args.name is None:
    args.name = args_main.experiment + "-" + args.optimizer
    
# Set the seed and use deterministic algorithms for reproducibilty
if args.seed is not None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    deterministic = True
    print(np.random.get_state()[1][0])
else: deterministic = False

# Saving parameters
files = dict()
files["stats"] = "./results/" + args.name + ".json"
files["plot"] = "./results/" + args.name + ".png"
# Used by the PDE experiment
files["plot_function"] = "./results/" + args.name + "-function.png"
files["plot_check_losses"] = "./results/" + args.name + "-check_losses.png"
files["parameters"] = "./results/" + args.name + "-parameters.json"
files["inputs"] = "./results/" + args.name + "-training-inputs.json"

files["checkpoint"] = "./checkpoints/" + args.name + "-ckpt.pth"
if not os.path.isdir("checkpoints"):
    os.mkdir("checkpoints")
if not os.path.isdir("results"):
    os.mkdir("results")

if args.device == "cuda" and torch.cuda.is_available():
    args.device = torch.device("cuda")
else:
    args.device = torch.device("cpu")
print("The experiment will run on", args.device)

print("===> Initializing experiment...")
experiment.initialize(args, files)
experiment.net = experiment.net.to(args.device)

# Set-up the optimizer
if args.optimizer == "sgd":
    optimizer = optim.SGD(
        experiment.net.parameters(),
        lr=args.lr,
        momentum=args.rho,
        weight_decay=args.alpha,
    )


elif args.optimizer == "BBI":
    optimizer = BBI(
        experiment.net.parameters(),
        lr=args.lr,
        v0=args.v0,
        threshold0 = args.threshold0,
        threshold = args.threshold,
        deltaEn = args.deltaEn,
        consEn = args.consEn,
       n_fixed_bounces = args.nFixedBounces
    )


start_epoch = 0  # Start from epoch 0 or last checkpoint epoch

if args.device.type == "cuda" and torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs")
    experiment.net = torch.nn.DataParallel(experiment.net)

if args.device.type == "cuda":
    if (deterministic):
        torch.cuda.manual_seed(args.seed)
        cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8" # See here https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
    else:
        cudnn.benchmark = True
    
if args.resume:
    # Load checkpoint
    print("==> Resuming from checkpoint..")
    assert os.path.isfile(files["checkpoint"]), "Error: checkpoint not found!"
    checkpoint = torch.load(files["checkpoint"])
    experiment.resume(checkpoint)
    start_epoch = checkpoint["epoch"]+1
    
for epoch in range(start_epoch, start_epoch + args.epochs):
    print(args.name)
    print("\nEpoch: %d" % epoch)
    experiment.train(epoch, optimizer)
    experiment.test(epoch)
    experiment.save(epoch)
