from .base import base
from .models import *
from utils import progressBar, L2
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import sys


class cifar(base):
    def __init__(self):
        super().__init__()

    def parser(self):
        parser = argparse.ArgumentParser(
            description="CIFAR-10 experiment", add_help=False
        )
        parser.add_argument(
            "--batch",
            default=128,
            type=int,
            metavar="B",
            help="Size of the minibatches.",
        )

        parser.add_argument(
            "--progress",
            default="true",
            type=str,
            metavar="progress",
            help="Show the progress bar, default true.",
        )

        return parser

    def initialize(self, args, files):
        super().initialize(args, files)
        self.best_acc = 0
        self.net = ResNet18()
        self.progress = args.progress

        self.criterion = nn.CrossEntropyLoss()
        self.stats["acc train"] = list()
        self.stats["acc test"] = list()
        self.stats["loss train"] = list()
        self.stats["loss test"] = list()

        self.transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        self.transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        self.trainset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=self.transform_train
        )
        self.trainloader = torch.utils.data.DataLoader(
            self.trainset, batch_size=args.batch, shuffle=True, num_workers=2
        )

        self.testset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=self.transform_test
        )
        self.testloader = torch.utils.data.DataLoader(
            self.testset, batch_size=100, shuffle=False, num_workers=2
        )

    def train(self, epoch, optimizer):
        super().train(epoch, optimizer)
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            optimizer.zero_grad()
            outputs = self.net(inputs)
            l2_reg = L2(self.net.parameters(), self.l2, self.device)
            loss = self.criterion(outputs, targets) + l2_reg
            loss.backward()

            def closure():
                return loss

            optimizer.step(closure)

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            
            if  self.progress == "true":
                self.progress_bar.next(
                    batch_idx,
                    len(self.trainloader),
                    "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                    % (
                        train_loss / (batch_idx + 1),
                        100.0 * correct / total,
                        correct,
                        total,
                    ),
            )

        # We save the average loss per data point
        self.stats["loss train"].append(train_loss / len(self.trainloader))
        self.stats["acc train"].append(correct / total)

    def test(self, epoch):
        super().test(epoch)
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.testloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                if  self.progress == "true":
                        self.progress_bar.next(
                        batch_idx,
                        len(self.testloader),
                        "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                        % (
                            test_loss / (batch_idx + 1),
                            100.0 * correct / total,
                            correct,
                            total,
                        ),
                    )

        self.stats["loss test"].append(test_loss / len(self.testloader))
        self.stats["acc test"].append(correct / total)

        # Save checkpoint.
        acc = 100.0 * correct / total
        if acc > self.best_acc:
            if  self.progress == "true": print("Saving..")
            state = {
                "net": self.net.state_dict(),
                "acc": acc,
                "epoch": epoch,
            }
            torch.save(state, self.files["checkpoint"])
            self.best_acc = acc

    def save(self, epoch):
        super().save(epoch)
        plt.figure(1)

        # set the title
        plt.suptitle("Loss and Accuracy for CIFAR10")
        plt.title(" ".join(sys.argv[1:]))

        # plot the test and training losses
        plt.plot(self.stats["epoch"], self.stats["loss test"], "--.", color="tab:blue")
        plt.plot(self.stats["epoch"], self.stats["loss train"], "--.", color="tab:red")

        # label axes
        plt.xlabel("epoch")
        plt.ylabel("loss")

        # add additional axes to right hand side to include accuracy
        plt.gca().twinx()
        plt.ylabel("accuracy")

        # plot the test and training accuracies
        plt.plot(
            self.stats["epoch"],
            self.stats["acc test"],
            "--.",
            color="tab:blue",
            label="test",
        )
        plt.plot(
            self.stats["epoch"],
            self.stats["acc train"],
            "--.",
            color="tab:red",
            label="train",
        )

        # stylistic things, including legend
        plt.grid()
        plt.setp(plt.gca().spines.values(), color="#bfbfbf")
        lgd = plt.gca().legend()
        lgd.set_frame_on(False)

        # save the image
        plt.savefig(self.files["plot"])
        plt.clf()

        if  self.progress == "true": print("Image saved as: %s" % self.files["plot"])

    def resume(self, checkpoint):
        self.best_acc = checkpoint["acc"]
