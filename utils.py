"""
Some helper functions for PyTorch
"""

# Adapted from https://github.com/kuangliu/pytorch-cifar
import os
import sys
import time
import torch


class progressBar:
    def __init__(self, bar_length=65.0, term_width=None):
        self.bar_length = bar_length
        self.last_time = time.time()
        self.begin_time = self.last_time
        if term_width:
            self.term_width = term_width
        else:
            try:
                _, term_width = os.popen("stty size", "r").read().split()
                self.term_width = int(term_width)
            except:
                self.term_width = 30

    def next(self, current, total, msg=None):
        if current == 0:
            self.begin_time = time.time()  # Reset for new bar.

        cur_len = int(self.bar_length * current / total)
        rest_len = int(self.bar_length - cur_len) - 1

        sys.stdout.write(" [")
        for _ in range(cur_len):
            sys.stdout.write("=")
        sys.stdout.write(">")
        for _ in range(rest_len):
            sys.stdout.write(".")
        sys.stdout.write("]")

        cur_time = time.time()
        step_time = cur_time - self.last_time
        self.last_time = cur_time
        tot_time = cur_time - self.begin_time

        L = []
        L.append("  Step: %s" % self.format_time(step_time))
        L.append(" | Tot: %s" % self.format_time(tot_time))
        if msg:
            L.append(" | " + msg)

        msg = "".join(L)
        sys.stdout.write(msg)
        for _ in range(self.term_width - int(self.bar_length) - len(msg) - 3):
            sys.stdout.write(" ")

        # Go back to the center of the bar.
        for _ in range(self.term_width - int(self.bar_length / 2) + 2):
            sys.stdout.write("\b")
        sys.stdout.write(" %d/%d " % (current + 1, total))

        if current < total - 1:
            sys.stdout.write("\r")
        else:
            sys.stdout.write("\n")
        sys.stdout.flush()

    @staticmethod
    def format_time(seconds):
        days = int(seconds / 3600 / 24)
        seconds = seconds - days * 3600 * 24
        hours = int(seconds / 3600)
        seconds = seconds - hours * 3600
        minutes = int(seconds / 60)
        seconds = seconds - minutes * 60
        secondsf = int(seconds)
        seconds = seconds - secondsf
        millis = int(seconds * 1000)

        f = ""
        i = 1
        if days > 0:
            f += str(days) + "D"
            i += 1
        if hours > 0 and i <= 2:
            f += str(hours) + "h"
            i += 1
        if minutes > 0 and i <= 2:
            f += str(minutes) + "m"
            i += 1
        if secondsf > 0 and i <= 2:
            f += str(secondsf) + "s"
            i += 1
        if millis > 0 and i <= 2:
            f += str(millis) + "ms"
            i += 1
        if f == "":
            f = "0ms"
        return f


def L2(params, l2, device):
    """Computes L2-regularization.

    Arguments:
        params: Pytorch net paramateters
        l2: L2 coefficient.
    """

    if l2 <= 0.0:
        return 0.0
    l2_reg = torch.tensor(0.0, device=device)
    for param in params:
        l2_reg += param.norm(2) ** 2
    return l2_reg * l2 * 0.5
