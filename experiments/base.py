from utils import progressBar
import json

from pdb import set_trace as bp
import os

class base:
    r"""Base class for all experiments.

    All experiment classes should inherit this class and
    override all its methods.
    """

    def __init__(self):
        self.stats = dict()
        self.progress_bar = progressBar()

    def parser(self):
        """Return an argparser with a description of the
        experiment and all the arguments of the
        experiment.
        """
        pass

    def initialize(self, args, files):
        """Initializes the experiment.

        Arguments:
            args (iterable): A list of arguments provided from
                the command line or the defaults defined in
                parser
        """

        self.stats["epoch"] = list()
        self.device = args.device
        self.l2 = args.l2
        self.files = files

        # If 1 means to save at each epoch
        self.saving_epochs = 1
        self.start_epoch = 0

    def resume(self, checkpoint):
        """Resume the training from a checkpoint.

        Arguments:
            checkpoint: Pytorch checkpoint
        """

        self.net.load_state_dict(checkpoint["net"])
        self.start_epoch = checkpoint["epoch"]+1
        
        if os.path.isfile(self.files["stats"]):
             print("Loading the previous stats...")
             with open(self.files['stats'], 'r') as f: self.stats = json.load(f)
             
        else:
            print("Stats file not found. Starting with a new one.")
        
    def train(self, epoch, optimizer):
        """Performs an epoch of training.

        Arguments:
            epoch (int): Epoch number.
            optimizer: Optimizer instance to use.
        """

        self.stats["epoch"].append(epoch + 1)
        self.net.train()

    def test(self, epoch):
        """Performs an epoch of testing.

        Arguments:
            epoch (int): Epoch number.
        """

        self.net.eval()

    def save(self, epoch):
        """Saves an epoch. Saving the stats is done in this
        function. Normally, the experiment should just produce
        the relevant plots.

        Arguments:
            epoch (int): Epoch number.
        """

        # saving stats
        
        if (epoch%self.saving_epochs == 0) or (epoch == self.start_epoch+self.epochs-1)  :
            with open(self.files["stats"], "w") as f:
                json.dump(self.stats, f, sort_keys=True, indent=4)
                if  self.progress == "true": print("Stats saved as: %s" % self.files["stats"])
