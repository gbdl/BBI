from torch.autograd.grad_mode import no_grad
from .base import base
import argparse
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import sys
import numpy as np

from derivs import *


# generates point in the interior and on the boundary of a unit ball in d dimensions.
# It also puts all of them together

def generate_points(dimension, npoints_interior, npoints_boundary):

    #(Muller method)
    # It returns an array
    
    x_array = []
    ## Points in the interior
    for i in range(npoints_interior):
        u = np.random.normal(0,1,dimension)
        r = np.random.rand()**(1.0/dimension)
        w = r*u/np.sqrt(np.sum(u**2))
        x_array.append(w)

    ## Points on the boundary

    for i in range(npoints_boundary):
        u = np.random.normal(0,1,dimension)
        w = u/np.sqrt(np.sum(u**2))
        x_array.append(w)

    return x_array

# defines the network for the PDE problem
class network_PDE(torch.nn.Module):
    def __init__(self):
        # call constructor from superclass
        super().__init__()
        
        self.depth = 3

        self.width = 200
        
        # here define the layers
        # first layer
        self.linear_input = torch.nn.Linear(2, self.width)
        # middle layers
        self.linear_layers = torch.nn.ModuleList([torch.nn.Linear(self.width, self.width) for i in range(self.depth)])
        # final layer
        self.linear_output = torch.nn.Linear(self.width, 1)
        
        # here we initialize the network
        torch.nn.init.xavier_normal_(self.linear_input.weight, gain=torch.sqrt(torch.tensor(1.)))
        
        for i in range(self.depth):
            torch.nn.init.xavier_normal_(self.linear_layers[i].weight, gain=torch.sqrt(torch.tensor(1.)))
        
        
        torch.nn.init.xavier_normal_(self.linear_output.weight, gain=torch.sqrt(torch.tensor(1.)))    
            
    def forward(self, x):
        # This is a simple residual NN.
        # The difference with a fully connected is the x+ ... in each intermediate layer
        # notice that by design all the intermediate layers have the same width
       
        x = torch.sigmoid(self.linear_input(x))
        
        for layer in (self.linear_layers):
            x = x + torch.sigmoid(layer(x))
        
        x = self.linear_output(x)
            
        return x
    
class PDE_PoissonD(base):
    def __init__(self):
        super().__init__()

    def parser(self):
        parser = argparse.ArgumentParser(
            description="Using NNs to solve Poisson equations in arbitrary dimensions", add_help=False
        )
        parser.add_argument(
            "--iterations",
            default=1000,
            type=int,
            metavar="iterations_per_epoch",
            help="Number of iterations for each epoch.",
        )

        parser.add_argument(
            "--progress",
            default="true",
            type=str,
            metavar="progress",
            help="Show the progress bar, default true.",
        )

        parser.add_argument(
            "--feature",
            default=0,
            type=int,
            metavar="feature",
            help="The feature number to add to the loss. Defaults to 0 (no feature).",
        )

        return parser

    # Here the different features to add to the loss

    def feature_1(self):
        # just \theta^2 (L2)
        barrier_smooth = torch.tensor(0., device=self.device)
        for param in self.net.parameters():
            barrier_smooth += 2e-4*torch.sum(torch.pow(param,2))
        return barrier_smooth


    def feature_42(self):
        # This is an "empty" feature. 
        barrier_smooth = torch.tensor(0., device=self.device)
        return barrier_smooth
    
    
    def initialize(self, args, files):
        super().initialize(args, files)
        
        # If "True" generates new points in the domain at each epoch (adds noise)
        self.update_points = False

        self.dtype = torch.float32
        
        self.iterations_per_epoch = args.iterations
        self.criterion = torch.nn.MSELoss()
        
        self.epochs = args.epochs
        self.progress = args.progress

        self.criterion_test = torch.nn.L1Loss()

        self.stats["loss train"] = list()
        self.stats["solution"] = list()

        self.stats["losses check"] = list()
        self.stats["parameters"] = list()
        self.stats["inputs"] = list()
        
        ## The equation is Delta u +coeff*u^2 = inhom
        ## we impose also Dirichlet boundary conditions
         
        self.dimension = 2
        self.non_linearity = 1.0
        self.analytic_sol = lambda x: torch.sin(20*torch.norm(x,dim = 1)**2)**2
        self.inhom_term = lambda x: 0.125*( 3-4*(1+6400*torch.norm(x,dim = 1)**2)*torch.cos(40*(torch.norm(x,dim = 1)**2))+torch.cos(80*(torch.norm(x,dim = 1)**2))-640*torch.sin(40*(torch.norm(x,dim = 1)**2)))
        self.Dirichlet_value = 0.833469
        
        # network
        self.net = network_PDE().to(self.device)

        self.npoints_interior = int(1e4)
        self.npoints_boundary = int(1e3)

        # The weight factor of the boundary conditions in the loss
        self.bc_weight = 100000.0


        #If this is 1 it saves the full model at each epoch, otherwise only each self.saving.epochs epochs
        # The features, loss, and plots are always saved after each epoch
        self.saving_epochs = 1000
        
        
        ## Remember to add the feature also here!
        if args.feature == 1:
            self.feature = self.feature_1
        
        elif args.feature == 42:
            self.feature = self.feature_42
        
        else: 
            if (self.progress == 'true'): print("Feature not defined, or no feature specified. Working without features")
            self.feature = lambda : torch.tensor(0., device=self.device)
        
        
        #Points
        x_array = generate_points(self.dimension, self.npoints_interior, self.npoints_boundary)

        self.x = torch.tensor(x_array, device=self.device, requires_grad=True, dtype = self.dtype)
        
        ### Check the analytic solution
        points = torch.tensor(x_array, device='cpu', requires_grad=True, dtype = self.dtype)
        f1 = self.analytic_sol(points)
        f1p = diff_single(f1, points, device = 'cpu')
        f1pp = diff_many(f1p, points, device = 'cpu')

        lap = torch.zeros_like(f1pp[0][0], device='cpu')
        for i in range(self.dimension): lap-= f1pp[i][i] 
        diff_eq1 =lap + self.non_linearity*(f1**2) - self.inhom_term(points)
        
        if (self.progress == 'true'): print("L1 error of the analytic solution (should be small): ", torch.norm(diff_eq1).item())
        if (self.update_points == False): self.stats["inputs"].append(self.x.tolist())
            
        
    def train(self, epoch, optimizer):
        super().train(epoch, optimizer)
        
        for i in range(self.iterations_per_epoch):
            

            y = self.net(self.x)
            loss = self.get_loss(y)
            
            # kill if there are nans
            if (torch.isnan(loss).any().item() == True): sys.exit()

            #bp()
            optimizer.zero_grad()
            loss.backward()
            
            if  self.progress == "true":
                if i == 0: print("Starting loss: ", loss.item())
            
            def closure():
                return loss

            optimizer.step(closure)
            if  self.progress == "true":
                self.progress_bar.next(
                    i, self.iterations_per_epoch, "Loss: %.3f" % (loss.item())
                )

        if (self.update_points == True):
            x_array = generate_points(self.dimension, self.npoints_interior, self.npoints_boundary)
            self.x = torch.tensor(x_array, device=self.device, requires_grad=True, dtype = self.dtype)


        self.stats["loss train"].append(loss.item())
        self.stats["solution"].append(y.tolist())
        
        if (epoch%self.saving_epochs == 0) or (epoch == self.start_epoch+self.epochs-1):  
            new_list = list(self.net.parameters())
            list_pars = []
            for item in new_list:
                list_pars.append(item.tolist())
  
            self.stats["parameters"].append(list_pars)

    def test(self, epoch):
        super().test(epoch)

        y = self.net(self.x)
        losses = self.get_loss_test(y)
        if  self.progress == "true": print(losses)
        
        #if (epoch%self.saving_epochs == 0):
        state = {
            "net": self.net.state_dict(),
            "epoch": epoch,
        }

        torch.save(state, self.files["checkpoint"])

        self.stats["losses check"].append(losses)
        
         # saving plot loss
        plt.figure(1)

        # set the title
        plt.suptitle("Inhomogeneous Poisson")
        plt.suptitle(" ".join(sys.argv[1:]))

        ii = 0
        for loss in list(zip(*self.stats["losses check"])):
            
            if (ii == 0): lab ="log10(L1 error eq 1)"
            if (ii == 1): lab ="log10(L1 error bc 1)"
            if (ii == 2): lab ="log10(max abs error eq 1)"
            
            plt.plot(
                self.stats["epoch"],
                np.log10(loss),
                "--.",
                #color="tab:red",
                label = lab,
            )
            plt.grid()
            ii += 1

        plt.setp(plt.gca().spines.values(), color="#bfbfbf")
        lgd = plt.gca().legend()
        lgd.set_frame_on(True)

        # save the image
        plt.savefig(self.files["plot_check_losses"])
        plt.clf()
        plt.close()        


    def save(self, epoch):
        super().save(epoch)

        # saving plot loss
        plt.figure(1)

        # set the title
        plt.suptitle("Solving higher dimensional PDE")
        plt.suptitle(" ".join(sys.argv[1:]))

        plt.plot(
            self.stats["epoch"],
            np.log10(self.stats["loss train"]),
            "--.",
            color="tab:red",
            label="log10(loss)",
        )
        plt.grid()

        plt.setp(plt.gca().spines.values(), color="#bfbfbf")
        lgd = plt.gca().legend()
        lgd.set_frame_on(True)

        # save the image
        plt.savefig(self.files["plot"])
        plt.clf()
        plt.close()

        #save the first function
    
        starting_boundary =self.npoints_interior
        
        
        #plot a radial slice
        ax = plt.axes()

        points_plot = []
        elems = np.linspace(-1, 1, 1000)
        xs = np.array(elems)
        
        # create the radial slice
        for elem in elems: 
            temp = np.zeros(self.dimension-1)
            temp = np.insert(temp,0,elem)
            points_plot.append(temp)

        xs2 = torch.tensor(points_plot, device = self.device, dtype = self.dtype, requires_grad = True)
        
        with torch.no_grad():
            pred = np.array(self.net(xs2).tolist())
            analytic = np.array(self.analytic_sol(xs2).tolist()) 

        
        ax.plot(xs, 
            pred,
            ".",
            color="tab:orange",
            label="NN")
        ax.plot(xs, 
            analytic,
            ".",
            color="tab:blue",
            label="analytic")
        #bp()
        
        plt.grid()

        plt.setp(plt.gca().spines.values(), color="#bfbfbf")
        lgd = plt.gca().legend()
        lgd.set_frame_on(True)

        # save the image
        plt.savefig(self.files["plot_function"])
        plt.clf()
        plt.close()

        if  self.progress == "true": print("Image saved as: %s" % self.files["plot"])
        

    def get_loss(self, y):
        f1 = y.transpose(0,1)[0]
       
        f1p = diff_single(f1, self.x, self.device) 
        f1pp = diff_many(f1p, self.x, self.device) 

        
        starting_boundary =self.npoints_interior

        lap = torch.zeros_like(f1pp[0][0], device=self.device)
        for i in range(self.dimension): lap-= f1pp[i][i] 
        diff_eq1 =lap + self.non_linearity*(f1**2) - self.inhom_term(self.x)
        
        eqn1 =  self.criterion(diff_eq1, torch.zeros_like(diff_eq1))+self.feature()
        
        if self.bc_weight == 0: return eqn1
        else: 
            bc1 =  self.criterion(f1[starting_boundary:], torch.full_like(f1[starting_boundary:],self.Dirichlet_value))
            return eqn1+self.bc_weight*bc1

    def get_loss_test(self, y):
    
        f1 = y.transpose(0,1)[0]
   
        f1p = diff_single(f1, self.x, self.device) 
        f1pp = diff_many(f1p, self.x, self.device) 

        
        starting_boundary =self.npoints_interior

        bc1 =  self.criterion_test(f1[starting_boundary:], torch.full_like(f1[starting_boundary:],self.Dirichlet_value))
        
        lap = torch.zeros_like(f1pp[0][0], device=self.device)
        for i in range(self.dimension): lap-= f1pp[i][i] 
        diff_eq1 =lap + self.non_linearity*(f1**2) - self.inhom_term(self.x)
        
        
        eqn1 =  self.criterion_test(diff_eq1, torch.zeros_like(diff_eq1))
        
        max1 = (torch.max(torch.abs(diff_eq1)))
        
        return (eqn1.item(), bc1.item(), max1.item())