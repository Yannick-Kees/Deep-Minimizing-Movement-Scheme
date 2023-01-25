from misc import *


def report_progress(current, total, error):
   # Prints out, how far the training process is

   # Parameters:
   #     current:    where we are right now
   #     total:      how much to go
   #     error:      Current Error, i.e. evaluation of the loss functional
   
   sys.stdout.write('\rProgress: {:.2%}, Current Error: {:}'.format(float(current)/total, error))
   
   if current==total:
      sys.stdout.write('\n')
      
   sys.stdout.flush()

class Network(nn.Module):
    
    #   !! COPYRIGHT for this network class : https://github.com/amosgropp/IGR/blob/master/code/model/network.py  !!

    def __init__(
        self,
        d_in, 
        dims,
        skip_in=(),
        radius_init=.3,
        beta=100
    ):
        # The network structure, that was proposed in Park et al.
        
        # Parameters:
        #   self:               Neuronal Network
        #   d_in:               dimension of points in point cloud
        #   dims:               array, where dims[i] are the number of neurons in layer i
        #   skip_in:            array containg the layer indices for skipping layers
        #   radius_init:        Radius for Geometric initialisation
        #   beta:               Value for softmax activation function

        
        super().__init__()
        
        self.d_in = d_in
        
        dims = [self.d_in] + dims + [1]     # Number of neurons in each layer
        
        self.num_layers = len(dims)         # Number of layers
        self.skip_in = skip_in              # Skipping layers

        for layer in range(0, self.num_layers - 1):

            if layer + 1 in skip_in:
                out_dim = dims[layer + 1] - d_in
                
            else:
                out_dim = dims[layer + 1]

            lin = nn.Linear(dims[layer], out_dim) # Affine linear transformation

            setattr(self, "lin" + str(layer), lin) # Save layer

        if beta > 0:
            self.activation = nn.Softplus(beta=beta) # Softplus activation function
            #self.activation = nn.Sigmoid()

        # vanilla ReLu
        else:
            self.activation = nn.ReLU()

    def forward(self, input):
        # forward pass of the NN

        x = input
        for layer in range(0, self.num_layers - 1):

            lin = getattr(self, "lin" + str(layer))

            if layer in self.skip_in:
                # Skipping layer
                
                x = torch.cat([x, input], -1) / np.sqrt(2)

            x = lin(x) # Apply layer

            if layer < self.num_layers - 2:
                x = self.activation(x)

        return x
    