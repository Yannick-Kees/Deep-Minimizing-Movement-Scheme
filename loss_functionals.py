from networks import *

def gradient(inputs, outputs):
    
    #  !! COPYRIGHT for this function : https://github.com/amosgropp/IGR/blob/master/code/model/network.py  !!

    # Returns:
    #   Pointwise gradient estimation [ Df(x_i) ]
    
    # Parameters:
    #   inputs:     [ x_i ]
    #   outputs:    [ f(x_i) ] 
    
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = torch.autograd.grad(outputs=outputs, inputs=inputs, grad_outputs=d_points, create_graph=True, retain_graph=True, only_inputs=True)[0][:,-3:]
    return points_grad

#############################
# Ambrosio Tortorelli #######
#############################

# One well potential
U = lambda s: (s- torch.tensor([1.0]).to(device))**2


def AT_Phasefield(f, eps, n, d):
    # Returns:
    #   Monte Carlo Integral of int_{[0,1]^2} W(u(x)) + eps * |Du(x)|^2 dx
    
    # Parameters:
    #   f:      Function to evaluate
    #   eps:    Epsilon
    #   n:      Number of samples drawn in the Monte Carlo Algorithm
    #   d:      Dimension of point cloud
     
    start_points = Variable(torch.rand(n, d), requires_grad =True)-torch.full(size=(n,d), fill_value=.5)   # Create random points [ x_i ]
    start_points = start_points.to(device)                          # Move points to GPU if possible
    gradients = gradient(start_points, f(start_points))             # Calculate their gradients [ Dx_i ]
    norms = gradients.norm(2,dim=-1)**2                             # [ |Dx_i| ]

    return ( (1.0/(4*eps))  * U(f(start_points))+eps*norms).mean()                    # returns 1/n * sum_{i=1}^n W(u(x_i)) + eps * |Du(x_i)|^2



def Zero_recontruction_loss_AT(f, pc, eps, m, c, d):
    # Returns:
    #   Monte Carlo Estimation of C * eps^(1/3) * 1/|X| * \sum_{x\in X} |\dashint_{B_delta}(x) u(s) ds|
    
    # Parameters:
    #   f:      Function to evaluate
    #   pc:     Pointcloud X
    #   eps:    Epsilon
    #   c:      Constant
    #   m:      Number of samples drawn in the Monte Carlo Algorithm
    #   d:      Dimension of point cloud
    

    return  c*eps**(-1.0/3.0) *  ( torch.abs(f(pc)).mean() )            # returns C * eps^(1/3) * 1/|X| * \sum_{x\in X} |\dashint_{B_delta(x)} g( u(x) ) dx|



def AT_loss(f, pointcloud, eps, n, m, c):
    # Returns:
    #   PHASE Loss = e^(-.5)(\int_\Omega W(u) +e|Du|^2 + Ce(^.3)/(n) sum_{p\in P} \dashint u ) + \mu/n \sum_{p\in P} |1-|w||
    
    # Parameters:
    #   f:      Function to evaluate
    #   pc:     Pointcloud X = [ x_i ]
    #   eps:    Epsilon
    #   n:      Number of Sample for Monte-Carlo in int_\Omega
    #   m:      Number of Sample for Monte-Carlo in int_{B_\delta}
    #   c:      Constant C, contribution of Zero recontruction loss
    
    d = pointcloud.shape[1] # dimension of point cloud
    
    return AT_Phasefield(f, eps, n, d) +  Zero_recontruction_loss_AT(f, pointcloud, eps, m, c, d)


def sobolev(f,g, MCS, Tau):
    start_points = Variable(torch.rand(MCS, 2), requires_grad =True)-torch.full(size=(MCS,2), fill_value=.5)   # Create random points [ x_i ]
    start_points = start_points.to(device)                          # Move points to GPU if possible
    L2 = (f(start_points) - g(start_points))**2
  
    gradients_f = gradient(start_points, f(start_points))             
    gradients_g = gradient(start_points, g(start_points))
    gradients = gradients_f - gradients_g
    H1 = gradients.norm(2,dim=-1)**2                             # [ |Dx_i| ]

    return L2.mean() + Tau * H1.mean()

