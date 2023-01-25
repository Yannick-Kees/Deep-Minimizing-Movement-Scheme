from packages import *

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
   

   
   
# enable computing on GPU if possible
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#############################
# Test-pointclouds ##########
#############################


g_quadrath = torch.tensor([ [.2,.2],[.2,.3],[.2,.4],[.2,.5],[.2,.6],[.2,.7],[.2,.8],[.8,.2],[.8,.3],[.8,.4],[.8,.5],[.8,.6],[.8,.7],[.8,.8] ,[.3,.2] ,[.3,.8] ,[.4,.2] ,[.4,.8] ,[.5,.2] ,[.5,.8],[.6,.2] ,[.6,.8],[.7,.2] ,[.7,.8]     ]    )


def signal(frequency):

    # define the range of x-values
    x_range = np.linspace(-5, 5, num=1000)

    # define the function for the curve
    def f_frequency(x):
        return np.sin(frequency*x) + np.cos(frequency*x)

    # evaluate the y-values of the curve over the given range
    y_values = f_frequency(x_range)

    # create 2D points by combining x and y values
    points = np.stack((x_range, y_values), axis=1)

    return Tensor(normalize(Tensor(points)))

#############################
# change pointclouds ########
#############################

def add_noise(pc):
    # Adding noise to every second point in pointcloud
    
    for i in range(len(pc)//2):
        pc[2*i] +=  uniform(-0.02,.02)
        
    return pc


def normalize(pc):
    # Scale all point of the point cloud in such a way, that every coordinate is betweeen -0.3 and 0.3 
    # That is how they are still away enough from the boundary
        
    pc = np.matrix(pc)
    
    if np.amin(pc)<0:
        pc = pc - np.amin(pc)
    
    norm = pc - np.amin(pc)
    norm = .6 * (1.0/np.amax(norm) *  norm  ) - .3
    return norm.tolist()


###################
# Plotting ########
###################

def draw_point_cloud(pc):
    # Plotting point cloud
    
    # Parameters:
    #   pc:      Tensor of points

   fig1, ax1 = plt.subplots()
   ax1.set_aspect('equal')
   pointcloud = pc.detach().numpy().T 
   plt.plot(pointcloud[0],pointcloud[1], '.')
   plt.xlim(-.5,.5)
   plt.ylim(-.5,.5)
   plt.show()
   return



def draw_phase_field_paper(f,x_,y_, i, film):
    # Creating Contour plot of f; visualization for the paper
    
    # Parameters:
    #   f:      Function to plot
    #   x_,y_:  Drawing the function on [0,x_] \times [0,y_]
    #   i:      Index number, for naming the image files
    #   film:   bool, weather to store the image file

    xlist = np.linspace(-x_, x_, 100)
    ylist = np.linspace(-y_, y_, 100)
    X, Y = np.meshgrid(xlist, ylist)
    Z = [[ f(Tensor([ X[i][j], Y[i][j] ] )).detach().numpy()[0]  for j in range(len(X[0]))  ] for i in range(len(X)) ] # Evaluate function in points
    
    fig = plt.figure()                                                      # Draw contour plot
    levels = np.arange(-1,1,.2)       # Specify contours/level set to plot
    contour = plt.contour(X, Y, Z, levels, colors='k')
    plt.clabel(contour, colors = 'k', fmt = '%2.1f', fontsize=12)
    contour_filled = plt.contourf(X,Y,Z, cmap = "cividis")
    plt.colorbar(contour_filled)
    if film:
        plt.savefig("images/pf" + str(i).zfill(5) + '.jpg')
        plt.close(fig)
    else:
        plt.show()
    return

   
   
 