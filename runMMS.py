from loss_functionals import *

####################
# Settings #########
####################

# Neuronal Network
NUM_TRAINING_SESSIONS = 200
START_LEARNING_RATE = 0.01                        
PATIENCE = 1000
NUM_NODES = 512

# LOSS
                                       
MONTE_CARLO_SAMPLES = 500
EPSILON = .01  
TAU = EPSILON
CONSTANT = 2.5     
K = 21

FILM = True

####################
# Main #############
####################


v_k = Network(2, [NUM_NODES])
v_k.to(device)

v_kplus1 = Network(2, [NUM_NODES])
v_kplus1.to(device)
points = Tensor(normalize(g_quadrath))
draw_point_cloud(points)
pointcloud = Variable( points , requires_grad=True).to(device)

for k in range(K):
    print("Step ",k)

    v_kplus1.load_state_dict(v_k.state_dict())
    
    v_k.eval() # Fix NN
    v_kplus1.train() # Train v_k+1

    optimizer = optim.Adam(v_kplus1.parameters(), START_LEARNING_RATE )
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=PATIENCE, verbose=False)

    for i in range(NUM_TRAINING_SESSIONS+1):
        # training the network

        loss =  AT_loss(v_kplus1, pointcloud, EPSILON, MONTE_CARLO_SAMPLES, 0, CONSTANT ) + TAU * sobolev(v_k,v_kplus1, MONTE_CARLO_SAMPLES, 0.0)
        if (i%10==0):
            report_progress(i, NUM_TRAINING_SESSIONS , loss.detach().numpy() )

        # backpropagation
            
        v_kplus1.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

    v_k.load_state_dict(v_kplus1.state_dict())


    draw_phase_field_paper(v_k, .5, .5, k, film =FILM)


