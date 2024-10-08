
from sklearn.datasets import make_spd_matrix
import numpy as np
from numpy.linalg import eig
from scipy.optimize import minimize
from scipy.optimize import Bounds
import quadprog 
import matplotlib.pyplot as plt
from qpsolvers import solve_qp
from numpy import linalg as LA
import gurobipy as gp 
from gurobipy import GRB
from mpl_toolkits.mplot3d import axes3d
import random
import control as ct

# figure resolution parameters
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

def projection(arr, lower_bound, upper_bound):
    return [min(max(x, lower_bound), upper_bound) for x in arr]
  

class Agent:
    def __init__(self, numStates,Q,y,ni):
        # self.x = 50*np.random.rand(numStates)
        self.x = [random.randint(-30, 30) for _ in range(numStates)]
        self.Q = Q
        self.yRef = y
        
        self.grad = (self.Q @ ( self.x - self.yRef) ).T #### UPDATE GRADIENT ####
        
        self.lastUpdate = 0 # store last update time
        self.lastComms = 0 # store last communication time
        self.lastSample = 0 # store last sample time
        
    def sampleProblem(self,Qnew,y):
        self.Q = Qnew
        self.yRef = y
        
    def resetLastComms(self):
        self.lastComms = 0
        
    def resetLastUpdate(self):
        self.lastUpdate = 0


## Problem Parameters
N = 5 # number of agents
ni = 2 # agent partition size
n = N*ni # size of Q matrix
numPairs = int(N*(N-1)/2)

t0 = 0 # initial time
tf = 500 # final time
h = 0.01 # Continuous time sampling rate
t = np.linspace(t0, tf, num = int( (tf-t0)/h ) + 1)
w = 0.01 # Hz

# Create Continuous Time varying target trajectory
y = np.empty((ni,len(t)))
for i in range(len(t)):
    y[:,i] = 100 * np.array([[np.cos(w*t[i])],
                        [np.sin(3*w*t[i])]]).reshape(ni,)
    
# Create Discrete Time varying target trajectory
ts = 1
td = np.linspace(t0, tf, num = int( (tf-t0)/ts ) + 1)

yd = np.empty((ni,len(td)))
for i in range(len(td)):
    yd[:,i] = 100 * np.array([[np.cos(w*td[i])],
                        [np.sin(3*w*td[i])]]).reshape(ni,)

# Target Trajectory Plot
# fig1, ax1 = plt.subplots()
# ax1.plot(y[0,:],y[1,:], label = "Continuous")
# ax1.plot(yd[0,:],yd[1,:], label = "Discrete")
# ax1.set_xlabel(r"$x \ (m)$")
# ax1.set_ylabel(r"$y \ (m)$")
# ax1.legend()
# plt.show()


UB = 200
LB = -UB

ub = UB * np.ones(n) # decision variable upper bound
lb = LB * np.ones(n) # decision variable lower bound
x0 = np.zeros(n)


#%% Asynchronous Algorithm

# Optimization Parameters
Q = 10 * np.identity(n)
gamma = 3e-3 # step size
B = 5 # update, and communication max delay
iters = 10 # Number of iterations completed at each time step
prob = 0.5 # Probability of completing operations
probSample = 0.1


xHist = []

# Create a new model
m = gp.Model("Aggregate")
m.setParam('OutputFlag', 0)
x = m.addMVar(shape=n, name="x")

x.ub = UB
x.lb = LB

xStarAgg = []
fStarAgg = []
fAsynch = []

yAgg = np.empty((n,len(td)))
yDis = np.vstack([yd]*N)

yAgg[:,0] = yDis[:,0].copy()

# Create instances of the Agent classes
x0 = []
Agents = []
for i in range(N):
    Agents.append(Agent(n,Q,yDis[:,0],ni))
    x0.append(Agents[i].x[i*ni:i*ni+ni])

# Define true state
xTrue = np.empty((n,))
for k in range(N):
    ind1 = k*ni
    ind2 = k*ni + ni
    xTrue[ind1:ind2] = Agents[k].x[ind1:ind2]

xHist.append(xTrue.copy())

## Discrete time loop
for tt in range(len(td)-1):
    
    ## SAMPLE OBJECTIVE FUNCTION
    sampleReality = np.random.uniform(0,1,N) # generate list to determine if agents communicate
    sample = list(filter(lambda x: sampleReality[x] < probSample, range(len(sampleReality))))
    print('SAMPLE TIME:', tt)
    
    if tt != 0:
        yAgg[:,tt] = yAgg[:,tt-1].copy()
    
    for k in sample:
        
        print('Agent ', k, 'sampling...')
        Agents[k].sampleProblem(Q.copy(), yDis[:,tt].copy())
                

        # Update Aggregate Problem
        if tt != 0:
            ind1 = k*ni
            ind2 = k*ni + ni
            yAgg[ind1:ind2,tt] = yDis[ind1:ind2,tt].copy()
    

    obj = 0.5* np.transpose( x -  yAgg[:,tt] ) @ Q @ ( x - yAgg[:,tt] ) 
    m.setObjective(obj, GRB.MINIMIZE)

    m.optimize()
    xStarAgg.append(x.X)
    fStarAgg.append(0.5* (xStarAgg[tt] -  yAgg[:,tt] ).T @ Q @ (xStarAgg[tt] -  yAgg[:,tt] ) )

    ## Asynchronous algorithm loop
    for j in range(iters):
        
        ## COMMUNICATE
        comReality = np.random.uniform(0,1,N) # generate list to determine if agents communicate
        communicate = list(filter(lambda x: comReality[x] < prob, range(len(comReality))))
        
        for ii in range(N):
            if j - Agents[ii].lastComms > B:
                if ii not in communicate:
                    communicate.append(ii)
        
        for k in communicate:
            Agents[k].lastComms = j
            ind1 = k*ni
            ind2 = k*ni + ni
            for i in range(N):
                Agents[i].x[ind1:ind2] = Agents[k].x[ind1:ind2].copy()
                

        ## UPDATE
        updateReality = np.random.uniform(0,1,N)
        update = list(filter(lambda x: updateReality[x] < prob, range(len(updateReality))))
        
        for ii in range(N):
            if j - Agents[ii].lastUpdate > B:
                if ii not in update:
                    update.append(ii)

        for k in update:
            Agents[k].lastUpdate = j
            ind1 = k*ni
            ind2 = k*ni + ni
            
            Agents[k].grad = (Agents[k].Q @ ( Agents[k].x - Agents[k].yRef) ) 
            step = Agents[k].grad[ind1:ind2].copy()
            xNew = projection(Agents[k].x[ind1:ind2].copy() - gamma*step,LB,UB)
            # ss[ind1:ind2,tt*iters+j] = xNew - Agents[k].x[ind1:ind2] # Store s(k)
            Agents[k].x[ind1:ind2] = xNew.copy()
            
            

        ## UPDATE TRUE STATE AND CALCULATE COST FUNCTION
        for k in range(N):
            ind1 = k*ni
            ind2 = k*ni + ni
            xTrue[ind1:ind2] = Agents[k].x[ind1:ind2]
            
        fAsynch.append(0.5* (xTrue - yAgg[:,tt] ).T @ Q @ (xTrue - yAgg[:,tt]) )
            
        # Store xTrue
        xHist.append(xTrue.copy())

    # Reset last update and communication time
    for j in range(N):
        Agents[j].resetLastComms()
        Agents[j].resetLastUpdate()


# xStarAggPlot = []

# Plotting
fStarAggPlot = []

for ii in range(len(td)-1):
    for ele in range(iters):
        fStarAggPlot.append(fStarAgg[ii])

fStarAggPlotArr = np.array([fStarAggPlot]).reshape((len(td)-1)*iters,)
fAsynchArr = np.array([fAsynch]).reshape((len(td)-1)*iters,)

alpha = fAsynchArr - fStarAggPlotArr



# Agent positions
States = np.empty((n,len(xHist)))

for j in range(States.shape[1]):
    States[:,j] = xHist[j]
#%%
yDisRepeat = [] 
yAggRepeat = []
for i in range(len(td)-1):
    for k in range(iters):
        yDisRepeat.append(yd[:,i])
        yAggRepeat.append(yAgg[:,i])

yDisRepeatArr = np.array(yDisRepeat).T
yAggRepeatArr = np.array(yAggRepeat).T

networkErrorDis = np.empty((iters*(len(td) - 1)))
networkErrorAgg = np.empty((iters*(len(td) - 1)))
agentErrorDis = np.empty((N,iters*(len(td) - 1)))
agentErrorAgg = np.empty((N,iters*(len(td) - 1)))
for i in range(iters*(len(td) - 1)):
    for k in range(N):
        agentErrorDis[k,i] = LA.norm(States[k*ni:k*ni+ni,i] - yDisRepeatArr[:,i])
        agentErrorAgg[k,i] = LA.norm(States[k*ni:k*ni+ni,i] - yAggRepeatArr[k*ni:k*ni+ni,i])
    networkErrorDis[i] = sum(agentErrorDis[:,i])
    networkErrorAgg[i] = sum(agentErrorAgg[:,i])

#%%
iterations = range(iters*(len(td)-1))
new_tick_locations  = np.arange(start=0, stop=iters*len(td), step=iters*50)

# fig2,ax2 = plt.subplots()
# ax2.semilogy(iterations,alpha,color='r', label=r'$\alpha$')
# # ax3.set_ylim([1e-7, 1e6])
# ax2.set_xlabel("Iterations $(k)$", fontsize =14)
# ax2.legend(loc = 'upper right', fontsize =11)
# ax2.grid()
# #plt.savefig('PythonPlotsNoTitle2/plot4.eps', format = 'eps', bbox_inches='tight')
# plt.show()

myColors = ['#B12CE6','#46E52F','#0AE5DF','#E56915','#1912E5','#E10901','#E600D6','#6CA9EB']

fig3,ax3 = plt.subplots()
ax3.plot(yd[0,:],yd[1,:],color='k', linestyle='dashed', label = "Target")
for i in range(N):
    ax3.plot(States[i*ni,:],States[i*ni + 1,:],color = myColors[i])
# ax3.set_ylim([1e-7, 1e6])
ax3.set_xlabel(r"$x \ (m)$", fontsize =14)
ax3.set_ylabel(r"$y \ (m)$", fontsize =14)
ax3.legend(["Reference", "Agents"])
ax3.grid()
# plt.savefig('C:/Users/gbehrendt3/OneDrive - Georgia Institute of Technology/UFstuff/Research/myPapers/Paper5/Plots/trajectories.eps', format = 'eps', bbox_inches='tight')
plt.show()

fig = plt.figure()
ax4 = fig.add_subplot(111)
ax5 = ax4.twiny()
ax4.plot(iterations,networkErrorDis,color = 'r', label = r"$\sum^N_{i=1} \| z_i(k) - r_d(t) \| $")
ax4.plot(iterations,networkErrorAgg,color = 'b', label = r"$\sum^N_{i=1} \| z_i(k) - r_{asynch} (t) \| $")
# ax5.set_ylim([1e-7, 1e6])
ax4.set_xlabel("Iterations $(k)$", fontsize =14)
ax4.set_ylabel("Position Error $(m)$", fontsize =14)
ax4.legend(loc = 'upper right', ncol=2, fontsize =11)
ax4.grid()
ax5.set_xlim(ax4.get_xlim())
ax5.set_xticks(new_tick_locations)
ax5.set_xticklabels((new_tick_locations/10).astype(int), fontsize = 8)
ax5.set_xlabel(r"Time $(s)$", fontsize =14)
# plt.savefig('C:/Users/gbehrendt3/OneDrive - Georgia Institute of Technology/UFstuff/Research/myPapers/Paper5/Plots/trajectoryError.eps', format = 'eps', bbox_inches='tight')
plt.show()



# fig6,ax6 = plt.subplots()
# ax6.plot(iterations,networkErrorDis,color = 'r', label = r"$\sum^N_{i=1} \| z_i(k) - r_d(t) \| $")
# ax6.plot(iterations,networkErrorAgg,color = 'b', label = r"$\sum^N_{i=1} \| z_i(k) - r_{asynch} (t) \| $")
# ax6.set_xlabel(r"Iterations $(k)$", fontsize =14)
# ax6.grid()
# ax6.legend(loc = 'upper right', ncol=2, fontsize =10)
# # plt.savefig('C:/Users/behre/OneDrive - Georgia Institute of Technology/UFstuff/myPapers/Paper5/Plots/trajectoryError.eps', format = 'eps', bbox_inches='tight')
# plt.show()

# fig4,ax4 = plt.subplots()
# for i in range(N):
#     ax4.semilogy(iterations,agentErrorDis[i,:],color = myColors[i])
# ax4.set_ylabel("$\| x^i(k) - x_f^* \|$", fontsize =14)
# ax4.set_xlabel(r"Iterations $(k)$", fontsize =14)
# ax4.grid()
# #plt.savefig('PythonPlotsNoTitle2/plot4.eps', format = 'eps', bbox_inches='tight')
# plt.show()


# fig6,ax6 = plt.subplots()
# for i in range(N):
#     ax6.semilogy(iterations,agentErrorAgg[i,:],color = myColors[i])
# ax6.set_ylabel("$\| x^i(k) - x^* \|$", fontsize =14)
# ax6.set_xlabel(r"Iterations $(k)$", fontsize =14)
# ax6.grid()
# #plt.savefig('PythonPlotsNoTitle2/plot4.eps', format = 'eps', bbox_inches='tight')
# plt.show()

# fig = plt.figure()
# ax4 = fig.add_subplot(111)
# ax5 = ax4.twiny()
# for i in range(N):
#     ax4.plot(iterations,agentError[i,:],color = myColors[i])
# # ax5.set_ylim([1e-7, 1e6])
# ax4.set_xlabel("Iterations $(k)$", fontsize =14)
# ax4.legend(loc = 'upper left', ncol=2, fontsize =11)
# ax4.grid()
# ax5.set_xlim(ax4.get_xlim())
# ax5.set_xticks(new_tick_locations)
# ax5.set_xticklabels((new_tick_locations/1000).astype(int), fontsize = 8)
# ax5.set_xlabel(r"Time Index $(t_\ell)$", fontsize =14)
# #plt.savefig('PythonPlotsNoTitle2/plot4.eps', format = 'eps', bbox_inches='tight')
# plt.show()







