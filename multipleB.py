# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

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

# figure resolution parameters
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

def projection(arr, lower_bound, upper_bound):
    return [min(max(x, lower_bound), upper_bound) for x in arr]

def J(x, *args):
    Q, r = args
    return 0.5* x.T @ Q @ x + r.T @ x

class Agent:
    def __init__(self, numStates,Q,r,ni):
        self.x = np.random.rand(numStates)
        self.Q = Q
        self.r = r

        self.grad = (self.Q @ self.x + self.r.T).T
        
        self.lastUpdate = 0 # store last update time
        self.lastComms = 0 # store last communication time
        self.lastSample = 0 # store last sample time
        
    def sampleProblem(self,Qnew,rNew):
        self.Q = Qnew
        self.r = rNew
        
    def resetLastComms(self):
        self.lastComms = 0
        
    def resetLastUpdate(self):
        self.lastUpdate = 0


    
#%% Solve for continuous time QP solution

## Problem Parameters
N = 10 # number of agents
ni = 2 # agent partition size
n = N*ni # size of Q matrix

t0 = 0 # initial time
tf = 50 # final time
h = 0.01 # Continuous time sampling rate
w = 0.1 # frequency

UB = 100
LB = -UB

ub = UB * np.ones(n) # decision variable upper bound
lb = LB * np.ones(n) # decision variable lower bound
x0 = np.zeros(n)

# Make PD Q matrix to ensure strong convexity
Q = 100*make_spd_matrix(n_dim=n, random_state=42)
Q = 0.5 * (Q + Q.T) # make Q symmetric for fun


# Create empty matrices to store continuous problem
t = np.linspace(t0, tf, int((tf - t0)/h)  )
Qcont = np.empty((n, n, len(t)))
rCont = np.empty((n, 1, len(t)))


# Fill Q(t) and r(t) and solve QP
for i in range(len(t)):
    Qcont[:,:,i] = Q + np.eye(n)*np.cos(w*t[i]) 
    rCont[:,:,i] = 100*np.ones((n,1))*np.sin(2*w*t[i]) 
    
    # Check if resulting Q(t) is strongly convex
    eigs, v = eig(Qcont[:,:,i])
    allPos = all(element > 0 for element in eigs) 
    if not allPos:
        print(i)
        raise Exception("Q is not Positive Definite")
    



#%% Solve for discrete time QP solution
ts = h*200 # Discrete time sampling rate
td = np.linspace(t0, tf, int((tf - t0)/ts) + 1 )

# Create empty matrices to store continuous problem
Qdis = np.empty((n, n, len(td)))
rDis = np.empty((n, 1, len(td)))

# declaring magnitude of repetition
K = int(ts/h)

# Fill Q(t) and r(t) and solve QP
for i in range(len(td)):
    Qdis[:,:,i] = Q + np.eye(n)*np.cos(w*td[i]) 
    rDis[:,:,i] = 100*np.ones((n,1))*np.sin(2*w*td[i]) 
    


#%% Implement asynchronous algorithm on sampled problem

Qagg = np.empty((n, n, len(td)))
rAgg = np.empty((n, 1, len(td)))
Qagg[:,:,0] = Qdis[:,:,0]
rAgg[:,:,0] = rDis[:,:,0]


gamma = 9e-4 # step size
iters = 500 # Number of iterations completed at each time step
prob = 0.0 # Probability of completing operations
probSample = 1.0
BB = [1,3,10,100] # update, and communication max delay

    
ss = np.zeros((n,len(td)*iters))


# Create a new model
m = gp.Model("Aggregate")
m.setParam('OutputFlag', 0)
x = m.addMVar(shape=n, name="x")

x.ub = UB
x.lb = LB
xInit = np.random.rand(n)


Bdict = {}

for B in BB:
    
    # Create instances of the Agent classes
    Agents = []
    for i in range(N):
        Agents.append(Agent(n,Qdis[:,:,0],rDis[:,:,0],ni))
        Agents[i].x = xInit.copy()
        
    # Define true state
    xTrue = np.empty((n,))
    for k in range(N):
        ind1 = k*ni
        ind2 = k*ni + ni
        xTrue[ind1:ind2] = Agents[k].x[ind1:ind2]
        
    # print(xTrue)
    
    xStarAgg = []
    fStarAgg = []
    fAsynch = []
    xHist = []

    ## Discrete time loop
    for tt in range(len(td)-1):
        
        ## SAMPLE OBJECTIVE FUNCTION
        sampleReality = np.random.uniform(0,1,N) # generate list to determine if agents communicate
        sample = list(filter(lambda x: sampleReality[x] < probSample, range(len(sampleReality))))
        # print('SAMPLE TIME:', tt)
        # sample = [1]
        
        if tt != 0:
            Qagg[:,:,tt] = Qagg[:,:,tt - 1].copy()
            rAgg[:,:,tt] = rAgg[:,:,tt-1].copy()
        
        for k in sample:
    
            Agents[k].sampleProblem(Qdis[:,:,tt].copy(), rDis[:,:,tt].copy())
            
            # Update Aggregate Problem
            if tt != 0:
                ind1 = k*ni
                ind2 = k*ni + ni
                Qagg[ind1:ind2,:,tt] = Qdis[ind1:ind2,:,tt].copy()
                rAgg[ind1:ind2,:,tt] = rDis[ind1:ind2,:,tt].copy()
        
        obj = 0.5* x.T @ Qagg[:,:,tt] @ x + rAgg[:,:,tt].T @ x
        m.setObjective(obj, GRB.MINIMIZE)
    
        m.optimize()
        xStarAgg.append(x.X)
        fStarAgg.append(0.5* xStarAgg[tt].T @ Qagg[:,:,tt] @ xStarAgg[tt] + rAgg[:,:,tt].T @ xStarAgg[tt])
        
        
        ## Asynchronous algorithm loop
        for j in range(iters):
            
            ## UPDATE TRUE STATE AND CALCULATE COST FUNCTION
            for k in range(N):
                ind1 = k*ni
                ind2 = k*ni + ni
                xTrue[ind1:ind2] = Agents[k].x[ind1:ind2]
                
            fAsynch.append(0.5* xTrue.T @ Qagg[:,:,tt] @ xTrue + rAgg[:,:,tt].T @ xTrue)
                
            # Store xTrue
            xHist.append(xTrue.copy())
            
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
                
                Agents[k].grad = (Agents[k].Q @ Agents[k].x) + Agents[k].r.reshape(n,)
                step = Agents[k].grad[ind1:ind2].copy()
                xNew = projection(Agents[k].x[ind1:ind2].copy() - gamma*step,LB,UB)
                ss[ind1:ind2,tt*iters +j] = xNew - Agents[k].x[ind1:ind2] # Store s(k)
                
                Agents[k].x[ind1:ind2] = xNew.copy()
                
                
    

    
        # Reset last update and communication time
        for j in range(N):
            Agents[j].resetLastComms()
            Agents[j].resetLastUpdate()
            

    # Plotting
    fStarAggPlot = []
    
    for ii in range(len(td)-1):
        for ele in range(iters):
            fStarAggPlot.append(fStarAgg[ii])

    fStarAggPlotArr = np.array([fStarAggPlot]).reshape((len(td)-1)*iters,)
    fAsynchArr = np.array([fAsynch]).reshape((len(td)-1)*iters,)

    
    # Calculate alpha & beta
    alpha = fAsynchArr - fStarAggPlotArr
    
    beta = np.zeros((len(td)-1)*iters)
    
    for ii in range(len(beta)):
        sum1 = 0
        for jj in range(ii-B,ii-1):
            if jj >= 0:
                sum1 += LA.norm( ss[:,jj] )**2
                
        beta[ii] = sum1
        
        
    Bdict[B] = {"alpha": alpha, "beta": beta}

#%%
new_tick_locations  = np.arange(start=0, stop=iters*len(td), step=iters)
iterations = range(iters*(len(td)-1))

myColors = ['r','b','g','m']
ctr = 0

fig = plt.figure()
ax9 = fig.add_subplot(111)
ax10 = ax9.twiny()
for key in Bdict.keys():
    lbl = str(key)
    ax9.semilogy(iterations,Bdict[key]['alpha'], color=myColors[ctr], label = lbl)
    ctr += 1
# ax9.set_ylim([1e-7, 1e6])
ax9.set_xlabel(r"Iterations $(k)$", fontsize =14)
ax9.set_ylabel(r"$\alpha(k;t)$", fontsize =14)
ax9.legend(title=r'Maximum Delay Bound $(B)$', loc = 'lower right', ncol=2, fontsize =11)
ax9.grid()
ax10.set_xlim(ax9.get_xlim())
ax10.set_xticks(new_tick_locations)
ax10.set_xticklabels(2*(new_tick_locations/500).astype(int), fontsize = 8)
ax10.set_xlabel(r"Time $(s)$", fontsize =14)
# plt.savefig('C:/Users/gbehrendt3/OneDrive - Georgia Institute of Technology/UFstuff/Research/myPapers/Paper5/Plots/multipleB.eps', format = 'eps', bbox_inches='tight')
plt.show()



















































