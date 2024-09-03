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

# def fun(Q,r,w,x,y):
#     return 0.5* x**2 * (Q + np.cos(w*y)) + 100*np.sin(2*w*y)*x
    
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

xStarContList = []
fStarContList = []

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
    

    xStarCont = solve_qp(Qcont[:,:,i], np.array(rCont[:,:,i]).reshape(n,), lb=lb, ub=ub, initvals=x0, solver="quadprog")
    fStarCont = 0.5* xStarCont.T @ Qcont[:,:,i] @ xStarCont + rCont[:,:,i].T @ xStarCont
    
    xStarContList.append(xStarCont)
    fStarContList.append(fStarCont)
    
xStarContArr = np.array([xStarContList]).reshape(len(t),n)
fStarContArr = np.array([fStarContList]).reshape(len(t),)


#%% Solve for discrete time QP solution
ts = h*200 # Discrete time sampling rate
td = np.linspace(t0, tf, int((tf - t0)/ts) + 1 )

# Create empty matrices to store continuous problem
Qdis = np.empty((n, n, len(td)))
rDis = np.empty((n, 1, len(td)))

# declaring magnitude of repetition
K = int(ts/h)
xStarDisList = []
fStarDisList = []
xStarDisHist = []

# Fill Q(t) and r(t) and solve QP
for i in range(len(td)):
    Qdis[:,:,i] = Q + np.eye(n)*np.cos(w*td[i]) 
    rDis[:,:,i] = 100*np.ones((n,1))*np.sin(2*w*td[i]) 
    
    # Check if resulting Q(t) is strongly convex
    eigs, v = eig(Qdis[:,:,i])
    allPos = all(element > 0 for element in eigs) 
    if not allPos:
        print(i)
        raise Exception("Q is not Positive Definite")
    

    xStarDis = solve_qp(Qdis[:,:,i], np.array(rDis[:,:,i]).reshape(n,), lb=lb, ub=ub, initvals=x0, solver="quadprog")
    fStarDis = 0.5* xStarDis.T @ Qdis[:,:,i] @ xStarDis + rDis[:,:,i].T @ xStarDis
    
    xStarDisHist.append(xStarDis)
    
    if i == len(td) - 1:
        i
        # xStarDisList.append(xStarDis)
        # fStarDisList.append(fStarDis)
    else:
        for ele in range(K):
            xStarDisList.append(xStarDis)
            fStarDisList.append(fStarDis)

xStarDisArr = np.array([xStarDisList]).reshape(len(t),n)
fStarDisArr = np.array([fStarDisList]).reshape(len(t),)

#%%
if n == 1:
    # Creating dataset for surface plot
    x = np.outer(np.linspace(-0.6, 0.6, 100), np.ones(100))
    y = np.outer(np.linspace(0, tf, 100), np.ones(100)).T # transpose
    z = 0.5* x**2 * (Q + np.cos(w*y)) + 100*np.sin(2*w*y)*x
    
    # Creating color map
    my_cmap = plt.get_cmap('cool')
    
    # Creating figure
    fig1 = plt.figure(figsize =(14, 9))
    ax1 = plt.axes(projection ='3d')
    # Creating plot
    surf = ax1.plot_surface(x, y, z,
                           cmap = my_cmap,
                           edgecolor ='none',
                           alpha=0.4,
                           label=r'$J(x,t)$')
    ax1.plot3D(xStarContArr.reshape(len(t),),t, fStarContArr, 'red', label = r'$x^*(t)$')
    ax1.plot3D(xStarDisArr.reshape(len(t),),t, fStarDisArr, 'blue', label = r'$x^*_f(\ell t_s)$')
    fig1.colorbar(surf, ax = ax1,
                 shrink = 0.5, aspect = 5)
     
    
    ax1.set_xlabel('x',fontsize=16)
    ax1.set_ylabel('Time $(s)$',fontsize=16)
    ax1.set_zlabel(r'$J(x,t)$',fontsize=16)
    ax1.legend(fontsize=14)
     
    # show plot
    plt.show()

xStarC2D = LA.norm(xStarContArr - xStarDisArr, axis = 1)
 
# fig2, ax2 = plt.subplots()
# ax2.plot(t,xStarC2D,color='r', label=r'$x^*(t) - x_{f}^*(t_\ell)$')
# ax2.set_xlabel("Time $(s)$", fontsize =14)
# ax2.set_ylabel(r'$ \| x*(t) - x*(t_\ell) \|$', fontsize =14)
# ax2.grid()
# plt.show()

#%% Gurobi

# Create a new model
# m = gp.Model("Aggregate")
# x = m.addMVar(shape=n, name="x")

# obj = 0.5* x.T @ Qdis[:,:,0] @ x + rDis[:,:,0].T @ x
# m.setObjective(obj, GRB.MINIMIZE)

# x.ub = UB
# x.lb = LB

# m.optimize()

# for v in m.getVars():
#     print(f"{v.VarName} {v.X:g}")

# print(f"Obj: {m.ObjVal:g}")


#%% Implement asynchronous algorithm on sampled problem

g = np.empty((len(td),))
Qagg = np.empty((n, n, len(td)))
rAgg = np.empty((n, 1, len(td)))
Qagg[:,:,0] = Qdis[:,:,0]
rAgg[:,:,0] = rDis[:,:,0]

# Create instances of the Agent classes
Agents = []
for i in range(N):
    Agents.append(Agent(n,Qdis[:,:,0],rDis[:,:,0],ni))


gamma = 1e-3 # step size
B = 10 # update, and communication max delay
iters = 500 # Number of iterations completed at each time step
prob = 0.6 # Probability of completing operations
probSample = 0.5


# Define true state
xTrue = np.empty((n,))
for k in range(N):
    ind1 = k*ni
    ind2 = k*ni + ni
    xTrue[ind1:ind2] = Agents[k].x[ind1:ind2]
    
# g[0] = 0.5* xTrue.T @ Qagg @ xTrue  + rAgg.T @ xTrue

xHist = []
ss = np.zeros((n,len(td)*iters))
# xHist.append(xTrue.copy())


# Create a new model
m = gp.Model("Aggregate")
m.setParam('OutputFlag', 0)
x = m.addMVar(shape=n, name="x")

x.ub = UB
x.lb = LB

xStarAgg = []
fStarAgg = []
fAsynch = []


## Discrete time loop
for tt in range(len(td)-1):
    
    ## SAMPLE OBJECTIVE FUNCTION
    sampleReality = np.random.uniform(0,1,N) # generate list to determine if agents communicate
    sample = list(filter(lambda x: sampleReality[x] < probSample, range(len(sampleReality))))
    print('SAMPLE TIME:', tt)
    
    if tt != 0:
        Qagg[:,:,tt] = Qagg[:,:,tt - 1].copy()
        rAgg[:,:,tt] = rAgg[:,:,tt-1].copy()
    
    for k in sample:
        
        print('Agent ', k, 'sampling...')
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
            ss[ind1:ind2,tt*iters+j] = xNew - Agents[k].x[ind1:ind2] # Store s(k)
            Agents[k].x[ind1:ind2] = xNew.copy()
            
            

        ## UPDATE TRUE STATE AND CALCULATE COST FUNCTION
        for k in range(N):
            ind1 = k*ni
            ind2 = k*ni + ni
            xTrue[ind1:ind2] = Agents[k].x[ind1:ind2]
            
        fAsynch.append(0.5* xTrue.T @ Qagg[:,:,tt] @ xTrue + rAgg[:,:,tt].T @ xTrue)
            
        # Store xTrue
        xHist.append(xTrue.copy())

    # Reset last update and communication time
    for j in range(N):
        Agents[j].resetLastComms()
        Agents[j].resetLastUpdate()
            

# Plotting
xStarDisPlot = []
xStarAggPlot = []
fStarAggPlot = []
iterations = range(iters*(len(td)-1))

xStarAggSurf = np.empty((len(xStarAgg)*K,n))
fStarAggSurf = np.empty(len(fStarAgg)*K)

# count = 0
for jj in range(len(xStarAgg)):
    for ele in range(K):
        xStarAggSurf[K*jj + ele,:] = xStarAgg[jj]
        fStarAggSurf[K*jj + ele] = fStarAgg[jj]
    # count += 1
            
for ii in range(len(xStarDisHist)-1):
    for ele in range(iters):
        xStarDisPlot.append(xStarDisHist[ii])
        xStarAggPlot.append(xStarAgg[ii])
        fStarAggPlot.append(fStarAgg[ii])

xStarDisPlotArr = np.array([xStarDisPlot]).reshape((len(td)-1)*iters,n)
xStarAggPlotArr = np.array([xStarAggPlot]).reshape((len(td)-1)*iters,n)
fStarAggPlotArr = np.array([fStarAggPlot]).reshape((len(td)-1)*iters,)
fAsynchArr = np.array([fAsynch]).reshape((len(td)-1)*iters,)
xHistArr = np.array([xHist]).reshape((len(td)-1)*iters,n)

xErrorD2S = LA.norm(xStarDisPlotArr - xHistArr, axis = 1)
xErrorA2S = LA.norm(xStarAggPlotArr - xHistArr, axis = 1)


# Calculate alpha & beta
alpha = fAsynchArr - fStarAggPlotArr

beta = np.zeros((len(td)-1)*iters)

for ii in range(len(beta)):
    sum1 = 0
    for jj in range(ii-B,ii-1):
        if jj >= 0:
            sum1 += LA.norm( ss[:,jj] )**2
            
    beta[ii] = sum1

#%%
new_tick_locations  = np.arange(start=0, stop=iters*len(td), step=iters)


fig = plt.figure()
ax3 = fig.add_subplot(111)
ax4 = ax3.twiny()
ax3.plot(iterations,xErrorD2S,color='darkviolet', label=r'$\| x_{f}^*(t) - x(k) \|$')
ax3.plot(iterations,xErrorA2S,color='deeppink', label=r'$\| x_k^*(t_z) - x(k) \|$')
# ax3.set_ylim([1e-7, 1e6])
ax3.set_xlabel("Iterations $(k)$", fontsize =14)
ax3.legend(loc = 'upper left', ncol=1, fontsize =11)
ax3.grid()
ax4.set_xlim(ax3.get_xlim())
ax4.set_xticks(new_tick_locations)
ax4.set_xticklabels(2*(new_tick_locations/500).astype(int), fontsize = 8)
ax4.set_xlabel(r"Time $(s)$", fontsize =14)
# plt.savefig('C:/Users/gbehrendt3/OneDrive - Georgia Institute of Technology/UFstuff/Research/myPapers/Paper5/Plots/trackingError.svg', format = 'svg', bbox_inches='tight')
plt.show()

# fig = plt.figure()
# ax5 = fig.add_subplot(111)
# ax6 = ax5.twiny()
# ax5.plot(iterations,xErrorA2S,color='b', label=r'$\| x^*(t_z) - x(k) \|$')
# # ax5.set_ylim([1e-7, 1e6])
# ax5.set_xlabel("Iterations $(k)$", fontsize =14)
# ax5.legend(loc = 'upper left', ncol=2, fontsize =11)
# ax5.grid()
# ax6.set_xlim(ax5.get_xlim())
# ax6.set_xticks(new_tick_locations)
# ax6.set_xticklabels((new_tick_locations/500).astype(int), fontsize = 8)
# ax6.set_xlabel(r"Time Index $(t_\ell)$", fontsize =14)
# #plt.savefig('PythonPlotsNoTitle2/plot4.eps', format = 'eps', bbox_inches='tight')
# plt.show()

# fig = plt.figure()
# ax9 = fig.add_subplot(111)
# ax10 = ax9.twiny()
# ax9.plot(iterations,xErrorD2S,color='r', label=r'$\| x_{f}^*(t_\ell) - x(k) \|$')
# # ax3.set_ylim([1e-7, 1e6])
# ax9.set_xlabel("Iterations $(k)$", fontsize =14)
# ax9.legend(loc = 'upper left', ncol=1, fontsize =11)
# ax9.grid()
# ax10.set_xlim(ax9.get_xlim())
# ax10.set_xticks(new_tick_locations)
# ax10.set_xticklabels((new_tick_locations/500).astype(int), fontsize = 8)
# ax10.set_xlabel(r"Time Index $(t_\ell)$", fontsize =14)
# #plt.savefig('PythonPlotsNoTitle2/plot4.eps', format = 'eps', bbox_inches='tight')
# plt.show()



fig = plt.figure()
ax7 = fig.add_subplot(111)
ax8 = ax7.twiny()
ax7.semilogy(iterations,alpha,color='r', label=r'$ \alpha(k;t)$')
ax7.semilogy(iterations,beta,color='b', label=r'$ \beta (k)$')
ax7.set_ylim([1e-9, 1e6])
ax7.set_xlabel("Iterations $(k)$", fontsize =14)
ax7.legend(loc = 'lower right', ncol=1, fontsize =14)
ax7.grid()
ax8.set_xlim(ax7.get_xlim())
ax8.set_xticks(new_tick_locations)
ax8.set_xticklabels(2*(new_tick_locations/500).astype(int), fontsize = 8)
ax8.set_xlabel(r"Time $(s)$", fontsize =14)
# plt.savefig('C:/Users/gbehrendt3/OneDrive - Georgia Institute of Technology/UFstuff/Research/myPapers/Paper5/Plots/alphaBeta.svg', format = 'svg', bbox_inches='tight')
plt.show()




#%%
if n == 1:
    Qagg[:,:,-1] = Qagg[:,:,-2].copy()
    rAgg[:,:,-1] = rAgg[:,:,-2].copy()
    # Creating dataset for surface plot
    x = np.linspace(-0.6, 0.6, 100)
    y = np.linspace(0, tf, len(td))
    X,Y = np.meshgrid(x,y)
    zs = np.empty(len(x)*len(y))
    ctr = 0
    for j in range(len(y)):
        for i in range(len(x)):
            # zs[ctr*len(x) + i] = 0.5* x[i]**2 * (Q + np.cos(w*y[j])) + 100*np.sin(2*w*y[j])*x[i]
            zs[ctr*len(x) + i] = 0.5* x[i]**2 * Qagg[:,:,j]  + rAgg[:,:,j]*x[i]
        ctr += 1
        
    Z = zs.reshape(X.shape)
    
    
    # Creating color map
    my_cmap = plt.get_cmap('cool')
    
    # Creating figure
    fig50 = plt.figure(figsize =(14, 9))
    ax50 = plt.axes(projection ='3d')
    # Creating plot
    surf = ax50.plot_surface(X, Y, Z,
                           cmap = my_cmap,
                           edgecolor ='none',
                           alpha=0.4,
                           label=r'$g(x,t_z)$')
    ax50.plot3D(xStarAggSurf.reshape(len(t),),t, fStarAggSurf, 'red', label = r'$x*(t_z)$')
    # ax50.plot3D(xStarDisArr.reshape(len(t),),t, fStarDisArr, 'blue', label = r'$x*(t_\ell)$')
    fig50.colorbar(surf, ax = ax50,
                  shrink = 0.5, aspect = 5)
     
    
    ax50.set_xlabel('x',fontsize=16)
    ax50.set_ylabel('Time (s)',fontsize=16)
    ax50.set_zlabel(r'$g(x,t_z)$',fontsize=16)
    # ax50.set_title('Sampled Problem',fontsize=24)
    ax50.legend(fontsize=14)
     
    # show plot
    plt.show()

























































