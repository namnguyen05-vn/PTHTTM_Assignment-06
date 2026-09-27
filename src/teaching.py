"""Tiny NumPy examples for recurrence and backpropagation through time."""
import numpy as np

def rnn_forward(x, wx, wh, bias):
    h=np.zeros(wh.shape[0],dtype="float64")
    states=[h.copy()]
    for xt in x:
        h=np.tanh(xt@wx+h@wh+bias)
        states.append(h.copy())
    return np.asarray(states)

def last_state_gradient(x,wx,wh,bias,target):
    states=rnn_forward(x,wx,wh,bias)
    loss=0.5*np.sum((states[-1]-target)**2)
    dh=states[-1]-target
    dwx=np.zeros_like(wx);dwh=np.zeros_like(wh);db=np.zeros_like(bias)
    for t in range(len(x)-1,-1,-1):
        dz=dh*(1-states[t+1]**2)
        dwx+=np.outer(x[t],dz)
        dwh+=np.outer(states[t],dz)
        db+=dz
        dh=dz@wh.T
    return loss,(dwx,dwh,db)

def finite_difference(function, array, epsilon=1e-6):
    gradient=np.zeros_like(array)
    for index in np.ndindex(array.shape):
        plus=array.copy();minus=array.copy()
        plus[index]+=epsilon;minus[index]-=epsilon
        gradient[index]=(function(plus)-function(minus))/(2*epsilon)
    return gradient
