# QUANTUM_WALK_UNDER_MARKOVIAN_NOISE: Measure of Non-Markovianity in quantum walk evolution 
#                                   : BPL measure based on trace distance: ref: Phys.Rev.Lett,103,210401 (2009)
#
# Main variables: freq        - switching rate of the noise.
#               : amp         - Amplitude of the noise.
#               : t           - number of steps of quantum walk.                               
#               : sites       - number of lattice positions available to the walker (2*t+1).   
#               : coin_angle  - parameter of the SU(2) coin.
#               : qubit_state - input coin state
#               : z (1/2)     - plot with zeros/plot without zeros 
##########################################################################################################################

from qutip import *
import numpy as np
import matplotlib.pyplot as plt
from math import *
import cmath as cm
from scipy import linalg
from scipy import misc

#Basis states
ket0 = basis(2,0).unit() # |0>
ket1 = basis(2,1).unit() # |1>
psip = (basis(2,0)+basis(2,1)*1j).unit() # |0>+i|1>/sqrt(2)
psim = (basis(2,0)-basis(2,1)*1j).unit() # |0>-i|1>/sqrt(2)
psiplus = (basis(2,0)+basis(2,1)).unit()
psiminus = (basis(2,0)-basis(2,1)).unit()

#Coin transformation
def coin(coin_angle):
    C_hat = qutip.Qobj([[cos(radians(coin_angle)), sin(radians(coin_angle))],  #one paramter SU(2) matrix
                       [sin(radians(coin_angle)), -cos(radians(coin_angle))]])
    return C_hat

#Position transformation
def shift(sites):
    shift_l = qutip.Qobj(np.roll(np.eye(sites), 1, axis=0)) 
    shift_r = qutip.Qobj(np.roll(np.eye(sites), -1, axis=0))
    S_hat = tensor(ket0*ket0.dag(),shift_l) + tensor(ket1*ket1.dag(),shift_r) 
    return S_hat

#Walk operator: Evolution operator for DTQW
def walk(sites,coin_angle):
    C_hat = coin(coin_angle) 
    S_hat = shift(sites)     
    W_hat = S_hat*(tensor(C_hat,qeye(sites))) 
    return W_hat

#Kraus operators for the non-Markovian master equation under RTN
def telegraph_noise(t,sites,qstate,freq,amp):
    nu = t*freq; mu = cm.sqrt((2.0*amp*(1/freq))**2-1.0)      # Noise parameters
    gamma = exp(-nu)*(np.cos(nu*mu)+np.sin(nu*mu)/mu)      # Memory kernal
    K1 = tensor(sqrt((1+gamma)/2.0)*qeye(2), qeye(sites))  # Krauss operators
    K2 = tensor(sqrt((1-gamma)/2.0)*sigmaz(),qeye(sites))
    dstate = K1*qstate*K1.dag()+K2*qstate*K2.dag()
    return dstate

#Quantum walk generator: outputs the evolved wave function after 't' steps.
def qwalk_gen_nonmarkov(t,qubit_state,coin_angle,freq,amp):
    sites=2*t+1
    Position_state = basis(sites,t)           
    Psi = ket2dm(tensor(qubit_state,Position_state))                  # Initial state - \rho(0)  
    W_hat = walk(sites,coin_angle)                                    # Total Qwalk operator
    for i in range(1,t):                                              # Apply the walk operator 't' times
        Psi = W_hat*telegraph_noise(i,sites,Psi,freq,amp)*W_hat.dag() # krauss op 
    return Psi                                                        # returns decohered state

#Trace distance:
def trace_distance(rho1,rho2):
    diff = ket2dm(rho1) - ket2dm(rho2)
    Eval = diff.eigenenergies()
    return 0.5*sum(abs(Eval))

#Derevative of trace distance
def blp_measure(f):
    blp=[];amp=0.8
    def sigma(t):
        dist=0
        if t != 0:
            rho1 = qwalk_gen_nonmarkov(int(t),psiplus,45,f,amp).ptrace(0)
            rho2 = qwalk_gen_nonmarkov(int(t),psiminus,45,f,amp).ptrace(0)
            dist = trace_distance(rho1,rho2) # D(\rho1(t),\rho2(t))
        return dist
        
    nmarkov_diff=[]
    for y in range(1,51):
        nmarkov_diff.append(misc.derivative(sigma,y))
    
    measure=[]    
    for i in range(0,len(nmarkov_diff)):
        if nmarkov_diff[i]>0:
            measure.append(nmarkov_diff[i])
    blp.append(np.trapz(measure)) 
    return blp
    
    
