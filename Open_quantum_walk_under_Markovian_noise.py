# QUANTUM_WALK_UNDER_MARKOVIAN_NOISE: Quantum walk evolution subjected to markovian noise channels 
#                                   : is implemented using 4 different noise channels.
#
# Main variables: p_c         - error probablity for the given channel
#                 kai         - noise parameter
#               : t           - number of steps of quantum walk.                               
#               : sites       - number of lattice positions available to the walker (2*t+1).   
#               : coin_angle  - parameter of the SU(2) coin.
#               : qubit_state - input coin state
#               : z (1/2)     - plot with zeros/plot without zeros 
############################################################################################

from qutip import *
import numpy as np
import matplotlib.pyplot as plt
from math import *
import seaborn as sns


#Basis states
ket0 = basis(2,0).unit() # |0>
ket1 = basis(2,1).unit() # |1>
psip = (basis(2,0)+basis(2,1)*1j).unit() # |0>+i|1>/sqrt(2)
psim = (basis(2,0)-basis(2,1)*1j).unit() # |0>-i|1>/sqrt(2)

#Coin transformation
def coin(coin_angle):
    C_hat = qutip.Qobj([[cos(radians(coin_angle)), sin(radians(coin_angle))],  #one paramter SU(2) matrix
                     [sin(radians(coin_angle)), -cos(radians(coin_angle))]])
    return C_hat

#Position transformation
def shift(t):
    sites = 2*t+1
    shift_l = qutip.Qobj(np.roll(np.eye(sites), 1, axis=0)) #left chairality #roll function is a general way to realize shift operator matrix
    shift_r = qutip.Qobj(np.roll(np.eye(sites), -1, axis=0)) #right chairality
    S_hat = tensor(ket0*ket0.dag(),shift_l) + tensor(ket1*ket1.dag(),shift_r) 
    return S_hat

#Walk operator: Evolution operator for DTQW
def walk(t,coin_angle):
    sites = 2*t+1
    C_hat = coin(coin_angle) 
    S_hat = shift(t)     
    W_hat = S_hat*(tensor(C_hat,qeye(sites))) #combine both coin and shift
    return W_hat

#Dephasing channel
def dephasing(t,qstate,p_c):
    sites=2*t+1
    dstate = (1-p_c)*qstate+p_c*(tensor(sigmaz(),qeye(sites))*qstate*tensor(sigmaz(),qeye(sites)))
    return dstate

#Deplorizing channel
def depolarizing(t,qstate,p_c):
    sites=2*t+1
    dstate = (1-p_c)*qstate+p_c/3*(tensor(sigmax(),qeye(sites))*(qstate)*tensor(sigmax(),qeye(sites))*
                           tensor(sigmay(),qeye(sites))*(qstate)*tensor(sigmay(),qeye(sites))*
                           tensor(sigmaz(),qeye(sites))*(qstate)*tensor(sigmaz(),qeye(sites)))
    return dstate

#Amplitude damping
def ampdamping(t,qstate,p_c):
    sites=2*t+1
    K1 = qutip.Qobj([[1,        0],  
                 [0,sqrt(1-p_c)]])
    
    K2 = qutip.Qobj([[0,  sqrt(p_c)],
                 [0,        0]])
    dstate = (tensor(K1,qeye(sites))*(qstate)*tensor(K1.dag(),qeye(sites))+
              tensor(K2,qeye(sites))*(qstate)*tensor(K2.dag(),qeye(sites)))
    return dstate 

#Generalized amplitude damping channel
def Gampdamping(t,qstate,p_c):
    sites=2*t+1
    kai = 1
    K1 = qutip.Qobj([[sqrt(kai),                   0],
                 [0,        sqrt(kai)*sqrt(1-p_c) ]])
    
    K2 = qutip.Qobj([[0,       sqrt(kai)*sqrt(p_c)],
                 [0,                           0]])
    
    K3 = qutip.Qobj([[sqrt(1-kai)*sqrt(1-p_c),     0],
                 [0,                 sqrt(1-kai_c)]])
    
    K4 = qutip.Qobj([[0,                           0],
                 [sqrt(1-kai)*sqrt(p_c),         0]])
    
    dstate = (tensor(K1,qeye(sites))*(qstate)*tensor(K1.dag(),qeye(sites))+
              tensor(K2,qeye(sites))*(qstate)*tensor(K2.dag(),qeye(sites))+
              tensor(K3,qeye(sites))*(qstate)*tensor(K3.dag(),qeye(sites))+
              tensor(K4,qeye(sites))*(qstate)*tensor(K4.dag(),qeye(sites)))
    return dstate

#Quantum walk generator: outputs the evolved wave function after 't' steps.
def qwalk_gen_markov(t,qubit_state,coin_angle,p_c):
    sites=2*t+1
    Position_state = basis(sites,t)           
    Psi = ket2dm(tensor(qubit_state,Position_state)) # Initial state - \rho(0)  
    W_hat = walk(t,coin_angle)                       # Total Qwalk operator
    for i in range(t):                               # Apply the walk operator 't' times
        Psi = W_hat*dephasing(t,Psi,p_c)*W_hat.dag() # krauss op is applied 't' times; change the name 'dephasing' for other modes
    return Psi                                       # returns decohered state

#Projective measurement on the position basis states. 
def measurement(t,Psi,z):
    sites=2*t+1
    prob=[]
    for i in range(0,sites,z):
        M_p = basis(sites,i)*basis(sites,i).dag()
        Measure = tensor(qeye(2),M_p)                 # I_c \tensor M_p 
        p = abs((Psi*Measure).tr())                   # Probablity
        prob.append(p)
    return prob

#Plot the probablity distribution
def plot_pdf(P_p):
    lattice_positions = range(-len(P_p)+1,len(P_p)+1,2)
    plt.plot(lattice_positions,P_p)
    plt.xlim([-len(P_p)+1,len(P_p)+1])           #sets the limits by finding min and max from dataset
    plt.ylim([min(P_p),max(P_p)+0.01])
    plt.ylabel(r'$Probablity$')
    plt.xlabel(r'$Particle \ position$')
    plt.show()

# Main instance of the program
if __name__ == "__main__":  # this line is not necessary (good practice to use though, will be convinient when writing classes))
    Psi_t = qwalk_gen_markov(100,psip,45,0.01) # call the qwalk generator
    P_p  = measurement(100,Psi_t,2)            # measure the wave function returned by qwalk generator. # here z=2 in the measurement func implies the measurements are made only at even positions 
    plot_pdf(P_p)
