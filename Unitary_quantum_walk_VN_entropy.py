# UNITARY_QUANTUM_WALK_VN_ENTROPY: Computes non-neumann entropy of the reeduced coin state and poition state   
#                                                                                        
#                                                                                        
# Main Variables: t           - number of steps of quantum walk.                               
#               : sites       - number of lattice positions available to the walker (2*t+1).   
#               : coin_angle  - parameter of the SU(2) coin.
#               : qubit_state - input coin state
#               : z (1/2)     - plot with zeros/plot without zeros           
#                                                                                           
#############################################################################################################

from qutip import *
import numpy as np
import matplotlib.pyplot as plt
from math import *


#Basis states
ket0 = basis(2,0).unit( )                # |0>
ket1 = basis(2,1).unit()                 # |1>
psip = (basis(2,0)+basis(2,1)*1j).unit() # |0>+i|1>/sqrt(2)
psim = (basis(2,0)-basis(2,1)*1j).unit() # |0>-i|1>/sqrt(2)

#Coin transformation
def coin(coin_angle):
  C_hat = qutip.Qobj([[cos(radians(coin_angle)), sin(radians(coin_angle))],
                      [sin(radians(coin_angle)), -cos(radians(coin_angle))]])
  return C_hat

#Position transformation
def shift(t):
  sites = 2*t+1
  shift_l = qutip.Qobj(np.roll(np.eye(sites), 1, axis=0))  #left chairality. Roll function is a general way to realize shift operator matrix
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

#Quantum walk generator: outputs the evolved wave function after 't' steps.
def qwalk_gen(t,qubit_state,coin_angle):
  sites=2*t+1
  Position_state = basis(sites,t)           
  Psi = ket2dm(tensor(qubit_state,Position_state)) # Initial state - \rho(0)  
  W_hat = walk(t,coin_angle)              
  for i in range(t):          
    Psi = W_hat*Psi*W_hat.dag()
  return Psi

#Projective measurement on the position basis states. 
#The walker has a zero probablity at odd positions of the lattice. 
#The zeros can be avoided if we measure the qubit only at even positions, this can be done by setting z=2.
def measurement(t,Psi,z):
  sites=2*t+1
  prob=[]
  for i in range(0,sites,z):
    M_p = basis(sites,i)*basis(sites,i).dag() #Outer product
    Measure = tensor(qeye(2),M_p)             #Identity on coin M_p on position
    p = abs((Psi*Measure).tr())               #Probablity
    prob.append(p)
  return prob

# von neumann entropy:
def von_entropy(state):
    ev = state.eigenenergies() #obtain all the eigen values
    Vn_ent=[]
    for i in range(len(ev)):
        Vn_ent.append(ev[i]*np.log2(ev[i])) 
    Vn_ent = filter(lambda v: v==v, Vn_ent) #to remove undefined values from the list; ex: log2(0)
    return np.real(-sum(Vn_ent))

#Plot 
def plot_vn_entropy(Vn):
    t_vn = np.arange(0,101)
    plt.plot(t_vn,Vn)
    plt.ylim([0,1])
    plt.ylabel(r'$Von \ neumann \ entropy$')
    plt.xlabel(r'$t$')
    plt.show()

# the main instance of the program
if __name__ == "__main__":  #this line is not necessary(good practice to use though, will be convinient when writing classes)
    vn_entropy=[0]
    for t in range(1,101):
        rho_c = qwalk_gen(t,psip,45).ptrace(0) #partial trace function. ptrace(0) implies remove all system other than 0. Here keep the coin and remove position degress of freedom
        vn_entropy.append(von_entropy(rho_c))

    plot_vn_entropy(vn_entropy)
