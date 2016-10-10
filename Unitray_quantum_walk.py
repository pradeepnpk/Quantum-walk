#      UNITARY_QUANTUM_WALK: Simulates discrete time quantum walk on a 1-d lattice.      
#                          : Probablity distribution plot                                                             
#                                                                                        
# Main Variables: t           - number of steps of quantum walk.                               
#               : sites       - number of lattice positions available to the walker (2*t+1).   
#               : coin_angle  - parameter of the SU(2) coin.
#               : qubit_state - input coin state
#               : z (1/2)     - plot with zeros/plot without zeros           
# Useful ref    : qutip - http://qutip.org/docs/3.1.0/guide/guide-basics.html                                                                                    
##########################################################################################

from qutip import *
import numpy as np
import matplotlib.pyplot as plt
from math import *


#Basis states
#basis(d,n), creates a coloum vector with dimension d and n=0,1..d-1 representing the position where one appears.
#Ex: basis(2,0): 2-d vector with '1' appearing in the 0th position [1,0]^Transpose
ket0 = basis(2,0).unit( )                # |0>
ket1 = basis(2,1).unit()                 # |1>
psip = (basis(2,0)+basis(2,1)*1j).unit() # |0>+i|1>/sqrt(2); unit() is used to normalize the state
psim = (basis(2,0)-basis(2,1)*1j).unit() # |0>-i|1>/sqrt(2)

#Coin transformation
#qutip.Qobj is a ([[]]) is a way to create a matrix or an array in qutip
def coin(coin_angle):
  C_hat = qutip.Qobj([[cos(radians(coin_angle)), sin(radians(coin_angle))],
                      [sin(radians(coin_angle)), -cos(radians(coin_angle))]])
  return C_hat

#Position transformation
def shift(t):
  sites = 2*t+1
  shift_l = qutip.Qobj(np.roll(np.eye(sites), 1, axis=0))  #left chairality. Roll function is a general way to realize shift operator matrix
  shift_r = qutip.Qobj(np.roll(np.eye(sites), -1, axis=0)) #right chairality
  S_hat = tensor(ket0*ket0.dag(),shift_l) + tensor(ket1*ket1.dag(),shift_r) #Complete shift Operator form
  return S_hat

#Walk operator: Evolution operator for DTQW
#tensor is a function called from qutip
def walk(t,coin_angle):
  sites = 2*t+1
  C_hat = coin(coin_angle) 
  S_hat = shift(t)     
  W_hat = S_hat*(tensor(C_hat,qeye(sites))) #combine both coin and shift
  return W_hat

#Quantum walk generator: outputs the evolved wave function after 't' steps.
#ket2dm is a qutip function used to convert kets to density matrices 
def qwalk_gen(t,qubit_state,coin_angle):
  sites=2*t+1
  Position_state = basis(sites,t)           
  Psi = ket2dm(tensor(qubit_state,Position_state)) # Initial state - \rho(0)  
  W_hat = walk(t,coin_angle)              
  for i in range(t):                               #Apply the walk operator 't' times.
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

#Plot function
def plot_pdf(P_p):
    lattice_positions = range(-len(P_p)/2+1,len(P_p)/2+1)
    plt.plot(lattice_positions,P_p)
    plt.xlim([-len(P_p)/2+2,len(P_p)/2+2])
    plt.ylim([min(P_p),max(P_p)+0.01])
    plt.ylabel(r'$Probablity$')
    plt.xlabel(r'$Particle \ position$')
    plt.show()


# the main instance of the program
if __name__ == "__main__":           # this line is not necessary (good practice to use though)
  Psi_t = qwalk_gen(100,psip,45)     # call the qwalk genrator
  P_p  = measurement(100,Psi_t,2)    # measure the wave function returned by qwalk generator.
  plot_pdf(P_p)                      # here z=2 in the measurement func implies the measurements are made only at even positions 
