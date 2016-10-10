# QUANTUM_WALK_UNDER_MARKOVIAN_NOISE: Quantum walk evolution subjected to markovian noise 
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

#Walk operator: Evolution operator for DTQW
def walk(sites,coin_angle):
    C_hat = coin(coin_angle) 
    S_hat = shift(sites)     
    W_hat = S_hat*(tensor(C_hat,qeye(sites))) 
    return W_hat

#Kraus operators for the non-Markovian master equation under RTN
def telegraph_noise(t,sites,qstate,freq,amp):
    nu = t*freq; mu = sqrt((2.0*amp*(1/freq))**2-1.0)      # Noise parameters
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

#Variance of the quantum walk distribution
def walk_variance(t,qubit_state,coin_angle,freq,amp):
    sites=2*t+1
    Psi = qwalk_gen_nonmarkov(t,qubit_state,coin_angle,freq,amp)
    prob = measurement(t,Psi,1)
    #\sigma^2 = \sum_{i=1}^{i=sites} p_i(i-\mu)^2
    lattice_sites = [-t+i for i in range(0,sites)]               #generates the lattice positions "i"
    variance=[0]                                                 #Just an empty array to store the data
    for i in lattice_sites:
        variance.append(prob[i]*pow(lattice_sites[i]-np.average(prob),2))
    return sum(variance) 

#plot
def plot_variance(variance_data):
    t = range(0,len(variance_data))
    plt.plot(t,variance_data)
    plt.ylim(min(variance_data),max(variance_data))
    plt.xlim(min(t),max(t))
    plt.ylabel(r'$\sigma^2(t)$')
    plt.xlabel(r'$t$')
    plt.show() 

# the main instance of the program
if __name__ == "__main__":  #this line is not necessary(good practice to use though, will be convinient when writing classes)
    sigma_sq=[0]
    #noise parameters
    freq=0.001; amp=0.05
    #compute variance after every step 't' of the quantum walk and store it in an array
    for t in range(1,101):
        v=walk_variance(t,psip,45,freq,amp)
        sigma_sq.append(v)

    plot_variance(sigma_sq)
