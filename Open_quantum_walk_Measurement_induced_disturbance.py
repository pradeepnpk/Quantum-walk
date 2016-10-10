# QUANTUM_WALK_UNDER_MARKOVIAN_NOISE: Quantum walk correlation calculations under non-Markovian noise
#                                   : Implemented MID exactly from Luo's paper 
#                                   : Ref: Phy. Rev. A. 77, 022301 2008
#
# Main variables: p_c         - error probablity for the given channel
#                 kai         - noise parameter
#               : t           - number of steps of quantum walk.                               
#               : sites       - number of lattice positions available to the walker (2*t+1).   
#               : coin_angle  - parameter of the SU(2) coin.
#         
#              
####################################################################################################

from qutip import *
import numpy as np
import matplotlib.pyplot as plt
from math import *
import cmath as cm

#Basis states
ket0 = basis(2,0).unit() # |0>
ket1 = basis(2,1).unit() # |1>
psip = (basis(2,0)+basis(2,1)*1j).unit() # |0>+i|1>/sqrt(2)
psim = (basis(2,0)-basis(2,1)*1j).unit() # |0>-i|1>/sqrt(2)

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

#Kraus operators for the non-Markovian master equation under RTN
def telegraph_noise(t,sites,qstate,freq,amp):
    nu = t*freq; mu = cm.sqrt((2.0*amp*(1/freq))**2-1.0)   # Noise parameters
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

# von neumann entropy:
def von_entropy(state):
    ev = state.eigenenergies()
    Vn_ent=[]
    for i in range(len(ev)):
        Vn_ent.append(-ev[i]*np.log2(ev[i])) 
    Vn_ent = filter(lambda v: v==v, Vn_ent) #to remove nans from the list
    return sum(Vn_ent)

#Quantum mutual information.
def mutual_info(rho):
    rhoA = rho.ptrace(0)    #rho_c
    rhoB = rho.ptrace(1)    #rho_p
    out =  von_entropy(rhoA)+von_entropy(rhoB)-von_entropy(rho)
    return out

#Genrate projectors
def projectors(qstate):
    ev, evec = qstate.eigenstates()
    m=[]
    for l in range(len(evec)):
        m.append(ket2dm(evec[l]))
    return m

#Measurement induced disturbance:
#Ref: Phy. Rev. A. 77, 022301 2008
def quantum_mid(rho):
    proj_w=[];proj_rho=[]
    rho_c = rho.ptrace(0)
    rho_p = rho.ptrace(1)
    proj_c = projectors(rho_c)              #projectors for coin
    proj_p = projectors(rho_p)              #projectors for position
    for n in range(len(proj_c)):            #calculate the MID state by projective measurements
        for p in range(len(proj_p)):
            M = tensor(proj_c[n],proj_p[p])
            proj_rho.append(M*rho*M.dag())
    rhoMID = sum(proj_rho)                  #Measurment Induced state
    Im_rho = mutual_info(rho)              
    Im_mid = mutual_info(rhoMID)
    Q_mid = Im_rho-Im_mid
    return np.real(Q_mid)                   #distance between raw qstate and MID qstate  

def plot_mid(mid_data):
    t = range(0,len(mid_data))
    plt.plot(t,mid_data)
    plt.ylim(min(mid_data),max(mid_data))
    plt.xlim(min(t),max(t)+1)
    plt.ylabel(r'$\mathcal{Q}_{MID}$')
    plt.xlabel(r'$t$')
    plt.show()

# the main instance of the program
if __name__ == "__main__":  #this line is not necessary(good practice to use though, will be convinient when writing classes)
    q_mid=[]
    #noise parameters:
    freq=7.0; amp=1.0
    for t in range(1,101):
        rho = qwalk_gen_nonmarkov(t,psip,45,freq,amp)
        q_mid.append(quantum_mid(rho)) 
    plot_mid(q_mid)
