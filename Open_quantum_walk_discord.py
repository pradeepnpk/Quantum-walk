# QUANTUM_WALK_UNDER_MARKOVIAN_NOISE: Quantum walk discord calulation under non_markovian noise 
#                                   : Implemened exactly from Zurek's paper
#                                   : Phys. Rev. Lett.88.017901
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
import cmath as cm
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
    Vn_ent = filter(lambda v: v==v, Vn_ent) #to remove undefined values from the list; ex: log2(0)
    return sum(Vn_ent)

#Quantum discord (Zurek's paper:PhysRevLett.88.017901)
def quantum_discord(t,rho):
    sites=2*t+1
    discord=[]
    for theta in range(0,180,9):
        for phi in range(0,360,12):
            E = qubit_gen(theta,phi).eigenstates() #Sample states from bloch sphere
            Eval = E[0]; Ev = E[1]
            Evec0=ket2dm(Ev[0])
            Evec1=ket2dm(Ev[1]) # compute projectors from states
            
            #Q_discord = H(A)-H(B)+H(B|\proj)
            Ipc = von_entropy(rho.ptrace(0))-von_entropy(rho) #H(A)-H(B)
            
            p0=(tensor(Evec0,qeye(sites))*rho).tr() #probality for outcomes
            p1=(tensor(Evec1,qeye(sites))*rho).tr()     
            rho_c1 = von_entropy((tensor(Evec0,qeye(sites))*rho*tensor(Evec0,qeye(sites)))/p0) #measurement induced state
            rho_c2 = von_entropy((tensor(Evec1,qeye(sites))*rho*tensor(Evec1,qeye(sites)))/p1)
            
            Q_discord = Ipc+(p0*rho_c1+p1*rho_c2) #(H(A)-H(B)) + H(B|\proj)
            discord.append(Q_discord.real)
    return min(discord)

#Bloch sphere sampling
def qubit_gen(theta,phi):
    psi = cos(radians(theta))*ket0+complex(cos(radians(phi)),sin(radians(phi)))*sin(radians(theta))*ket1 #general forma a qubit
    return ket2dm(psi) 

def plot_discord(dis_data):
    t = range(0,len(dis_data))
    plt.plot(t,dis_data)
    plt.ylim(min(dis_data),max(mid_data))
    plt.xlim(min(t),max(t))
    plt.ylabel(r'$\mathcal{Q}_{MID}$')
    plt.xlabel(r'$t$')
    plt.show()

# the main instance of the program
if __name__ == "__main__":  #this line is not necessary(good practice to use though, will be convinient when writing classes)
    q_dis=[]
    #noise parameters:
    freq=0.005; amp=1.0
    for t in range(1,51):
        rho = qwalk_gen_nonmarkov(t,psip,45,freq,amp)
        q_dis.append(quantum_discord(t,rho))
    
    plot_discord(q_dis)
