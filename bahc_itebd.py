""" iTEBD code to find the ground state of 
the 1D bond-alternating Heisenberg model on an infinite chain.
The results are compared to the exact results.
Frank Pollmann, frankp@pks.mpg.de"""

import numpy as np
from scipy import integrate
from scipy.linalg import expm 

# First define the parameters of the model / simulation
Jx=1.0; Jy=1.0; Jz=1.6; g=0.0; eps=0.0; chi=15; d=2; delta1=0.1; N1=250; delta2=0.005; N2=5000;

B=[];s=[]
for i in range(2):
    B.append(np.zeros([2,1,1]))
    s.append(np.ones([1]))
B[0][0,0,0]=0.8; B[0][1,0,0]=0.6; B[1][0,0,0]=0.6; B[1][1,0,0]=0.8
    
# Generate the two-site time evolution operator
H_bond = [np.array( [[Jz*(1-eps),-g/2,-g/2,(Jx-Jy)*(1-eps)], [-g/2,-Jz*(1-eps),(Jx+Jy)*(1-eps),-g/2], [-g/2,(Jx+Jy)*(1-eps),-Jz*(1-eps),-g/2], [0,-g/2,-g/2,Jz*(1-eps)]] ),
          np.array( [[Jz*(1+eps),-g/2,-g/2,(Jx-Jy)*(1+eps)], [-g/2,-Jz*(1+eps),(Jx+Jy)*(1+eps),-g/2], [-g/2,(Jx+Jy)*(1+eps),-Jz*(1+eps),-g/2], [0,-g/2,-g/2,Jz*(1+eps)]] )]

U1 = []
for i in range(2):
    U1.append(np.reshape(expm(-delta1*H_bond[i]),(2,2,2,2)))
    
    # Perform the imaginary time evolution alternating on A and B bonds(step1)
for step in range(0, N1):
    for i_bond in [0,1]:
        ia = np.mod(i_bond-1,2); ib = np.mod(i_bond,2); ic = np.mod(i_bond+1,2)
        chia = B[ib].shape[1]; chic = B[ic].shape[2]
        
        # Construct theta matrix and time evolution #
        theta = np.tensordot(B[ib],B[ic],axes=(2,1)) # i a j b
        theta = np.tensordot(U1[ib],theta,axes=([2,3],[0,2])) # ip jp a b
        theta = np.tensordot(np.diag(s[ia]),theta,axes=([1,2])) # a ip jp b 
        theta = np.reshape(np.transpose(theta,(1,0,2,3)),(d*chia,d*chic)) # ip a jp b
        
        # Schmidt decomposition #
        X, Y, Z = np.linalg.svd(theta,full_matrices=0)
        chi2 = np.min([np.sum(Y>10.**(-15)), chi])
        
        piv = np.zeros(len(Y), np.bool)
        piv[(np.argsort(Y)[::-1])[:chi2]] = True
        
        Y = Y[piv]; invsq = np.sqrt(sum(Y**2))
        X = X[:,piv] 
        Z = Z[piv,:]
        
        # Obtain the new values for B and s #
        s[ib] = Y/invsq 
        
        X=np.reshape(X,(d,chia,chi2))
        X = np.transpose(np.tensordot(np.diag(s[ia]**(-1)),X,axes=(1,1)),(1,0,2))
        B[ib] = np.tensordot(X, np.diag(s[ib]),axes=(2,0))
        
        B[ic] = np.transpose(np.reshape(Z,(chi2,d,chic)),(1,0,2))


U2 = []
for i in range(2):
    U2.append(np.reshape(expm(-delta2*H_bond[i]),(2,2,2,2)))

# Perform the imaginary time evolution alternating on A and B bonds(step2)
for step in range(0, N2):
    for i_bond in [0,1]:
        ia = np.mod(i_bond-1,2); ib = np.mod(i_bond,2); ic = np.mod(i_bond+1,2)
        chia = B[ib].shape[1]; chic = B[ic].shape[2]
        
        # Construct theta matrix and time evolution #
        theta = np.tensordot(B[ib],B[ic],axes=(2,1)) # i a j b
        theta = np.tensordot(U2[ib],theta,axes=([2,3],[0,2])) # ip jp a b
        theta = np.tensordot(np.diag(s[ia]),theta,axes=([1,2])) # a ip jp b
        theta = np.reshape(np.transpose(theta,(1,0,2,3)),(d*chia,d*chic)) # ip a jp b
        
        # Schmidt decomposition #
        X, Y, Z = np.linalg.svd(theta,full_matrices=0)
        chi2 = np.min([np.sum(Y>10.**(-15)), chi])
        
        piv = np.zeros(len(Y), np.bool)
        piv[(np.argsort(Y)[::-1])[:chi2]] = True
        
        Y = Y[piv]; invsq = np.sqrt(sum(Y**2))
        X = X[:,piv]
        Z = Z[piv,:]
        
        # Obtain the new values for B and s #
        s[ib] = Y/invsq
        
        X=np.reshape(X,(d,chia,chi2))
        X = np.transpose(np.tensordot(np.diag(s[ia]**(-1)),X,axes=(1,1)),(1,0,2))
        B[ib] = np.tensordot(X, np.diag(s[ib]),axes=(2,0))
        
        B[ic] = np.transpose(np.reshape(Z,(chi2,d,chic)),(1,0,2))

# Get the bond energies
E=[]
for i_bond in range(2):
    BB = np.tensordot(B[i_bond],B[np.mod(i_bond+1,2)],axes=(2,1))
    sBB = np.tensordot(np.diag(s[np.mod(i_bond-1,2)]),BB,axes=(1,1))
    C = np.tensordot(sBB,np.reshape(H_bond[i_bond],[d,d,d,d]),axes=([1,2],[2,3]))
    sBB=np.conj(sBB)
    E.append(np.squeeze(np.tensordot(sBB,C,axes=([0,3,1,2],[0,1,2,3]))).item()) 
print "E_iTEBD =", np.mean(E)
    
# Get the expectation value of twist operator

jsz = np.array([[complex(0,1),0],[0,complex(0,-1)]],dtype=np.complex)

for k in range(50):
    L = 2*(k+1)
    Tz = []
    for i in range(L):
        Tz.append(expm(2*i*np.pi*jsz/L))
        
    BT = np.tensordot(B[0],Tz[0],axes=(0,0))
    C = np.transpose(np.tensordot(BT,np.conj(B[0]),axes=(2,0)),(0,2,1,3))
    N = np.transpose(np.tensordot(B[0],np.conj(B[0]),axes=(0,0)),(0,2,1,3))
    for i in range(L-1):
        BT = np.tensordot(B[np.mod(i+1,2)],Tz[i+1],axes=(0,0))
        BTB = np.tensordot(BT,np.conj(B[np.mod(i+1,2)]),axes=(2,0))
        BTB = np.transpose(BTB,(0,2,1,3))
        BB = np.transpose(np.tensordot(B[np.mod(i+1,2)],np.conj(B[np.mod(i+1,2)]),axes=(0,0)),(0,2,1,3))
        C = np.tensordot(C,BTB)
        N = np.tensordot(N,BB)
        
    zL = np.trace(np.trace(C,axis1=0,axis2=2))/np.trace(np.trace(N,axis1=0,axis2=2))
    print L, np.real(zL), np.imag(zL)
