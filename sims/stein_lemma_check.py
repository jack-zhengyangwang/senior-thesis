import numpy as np
rng=np.random.default_rng(0)
N,T,R=3,8,400000
A=rng.normal(size=(N,N)); Sig=A@A.T+np.eye(N)
L=np.linalg.cholesky(Sig)
X=rng.normal(size=(R,T,N))@L.T              # R reps, T days, N assets, mean 0
S=np.einsum('rti,rtj->rij',X,X)/T
i,j=0,1
s=S[:,i,j]
print("Var(s_ij) MC     ",s.var())
print("Gaussian formula ",(Sig[i,i]*Sig[j,j]+Sig[i,j]**2)/T)
est=(T*S[:,i,i]*S[:,j,j]+(T-2)*s**2)/((T+2)*(T-1))
print("E[unbiased est]  ",est.mean(), "+/-", est.std()/np.sqrt(R))
# Stein identity check: h = s_ij^2 (nonlinear), t=0
h=s**2
lhs=np.mean((X[:,0,i]*X[:,0,j]-Sig[i,j])*h)
# dh/dx_{0k} = 2 s_ij * (d_ik x_0j + d_jk x_0i)/T
rhs=0
for k in range(N):
    dh=2*s*((i==k)*X[:,0,j]+(j==k)*X[:,0,i])/T
    rhs+=Sig[i,k]*np.mean(X[:,0,j]*dh)
print("Stein identity   ",lhs, rhs)
