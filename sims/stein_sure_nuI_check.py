import numpy as np
rng=np.random.default_rng(1)
N,T,R=4,10,300000
A=rng.normal(size=(N,N)); Sig=A@A.T/N+np.diag([1,2,3,4.])
X=rng.normal(size=(R,T,N))@np.linalg.cholesky(Sig).T
S=np.einsum('rti,rtj->rij',X,X)/T
trS=np.trace(S,axis1=1,axis2=2); trS2=np.einsum('rij,rji->r',S,S)
nu=trS/N; F=nu[:,None,None]*np.eye(N)
a=np.trace(Sig)**2; b=np.trace(Sig@Sig)
FS2=((F-S)**2).sum((1,2))
Vhat=(T*trS**2+(T-2)*trS2)/((T+2)*(T-1))          # unbiased for (a+b)/T
bhat=T*(T*trS2-trS**2)/((T+2)*(T-1))              # unbiased for tr(Sigma^2)
print("E Vhat vs (a+b)/T:", Vhat.mean(), (a+b)/T)
print("E bhat vs b      :", bhat.mean(), b)
for al in [0,0.3,0.7,1.0]:
    risk_mc=(((al*F+(1-al)*S-Sig)**2).sum((1,2))).mean()
    risk_formula=(1-2*al)*(a+b)/T+al**2*FS2.mean()+4*al*b/(N*T)
    sure=((1-2*al)*Vhat+al**2*FS2+4*al*bhat/(N*T)).mean()
    print(f"alpha={al}: MC risk {risk_mc:.4f}  formula {risk_formula:.4f}  E[SURE] {sure:.4f}")
