import numpy as np
rng=np.random.default_rng(0)
S=np.array([[1.0,0.5],[0.5,2.0]]); L=np.linalg.cholesky(S)
for n in (10,30):
    R=400000; X=rng.standard_normal((R,n,2))@L.T + np.array([0.3,-0.2])
    Xc=X-X.mean(1,keepdims=True)
    mle=np.einsum('rni,rnj->rij',Xc,Xc)/n          # sigma tilde (MLE)
    us=mle*n/(n-1)                                 # sigma tilde^s (unbiased)
    true=(S[0,1]**2+S[0,0]*S[1,1])/(n-1)
    D=n**3+n**2-2*n-4
    yz=n**2*(n**2-n-4)/((n-1)**2*D)*mle[:,0,1]**2 + n**3/((n-1)*D)*mle[:,0,0]*mle[:,1,1]; yz2=n**2*(n**2-n-4)/((n-1)**2*D)*us[:,0,1]**2 + n**3/((n-1)*D)*us[:,0,0]*us[:,1,1]
    m=n-1  # our formula, Wishart df m, applied to unbiased S
    ours=((m-2)*us[:,0,1]**2 + m*us[:,0,0]*us[:,1,1])/((m+2)*(m-1))
    print(f"n={n}: true Var={true:.5f}  empirical Var={us[:,0,1].var():.5f}  E[YZ 2.4]={yz.mean():.5f}  E[ours]={ours.mean():.5f}  E[YZ with unbiased S]={yz2.mean():.5f}")
