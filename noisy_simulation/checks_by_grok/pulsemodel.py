"""Independent QuTiP layer model for c7/c11: finite-duration pulses with Lindblad noise during them.
Modes (d, e, A, B) [+ coupler q during JP]. Transmon dim 2 (qubit) or 3 (qutrit, |f>).
Assumptions (stated in report): rotation pulses sin^2 envelope, t_q = 24 ns (Eickbusch tq);
ECD = echoed conditional displacement: CD half (generator Z(x)G, G = beta/4 a^dag - h.c.), echo pi pulse
(24 ns, X), CD half with -G; block total = tau_ECD = 0.65 us, so each CD half = (0.65 - 0.048)/2 us.
JP gate: same physical sequence as NOISE.md (instantaneous q pulses, two chi*t = pi waits)."""
import numpy as np, qutip as qt
OPT={"atol":1e-10,"rtol":1e-8,"nsteps":10**6,"max_step":2e-9}
TQ=24e-9
def env(T):  # sin^2 envelope, integral 1
    return lambda t: (2.0/T)*np.sin(np.pi*t/T)**2 if 0<=t<=T else 0.0
class Model:
    def __init__(s,dev,nf,dq=2,nth_q=0.0,nth_c=0.0,anh=-2*np.pi*193e6):
        s.dev,s.nf,s.dq,s.nth_q,s.nth_c,s.anh=dev,nf,dq,nth_q,nth_c,anh
        s.b=qt.destroy(dq); s.nq=s.b.dag()*s.b; s.a=qt.destroy(nf)
        s.Zq=qt.Qobj(np.diag([1.0-2*k for k in range(dq)]))   # g:+1, e:-1, f:-3 (CD amplitude ~ linear in level)
        s.Hanh=0.5*anh*s.nq*(s.nq-1)
    def emb(s,M,k,mods):
        dims={'q':2,'d':s.dq,'e':s.dq,'A':s.nf,'B':s.nf}
        return qt.tensor(*[M if m==k else qt.qeye(dims[m]) for m in mods])
    def cops(s,mods):
        d=s.dev; c=[]
        for m in mods:
            if m in 'AB':
                c+=[np.sqrt(d.kappa*(1+s.nth_c))*s.emb(s.a,m,mods)]
                if s.nth_c>0: c+=[np.sqrt(d.kappa*s.nth_c)*s.emb(s.a.dag(),m,mods)]
            else:
                b=qt.destroy(2) if m=='q' else s.b; n=b.dag()*b
                c+=[np.sqrt(d.gamma*(1+s.nth_q))*s.emb(b,m,mods),np.sqrt(2*d.gamma_phi)*s.emb(n,m,mods)]
                if s.nth_q>0: c+=[np.sqrt(d.gamma*s.nth_q)*s.emb(b.dag(),m,mods)]
        return [x for x in c if x.norm()>0]
    def H0(s,mods,anh=False):
        # anharmonic term only matters while a transmon drive is on: it commutes with the CD generator (diagonal
        # in the transmon), with the JP Hamiltonian, and D[b], D[n] are invariant in its rotating frame.
        H=0*s.emb(qt.qeye(2) if 'q' in mods else s.b,mods[0],mods)
        if s.dq>2 and anh:
            for m in 'de': H=H+s.emb(s.Hanh,m,mods)
        return H
    def rot_gen(s,theta,phi):   # Hermitian generator h with exp(-i h) = R(theta,phi) on the qubit subspace
        bp=s.b.dag(); return 0.5*theta*(np.exp(1j*phi)*bp+np.exp(-1j*phi)*s.b)
    def cd_gen(s,beta,m,c):     # Hermitian H with exp(-i H) = exp(Z (x) G), G = beta/4 a^dag - h.c.
        G=(beta/4)*s.a.dag()-np.conj(beta/4)*s.a; mods=['d','e','A','B']
        return 1j*s.emb(s.Zq,m,mods)*s.emb(G,c,mods)
    def evolve(s,rho,Hlist,T,mods,noisy,anh=False):
        c=s.cops(mods) if noisy else []
        H=[s.H0(mods,anh)]+Hlist
        return qt.mesolve(H,rho,[0,T],c,options=OPT).states[-1]
    def block(s,rho,lay,mode):
        """mode: 'pulse' (noise during finite pulses) | 'lumped' (ideal pulses, then idle tau_ECD) | 'ideal'"""
        mods=['d','e','A','B']; d=s.dev; tcd=(d.tau_ecd-2*TQ)/2
        X=lambda m: s.emb(s.rot_gen(np.pi,0.0),m,mods)
        segs=[(s.emb(s.rot_gen(lay['theta_d'],lay['phi_d']),'d',mods)+s.emb(s.rot_gen(lay['theta_e'],lay['phi_e']),'e',mods),TQ),
              (s.cd_gen(lay['beta_d'],'d','A')+s.cd_gen(lay['beta_e'],'e','B'),tcd),
              (X('d')+X('e'),TQ),
              (-(s.cd_gen(lay['beta_d'],'d','A')+s.cd_gen(lay['beta_e'],'e','B')),tcd)]
        for i,(Hg,T) in enumerate(segs):
            drv=(i%2==0)
            if mode=='pulse': rho=s.evolve(rho,[[Hg,env(T)]],T,mods,True,anh=drv)
            elif s.dq>2: rho=s.evolve(rho,[[Hg,env(T)]],T,mods,False,anh=drv)   # anharmonic: unitary still time-resolved
            else: U=(-1j*Hg).expm(); rho=U*rho*U.dag()
        if mode=='lumped': rho=s.evolve(rho,[],d.tau_ecd,mods,True)
        return rho
    def jp(s,rho,noisy=True):
        mods=['q','d','e','A','B']; d=s.dev
        r=qt.tensor(qt.basis(2,0).proj(),rho)
        ry=lambda th: s.emb(qt.Qobj(np.array([[np.cos(th/2),-np.sin(th/2)],[np.sin(th/2),np.cos(th/2)]])),'q',mods)
        Sd=s.emb(qt.Qobj(np.diag([1,-1j])),'q',mods)
        Pe=s.emb(qt.basis(2,1).proj(),'q',mods)
        H=-d.chi_qa*(s.emb(s.a.dag()*s.a,'A',mods)+s.emb(s.a.dag()*s.a,'B',mods))*Pe
        r=ry(np.pi/2)*r*ry(np.pi/2).dag()
        o=dict(OPT); o.pop("max_step")
        w=lambda r: qt.mesolve([s.H0(mods),H],r,[0,d.tau_wait],s.cops(mods) if noisy else [],options=o).states[-1]
        r=w(r); M=ry(np.pi/2)*Sd*ry(-np.pi/2); r=M*r*M.dag(); r=w(r)
        return r.ptrace([1,2,3,4])
    def layer(s,rho,lay,mode):
        return s.jp(s.block(rho,lay,mode),noisy=(mode!='ideal'))
    def vac(s):
        return qt.tensor(qt.basis(s.dq,0).proj(),qt.basis(s.dq,0).proj(),qt.basis(s.nf,0).proj(),qt.basis(s.nf,0).proj())

# ---- fast JP (same physics): d,e enter the JP gate only through their own noise, which commutes with the
# q-A-B evolution, so JP = C_AB (x) N_d (x) N_e. C_AB is built from the full q-A-B Lindblad propagator.
def _jp_fast(s,rho):
    d=s.dev; nf=s.nf; mods=['q','A','B']
    emb=lambda M,k: qt.tensor(*[M if m==k else qt.qeye(2 if m=='q' else nf) for m in mods])
    a=s.a; Pe=emb(qt.basis(2,1).proj(),'q'); H=-d.chi_qa*(emb(a.dag()*a,'A')+emb(a.dag()*a,'B'))*Pe
    c=[np.sqrt(d.kappa*(1+s.nth_c))*emb(a,'A'),np.sqrt(d.kappa*(1+s.nth_c))*emb(a,'B'),
       np.sqrt(d.gamma*(1+s.nth_q))*emb(qt.destroy(2),'q'),np.sqrt(2*d.gamma_phi)*emb(qt.num(2),'q')]
    if s.nth_c>0: c+=[np.sqrt(d.kappa*s.nth_c)*emb(a.dag(),'A'),np.sqrt(d.kappa*s.nth_c)*emb(a.dag(),'B')]
    if s.nth_q>0: c+=[np.sqrt(d.gamma*s.nth_q)*emb(qt.create(2),'q')]
    Pw=qt.propagator(H,d.tau_wait,c,options={"atol":1e-11,"rtol":1e-9}).full()
    ry=lambda th: np.array([[np.cos(th/2),-np.sin(th/2)],[np.sin(th/2),np.cos(th/2)]])
    D=2*nf*nf; I=np.eye(nf*nf)
    def Uq(M): U=np.kron(M,I); return lambda r: U@r@U.conj().T
    up=Uq(ry(np.pi/2)); mid=Uq(ry(np.pi/2)@np.diag([1,-1j])@ry(-np.pi/2))
    vec=lambda r: r.reshape(-1,order='F'); unvec=lambda v: v.reshape(D,D,order='F')
    P=nf*nf; C=np.zeros((P,P,P,P),complex)   # C[i,j,k,l]: out[i,j] from in[k,l]
    for k in range(P):
        for l in range(P):
            r=np.zeros((D,D),complex); r[k,l]=1   # q=g block index = k
            r=up(r); r=unvec(Pw@vec(r)); r=mid(r); r=unvec(Pw@vec(r))
            C[:,:,k,l]=r[:P,:P]+r[P:,P:]
    dq=s.dq; b=s.b; cq=[np.sqrt(d.gamma*(1+s.nth_q))*b,np.sqrt(2*d.gamma_phi)*s.nq]+([np.sqrt(d.gamma*s.nth_q)*b.dag()] if s.nth_q>0 else [])
    Nq=qt.propagator(0*s.nq,d.tau_jp,cq).full(); Nq4=Nq.reshape(dq,dq,dq,dq,order='F')  # vec(F) col-major: out[i,j]=sum Nq4[i,j,k,l] in[k,l]
    R=rho.full().reshape(dq,dq,P,dq,dq,P)   # (d,e,AB ; d',e',AB')
    # apply N_d on (axis0, axis3), N_e on (axis1, axis4), C on (axis2, axis5)
    R=np.einsum('ijkl,kbclef->ibcjef',Nq4,R)
    R=np.einsum('ijkl,akcdlf->aicdjf',Nq4,R)
    R=np.einsum('ijkl,abkdel->abidej',C,R)
    return qt.Qobj(R.reshape(dq*dq*P,dq*dq*P),dims=rho.dims)
Model.jp_fast=_jp_fast
