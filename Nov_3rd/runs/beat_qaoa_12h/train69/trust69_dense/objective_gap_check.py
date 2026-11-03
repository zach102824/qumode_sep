import sys,json,torch,numpy as np
sys.path.insert(0,'.'); sys.path.insert(0,'Nov_3rd/scripts')
torch.set_num_threads(4)
from train69_dense_trust_ball import BatchedForward
from train_beat_train69 import load_train_n, precompute_qaoa_inv
from Nov_3rd.ansatz import make_sim
from Nov_3rd.data_io import load_qaoa
from Nov_3rd.eval_scale import eval_n
from beat_params import phi_from_c, warm_from_qaoa
from Nov_3rd.data_io import load_split_E
from Nov_3rd import settings as S
b,g=load_qaoa(2); g0,c1,_,_=warm_from_qaoa(b,g)
for name in ('rho0.05_ns64','rho0.08_ns48'):
    st=torch.load(f'Nov_3rd/runs/beat_qaoa_12h/train69/trust69_dense/{name}/params_L2_rank0.pt',weights_only=False)
    phi=phi_from_c(c1,st['c2'],st['c3'])
    for n in (6,9):
        for split in ('train','scale1000'):
            if split=='train': E=load_train_n(n)
            else: E,_=load_split_E(n,'scale1000',1000,'cpu',data_dir=S.SCALE1000_DIR)
            iq=precompute_qaoa_inv(b,g,E,n,'cpu').numpy()
            with torch.no_grad(): ia=torch.exp(BatchedForward(make_sim(n,2,'cpu'),E).log_inv_p(g0,phi)).numpy()
            print(name,n,split,'median_of_ratios=%.4f ratio_of_medians=%.4f'%(np.median(ia/iq),np.median(ia)/np.median(iq)))
