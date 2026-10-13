import json, numpy as np, subprocess, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
rows=np.load("c7_rows.npy"); c11=json.load(open("c11_results.json"))
fig,ax=plt.subplots(1,3,figsize=(16,4.5))
L=rows[:,0]; ax[0].plot(L,rows[:,3],'o-',label="noise effect TD(repo, ideal)"); ax[0].plot(L,rows[:,4],'s-',label="TD(finite-pulse model, ideal)")
ax[0].plot(L,rows[:,5],'^-',label="model error TD(pulse, repo)"); ax[0].set_xlabel("layers"); ax[0].set_title("(7) noise inside pulses vs lumped"); ax[0].legend(fontsize=8)
names=list(c11); v=[c11[n][-1]['infid'] for n in names]; ax[1].barh(range(len(v)),v); ax[1].set_yticks(range(len(v))); ax[1].set_yticklabels(names,fontsize=7)
ax[1].set_xlabel("1-F after 2 layers"); ax[1].set_title("(11) thermal / |f> leakage")
txt=subprocess.run(["/workspace/venv-qumode/bin/python","c9_table.py"],capture_output=True,text=True).stdout
ax[2].axis("off"); ax[2].text(0,1,"(9) mutation testing\n"+txt,family="monospace",fontsize=6.3,va="top"); 
plt.tight_layout(); plt.savefig("model_tests.png",dpi=130)
