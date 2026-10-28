// Classical baselines for random 4-SAT: WalkSAT (SKC) and single-flip Metropolis SA.
// stdin: n m R mode param...  then m lines "v0 v1 v2 v3 s0 s1 s2 s3" (s=+1 means literal x_v, -1 means !x_v)
// mode 0 = walksat p ; mode 1 = SA b0 b1 steps_per_run
// output per run: flips evals   (one line per run). Target = any satisfying assignment (E==0).
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
static uint64_t S[2];
static inline uint64_t rnd(void){uint64_t s1=S[0];const uint64_t s0=S[1];S[0]=s0;s1^=s1<<23;S[1]=s1^s0^(s1>>17)^(s0>>26);return S[1]+s0;}
static inline double U(void){return (rnd()>>11)*(1.0/9007199254740992.0);}
static inline int RI(int k){return (int)(U()*k);}
int n,m,V[4000][4],SG[4000][4],tc[4000],x[64],occ[64][4000],no[64],unsat[4000],upos[4000],nu;
static void addu(int c){upos[c]=nu;unsat[nu++]=c;}
static void delu(int c){int p=upos[c];int l=unsat[--nu];unsat[p]=l;upos[l]=p;}
static inline int littrue(int c,int j){return (SG[c][j]>0)==(x[V[c][j]]==1);}
static void init(void){nu=0;for(int i=0;i<n;i++)x[i]=rnd()&1;for(int c=0;c<m;c++){tc[c]=0;for(int j=0;j<4;j++)tc[c]+=littrue(c,j);if(!tc[c])addu(c);}}
static void flip(int v){x[v]^=1;for(int k=0;k<no[v];k++){int c=occ[v][k];int j=0;while(V[c][j]!=v)j++;
  if(littrue(c,j)){if(tc[c]++==0)delu(c);}else{if(--tc[c]==0)addu(c);}}}
static int brk(int v){int b=0;for(int k=0;k<no[v];k++){int c=occ[v][k];if(tc[c]==1){int j=0;while(V[c][j]!=v)j++;if(littrue(c,j))b++;}}return b;}
static int mk(int v){int b=0;for(int k=0;k<no[v];k++){int c=occ[v][k];if(tc[c]==0)b++;}return b;}
int main(int argc,char**argv){int R,mode;double a,b,st;
  if(scanf("%d %d %d %d %lf %lf %lf %llu",&n,&m,&R,&mode,&a,&b,&st,(unsigned long long*)&S[0])!=8)return 1;S[1]=S[0]*0x9E3779B97F4A7C15ULL+12345;for(int i=0;i<20;i++)rnd();
  for(int c=0;c<m;c++){for(int j=0;j<4;j++)scanf("%d",&V[c][j]);for(int j=0;j<4;j++)scanf("%d",&SG[c][j]);for(int j=0;j<4;j++){int v=V[c][j];occ[v][no[v]++]=c;}}
  long long cap=200000000LL;
  for(int r=0;r<R;r++){long long fl=0,ev=0;
    if(mode==0){ init(); ev=1;
      while(nu>0&&fl<cap){int c=unsat[RI(nu)];int bb[4],best=1<<30,nb=0,cand[4];
        for(int j=0;j<4;j++){bb[j]=brk(V[c][j]);ev++;}
        for(int j=0;j<4;j++){if(bb[j]<best){best=bb[j];nb=0;}if(bb[j]==best)cand[nb++]=j;}
        int j; if(best==0||U()>=a) j=cand[RI(nb)]; else j=RI(4);
        flip(V[c][j]);fl++;}
    } else { long long steps=(long long)st; int hit=0;
      while(!hit&&ev<cap){ init(); ev++; if(nu==0){hit=1;break;}
        for(long long t=0;t<steps;t++){double beta=a+(b-a)*(double)t/(double)(steps>1?steps-1:1);
          int v=RI(n);int d=brk(v)-mk(v);ev++;   // d = change in #unsat
          if(d<=0||U()<exp(-beta*d)){flip(v);fl++;if(nu==0){hit=1;break;}}}
      }
    }
    printf("%lld %lld\n",fl,ev);}
  return 0;}
