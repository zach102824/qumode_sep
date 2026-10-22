/* WalkSATlm (Cai, Luo, Su 2014) pick rule as in Cai's WalkSATlm2013 pick_bbreak_gmake,
   parameterised (p=wp, w1=a, w2=b) as in Boulebnane & Montanaro.  Also: instance generator
   for CNF(n,k,r) (Poisson m, literals uniform with replacement), brute-force SAT filter.
   Modes:
     gen  n k r seed count           -> prints satisfiable instances (one per line: m then lits)
     run  maxflips restart seed      <- reads instances on stdin, prints: tries flips evals
   restart=1: fresh uniform random assignment at start of every try (paper protocol, max_flips=n)
   restart=0: continue the walk across tries (Cai's main loop). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
static uint64_t s[2];
static inline uint64_t rnd(void){uint64_t s1=s[0];const uint64_t s0=s[1];s[0]=s0;s1^=s1<<23;s[1]=s1^s0^(s1>>17)^(s0>>26);return s[1]+s0;}
static inline double urand(void){return (rnd()>>11)*(1.0/9007199254740992.0);}
static void seedr(uint64_t x){s[0]=x*0x9E3779B97F4A7C15ULL+1;s[1]=(x^0xD1B54A32D192ED03ULL)*0xBF58476D1CE4E5B9ULL+7;for(int i=0;i<20;i++)rnd();}
#define K 8
#define MAXM 8000
#define MAXN 32
int n,m,k; int lit[MAXM][K]; /* lit = +/-(v+1) */
int poisson(double lam){ /* normal approx not used: exact via sum of exps */ double L=0;int c=0; while(1){L+= -log(1-urand()); if(L>lam) return c; c++;}}
int brute_sat(void){
  uint32_t neg[MAXM],pos[MAXM]; for(int c=0;c<m;c++){neg[c]=pos[c]=0;for(int j=0;j<k;j++){int l=lit[c][j];if(l>0)pos[c]|=1u<<(l-1);else neg[c]|=1u<<(-l-1);}}
  for(uint32_t x=0;x<(1u<<n);x++){int ok=1;for(int c=0;c<m;c++){ if(!((x&pos[c])|(~x&neg[c]))){ok=0;break;} } if(ok) return 1;}
  return 0;}
/* local search state */
int sol[MAXN], satc[MAXM], occn[2*MAXN], *occ[2*MAXN]; int unsat[MAXM],upos[MAXM],nu; long long evals;
#define LI(l) ((l)>0? 2*((l)-1): 2*(-(l)-1)+1)
static inline int littrue(int l){return l>0? sol[l-1]: !sol[-l-1];}
void build_occ(void){ static int buf[MAXM*K]; int *p=buf; for(int i=0;i<2*n;i++)occn[i]=0;
  for(int c=0;c<m;c++)for(int j=0;j<k;j++)occn[LI(lit[c][j])]++;
  for(int i=0;i<2*n;i++){occ[i]=p;p+=occn[i];occn[i]=0;}
  for(int c=0;c<m;c++)for(int j=0;j<k;j++){int i=LI(lit[c][j]);occ[i][occn[i]++]=c;}}
void init_assign(void){ for(int v=0;v<n;v++)sol[v]=rnd()&1; nu=0;
  for(int c=0;c<m;c++){satc[c]=0;for(int j=0;j<k;j++)satc[c]+=littrue(lit[c][j]); if(!satc[c]){upos[c]=nu;unsat[nu++]=c;}} evals++; }
static inline void rm_unsat(int c){int last=unsat[--nu];unsat[upos[c]]=last;upos[last]=upos[c];}
static inline void add_unsat(int c){upos[c]=nu;unsat[nu++]=c;}
void flip(int v){ int tl= sol[v]? 2*v:2*v+1; int fl=tl^1; sol[v]^=1;
  for(int i=0;i<occn[tl];i++){int c=occ[tl][i]; if(--satc[c]==0) add_unsat(c);}
  for(int i=0;i<occn[fl];i++){int c=occ[fl][i]; if(satc[c]++==0) rm_unsat(c);} }
double wp; int A,B;
int brk(int v){int tl= sol[v]?2*v:2*v+1;int b=0;for(int i=0;i<occn[tl];i++)if(satc[occ[tl][i]]==1)b++;return b;}
int gmk(int v){int fl= sol[v]?2*v+1:2*v;int g=0;for(int i=0;i<occn[fl];i++){int s2=satc[occ[fl][i]];if(s2==0)g+=A;else if(s2==1)g+=B;}return g;}
int pick(void){ int c=unsat[rnd()%nu]; int best[K],nb=0,bb=1<<30;
  for(int j=0;j<k;j++){int v=abs(lit[c][j])-1; int b=brk(v); if(b<bb){bb=b;nb=0;best[nb++]=v;} else if(b==bb) best[nb++]=v;}
  if(bb!=0 && urand()<wp) return abs(lit[c][rnd()%k])-1;
  if(nb>1){int bg=-1,nb2=0; int b2[K]; for(int i=0;i<nb;i++){int g=gmk(best[i]); if(g>bg){bg=g;nb2=0;b2[nb2++]=best[i];} else if(g==bg) b2[nb2++]=best[i];} return b2[rnd()%nb2];}
  return best[rnd()%nb]; }
int readinst(void){ if(scanf("%d %d %d",&n,&k,&m)!=3) return 0; for(int c=0;c<m;c++)for(int j=0;j<k;j++)scanf("%d",&lit[c][j]); return 1;}
int main(int argc,char**argv){
  if(!strcmp(argv[1],"gen")){ n=atoi(argv[2]);k=atoi(argv[3]);double r=atof(argv[4]);seedr(atoll(argv[5]));int want=atoi(argv[6]); int got=0,tried=0;
    while(got<want){ m=poisson(r*n); tried++; for(int c=0;c<m;c++)for(int j=0;j<k;j++){int v=rnd()%n;lit[c][j]=(rnd()&1)?(v+1):-(v+1);}
      if(brute_sat()){printf("%d %d %d",n,k,m);for(int c=0;c<m;c++)for(int j=0;j<k;j++)printf(" %d",lit[c][j]);printf("\n");got++;}}
    fprintf(stderr,"n=%d sat %d/%d\n",n,got,tried); return 0;}
  /* run */
  long long maxflips=atoll(argv[2]); int restart=atoi(argv[3]); seedr(atoll(argv[4])); wp=atof(argv[5]); A=atoi(argv[6]); B=atoi(argv[7]);
  while(readinst()){ build_occ(); evals=0; long long tries=0,flips=0; init_assign();
    while(1){ tries++; if(restart && tries>1) init_assign(); long long st;
      for(st=0; st<maxflips && nu>0; st++){ flip(pick()); flips++; evals++; }
      if(nu==0) break; }
    printf("%lld %lld %lld\n",tries,flips,evals); fflush(stdout);}
  return 0;}
