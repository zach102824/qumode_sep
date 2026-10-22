# name env flipsdelta p w1 w2
cat <<V | while read name env d p a b; do for n in 12 14 16 18 20; do echo "$name $env $d $p $a $b $n"; done; done | xargs -P6 -L1 sh -c 'env $1 ./wslm run $(($6+$2)) 1 $((99+$6)) $3 $4 $5 < inst/n$6.txt > var/$0_n$6.txt' 
base X=0 0 0.15 6 5
randtie TIE=1 0 0.15 6 5
noiseall NOISEALL=1 0 0.15 6 5
flipsm1 X=0 -1 0.15 6 5
flipsp1 X=0 1 0.15 6 5
w11 X=0 0 0.15 1 1
w65swap X=0 0 0.15 5 6
p0 X=0 0 0.0 6 5
p30 X=0 0 0.30 6 5
V
