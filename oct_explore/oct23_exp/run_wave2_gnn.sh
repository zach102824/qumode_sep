#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

SCRIPT="/workspace/qumode_sep/oct_explore/oct23_exp/gnn_5block.py"
LOGDIR="/workspace/qumode_sep/oct_explore/logs/oct23_exp"
RESDIR="/workspace/qumode_sep/oct_explore/results/oct23_exp"
mkdir -p "$LOGDIR" "$RESDIR"

COMMANDS=(
  "--tag gnn5_n6_1p4x1_bpdec_pcap --n 6 --sched 1p4x1 --order csd --relabel bpdec --clause_w --arch mp --H 128 --rounds 4 --ntrain 10000 --opt adam --steps 3000 --bs 250 --lr 0.0012 --p_cap 0.65 --tailw 2.5 --init_base $RESDIR/n6_1p4x1_msdec_sdir.json"
  "--tag gnn5_n6_2p3x1_bpdec_pcap --n 6 --sched 2p3x1 --order csd --relabel bpdec --clause_w --arch mp --H 128 --rounds 4 --ntrain 10000 --opt adam --steps 3000 --bs 250 --lr 0.0012 --p_cap 0.65 --tailw 2.5"
  "--tag gnn5_n8_2p3x1_bpdec_pcap --n 8 --sched 2p3x1 --order csd --relabel bpdec --clause_w --arch mp --H 128 --rounds 4 --ntrain 10000 --opt adam --steps 3000 --bs 250 --lr 0.0012 --p_cap 0.60 --tailw 2.5"
  "--tag gnn5_n8_0p5x1_sdc_bpdec_pcap --n 8 --sched 0p5x1 --order sdc --relabel bpdec --clause_w --arch mp --H 128 --rounds 4 --ntrain 10000 --opt adam --steps 3000 --bs 250 --lr 0.0012 --p_cap 0.60 --tailw 2.5 --init_base $RESDIR/n8_0p5x1_sdc_msdec_sdir.json"
  "--tag gnn5_n6_0p5x1_sdc_bpdec_pcap --n 6 --sched 0p5x1 --order sdc --relabel bpdec --clause_w --arch mp --H 128 --rounds 4 --ntrain 10000 --opt adam --steps 3000 --bs 250 --lr 0.0012 --p_cap 0.65 --tailw 2.5"
  "--tag gnn5_n8_1p4x1_bpdec_pcap --n 8 --sched 1p4x1 --order csd --relabel bpdec --clause_w --arch mp --H 128 --rounds 4 --ntrain 10000 --opt adam --steps 3000 --bs 250 --lr 0.0012 --p_cap 0.60 --tailw 2.5 --init_base $RESDIR/n8_1p4x1_msdec_sdir.json"
  "--tag gnn5_n6_1p4x1_none_pcap --n 6 --sched 1p4x1 --order csd --relabel none --clause_w --arch mp --H 128 --rounds 4 --ntrain 10000 --opt adam --steps 3000 --bs 250 --lr 0.0012 --p_cap 0.65 --tailw 2.0 --init_base $RESDIR/n6_ntrain10000_none.json"
  "--tag gnn5_n8_2p3x1_none_pcap --n 8 --sched 2p3x1 --order csd --relabel none --clause_w --arch mp --H 128 --rounds 4 --ntrain 10000 --opt adam --steps 3000 --bs 250 --lr 0.0012 --p_cap 0.60 --tailw 2.0"
  "--tag gnn5_n6_1p4x1_bpdec_nocw --n 6 --sched 1p4x1 --order csd --relabel bpdec --arch mp --H 128 --rounds 4 --ntrain 10000 --opt adam --steps 3000 --bs 250 --lr 0.0012 --p_cap 0.65 --tailw 2.5 --init_base $RESDIR/n6_1p4x1_msdec_sdir.json"
  "--tag gnn5_n8_2p3x1_bpdec_nocw --n 8 --sched 2p3x1 --order csd --relabel bpdec --arch mp --H 128 --rounds 4 --ntrain 10000 --opt adam --steps 3000 --bs 250 --lr 0.0012 --p_cap 0.60 --tailw 2.5"
)

run_one() {
  local args="$1"
  local tag=$(echo "$args" | awk '{for(i=1;i<=NF;i++) if($i=="--tag") print $(i+1)}')
  if [ -f "$RESDIR/${tag}.json" ]; then
    echo "Skipping $tag (already done)"
    return 0
  fi
  echo "Starting $tag ..."
  python3 "$SCRIPT" $args > "$LOGDIR/${tag}.log" 2>&1
  echo "Finished $tag"
}
export -f run_one
export SCRIPT LOGDIR RESDIR

printf "%s\n" "${COMMANDS[@]}" | xargs -I {} -P 3 bash -c 'run_one "$@"' _ {}
echo "ALL WAVE 2 GNN JOBS COMPLETE"
