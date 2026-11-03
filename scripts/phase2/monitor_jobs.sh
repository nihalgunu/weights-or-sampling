#!/bin/bash
# Early-stop monitor for Phase 2 eval jobs. Exits (one notification) on the first anomaly or when
# the first job finishes. Usage: bash scripts/phase2/monitor_jobs.sh NAME [NAME...]
# Anomalies: driver FATAL/lost, Traceback in the mirrored stdout, non-zero equivalence diff,
# task-0 success below 60%, or a mirrored stdout that stops growing for 30 min after the first episode.
cd "$(dirname "$0")/../.."
STALL_MIN=30
while true; do
  for n in "$@"; do
    L="logs/babysit_g_p2_eval_libero_spatial_$n.log"; F="outputs/phase2/logs/pi05_sft_libero_spatial_$n.stdout"
    if grep -qE "FATAL|NOT terminating|runner not alive|instance lost" "$L" 2>/dev/null; then
      echo "ANOMALY[$n]: driver event"; grep -nE "FATAL|NOT terminating|runner not alive|instance lost" "$L" | tail -3; exit 0; fi
    if grep -qE "=== PHASE2 EVAL done|driver done" "$L" 2>/dev/null; then
      echo "DONE[$n] $(date +%H:%M:%S)"; grep -nE "summary:|terminat|driver done" "$L" | tail -4; exit 0; fi
    [ -f "$F" ] || continue
    if grep -qE "Traceback|Error:" "$F"; then echo "ANOMALY[$n]: error in stdout"; grep -nE -A3 "Traceback|Error:" "$F" | tail -12; exit 0; fi
    bad=$(grep -oE "equivalence so far: n=[0-9]+ max_abs_diff=[0-9.e+-]+" "$F" | grep -v "max_abs_diff=0.000e+00" | tail -1)
    [ -n "$bad" ] && { echo "ANOMALY[$n]: $bad"; exit 0; }
    t0=$(grep -oE "Successes: [0-9]+/[0-9]+" "$F" | head -1)
    if [ -n "$t0" ]; then a=${t0#Successes: }; s=${a%/*}; m=${a#*/}; if [ $((s * 10)) -lt $((m * 6)) ]; then echo "ANOMALY[$n]: task 0 only $a"; exit 0; fi; fi
    eps=$(grep -c "Success:" "$F")
    if [ "$eps" -gt 0 ]; then age=$(( ( $(date +%s) - $(stat -f %m "$F") ) / 60 )); if [ "$age" -gt "$STALL_MIN" ]; then echo "ANOMALY[$n]: stdout stale ${age} min after $eps episodes"; tail -3 "$F"; exit 0; fi; fi
  done
  sleep 120
done
