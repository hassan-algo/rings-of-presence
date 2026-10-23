#!/usr/bin/env bash
# Run every test and save its printed output in ./results.
# Run from the repository root after get_data.sh:  bash run_all.sh
set -uo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
SPARC=$(dirname "$(find -L "$ROOT/data/SPARC" -name '*_rotmod.dat' 2>/dev/null | head -1)")
LVDB="$ROOT/data/local_volume_database"
PPLUS="$ROOT/data/pantheon_plus/Pantheon+_Data/4_DISTANCES_AND_COVAR"
mkdir -p "$ROOT/results"; cd "$ROOT/results"
run() {
  local name=$1; shift; local start=$(date +%s)
  printf '%-28s' "$name"
  if python3 "$ROOT/scripts/$name.py" "$@" > "$name.txt" 2>&1; then echo "ok ($(( $(date +%s) - start )) s)"
  else echo "FAILED (see results/$name.txt)"; fi
}
# Solar system, clocks and light
run presence_model_tests
run presence_round2
run presence_judge
# Binary pulsar: presence arriving late, full orbit equations
run delayed_presence_binary 60
run fixed_binary 1000
# Galaxies and a0
run fit_a0 "$SPARC"
run galaxy_outside_pull "$SPARC"
run outside_presence_tests "$SPARC"
run k_ring_test "$SPARC"
run k_derivation_test "$SPARC"
# Satellite galaxies and dark matter
run satellite_predictions "$LVDB"
run satellite_history "$LVDB"
run sand_fit "$LVDB"
run glass_tests "$LVDB"
run ultrafaint_centres "$LVDB"
# Rings within rings, and the expansion
run ring_tests "$LVDB"
run expansion_tests "$PPLUS"
