#!/usr/bin/env bash
# SPARC and the Local Volume Database tables are already in ./data; this downloads the
# Pantheon+ supernova data (about 33 MB). It re-downloads the other two only if they are missing.
# Run from the repository root:  bash get_data.sh
set -euo pipefail
mkdir -p data && cd data

# 1. SPARC rotation curves of 175 galaxies (Lelli, McGaugh & Schombert 2016), CC BY 4.0
if [ -z "$(find -L SPARC -name '*_rotmod.dat' 2>/dev/null | head -1)" ]; then
  mkdir -p SPARC
  curl -fsSL -o SPARC/Rotmod_LTG.zip https://astroweb.case.edu/SPARC/Rotmod_LTG.zip \
    || curl -fsSL -o SPARC/Rotmod_LTG.zip "https://zenodo.org/records/16284118/files/Rotmod_LTG.zip?download=1"
  unzip -q -o SPARC/Rotmod_LTG.zip -d SPARC/Rotmod_LTG
fi

# 2. Local Volume Database of nearby dwarf galaxies (Pace), pinned to the version used for the results
if [ ! -d local_volume_database ]; then
  git clone -q --filter=blob:none https://github.com/apace7/local_volume_database.git
  git -C local_volume_database checkout -q bb444a777fa957872645cd5c1deef2a0aeddf6bf
fi

# 3. Pantheon+ supernovae: only the distance table and its error (covariance) matrix
if [ ! -d pantheon_plus ]; then
  git clone -q --depth 1 --filter=blob:none --sparse https://github.com/PantheonPlusSH0ES/DataRelease.git pantheon_plus
  git -C pantheon_plus sparse-checkout set "Pantheon+_Data/4_DISTANCES_AND_COVAR"
fi
echo "Data ready in ./data"
