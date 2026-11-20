#!/bin/bash
# KinyaColBERT pipeline step 2: parse a text file (one item per line) with MorphoKIN.
#   bash scripts/kinyacolbert/parse_morphokin.sh <folder> <file.txt> [threads]
# Writes <folder>/parsed_<file.txt> in the format c4ir-rw/ac-ai-models reads, and checks it has
# exactly one line per input line, so it stays aligned with the matching *_id.txt file.
set -euo pipefail
cd "$(dirname "$0")/../.."
set -a; source configs/paths.env; [ -f configs/paths.local.env ] && source configs/paths.local.env; set +a
DIR=$1; IN=$2; THREADS=${3:-16}
apptainer exec --bind "$MORPHODATA_DIR:/MORPHODATA" \
  --bind "$MORPHOKIN_LICENSE:/MORPHODATA/licenses/morphokin_license.dat" \
  --bind "$DIR:/DATA" "$MORPHOKIN_SIF" \
  morphokin --morphokin_working_dir /MORPHODATA --morphokin_config_file /MORPHODATA/data/analysis_config_file.conf \
    --task PTF --num_threads "$THREADS" --kinlp_license /MORPHODATA/licenses/morphokin_license.dat \
    --ca_roots_pem_file /MORPHODATA/data/roots.pem \
    --input_file "/DATA/$IN" --output_file "/DATA/preparsed_$IN"
python3 "$DEEPKIN_AGAI/scripts/scratch/post_ptf.py" "$DIR/preparsed_$IN" "$DIR/parsed_$IN"   # restore line order
n_in=$(wc -l < "$DIR/$IN"); n_out=$(wc -l < "$DIR/parsed_$IN")
echo "lines: input $n_in, parsed $n_out"
[ "$n_in" -eq "$n_out" ] || { echo "MISMATCH: parsed file is not aligned with its input"; exit 1; }
sha256sum "$DIR/parsed_$IN" | cut -c1-16 > "$DIR/parsed_$IN.sha256"
echo "done: $DIR/parsed_$IN (sha256 $(cat "$DIR/parsed_$IN.sha256"))"
