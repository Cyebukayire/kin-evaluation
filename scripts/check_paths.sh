#!/bin/bash
# Reports every external path from configs/paths.env as OK or MISSING.
cd "$(dirname "$0")/.."
set -a; source configs/paths.env; [ -f configs/paths.local.env ] && source configs/paths.local.env; set +a
for v in HF_HOME WIKI_DIR KINNEWS_TRAIN CORPUS_DIR AFRIMMLU_DIR KCB_WORK_DIR DEEPKIN_AGAI KINYACOLBERT_CKPT KINYACOLBERT_AG_DATA \
         MORPHOKIN_IMAGE MORPHODATA_DIR MORPHOKIN_LICENSE MORPHOKIN_SIF; do
  p=${!v}
  if [ -e "$p" ]; then s="OK     "; else s="MISSING"; fi
  printf '%s %-22s %s\n' "$s" "$v" "$p"
done
if [ -d "$DEEPKIN_AGAI/.git" ]; then
  echo "DeepKIN-AgAI commit: $(git -C "$DEEPKIN_AGAI" rev-parse HEAD)"
fi
