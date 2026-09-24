#!/usr/bin/env bash
# Stage trained weights into a solution tree, test it, and zip it for submission.
#
#     bash scripts/pack.sh 008a|011|012
#     DEMO=/path/to/demo bash scripts/pack.sh 008a     # also run inference on a demo folder
#
# DEMO must hold niftis/ and submission_format.csv, the layout the competition container mounts.
#
# Weights come from the training steps in recipes/<model>.sh:
#   sm0        008a: artifacts/ckpt/sm0                   (15)
#              011, 012: artifacts/ckpt/{bn_n25,s2b_n25}  (2 draws x 15, renamed d0/d1)
#   siamese    artifacts/ckpt/siamese                     (15)
#   slicenet   artifacts/ckpt/slice128                    (15)
#   med3d      artifacts/ckpt/med3d                       (15)
#   boosters   008a, 011: artifacts/sbr/models            (15 LightGBM)
#              012:       artifacts/ckpt/sbr_v2_feat      (15 LightGBM, 50 features)
#
# Output: dist/<model>_submission.zip. Refuses to emit a zip unless every gate passes and never
# overwrites an existing archive.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${1:?usage: bash scripts/pack.sh 008a|011|012}
case "$MODEL" in
    008a) TREE=solutions/1_008a; SBR=artifacts/sbr/models ;;
    011)  TREE=solutions/2_011;  SBR=artifacts/sbr/models ;;
    012)  TREE=solutions/3_012;  SBR=artifacts/ckpt/sbr_v2_feat ;;
    *) echo "unknown model '$MODEL' (expected 008a, 011 or 012)"; exit 1 ;;
esac
PY=${PY:-uv run python}
CK=artifacts/ckpt
OUT=dist/${MODEL}_submission.zip

say() { printf '\n=== %s ===\n' "$1"; }
need() {  # need <dir> <glob> <count>
    local n; n=$(ls -1 "$1"/$2 2>/dev/null | wc -l)
    printf '  %-28s %2d/%d\n' "$1" "$n" "$3"
    [ "$n" -eq "$3" ] || { echo "  !! incomplete - see recipes/$MODEL.sh"; exit 1; }
}

[ -e "$OUT" ] && { echo "!! $OUT exists - move it away first; archives are never overwritten"; exit 1; }

say "1. inventory ($MODEL)"
if [ "$MODEL" = 008a ]; then need "$CK/sm0" '*.pt' 15
else need "$CK/bn_n25" 'datnet_s*_f*.pt' 15; need "$CK/s2b_n25" 'datnet_s*_f*.pt' 15; fi
for d in siamese slice128 med3d; do need "$CK/$d" '*.pt' 15; done
need "$SBR" 'lgb_*.txt' 15

say "2. stage weights into $TREE/assets"
rm -rf "$TREE/assets/cnn" "$TREE/assets/models"
mkdir -p "$TREE/assets/models" "$TREE/assets/cnn"/{sm0,siamese,slicenet,med3d}
cp "$SBR"/lgb_*.txt "$TREE/assets/models/"
if [ "$MODEL" = 008a ]; then
    cp "$CK/sm0"/*.pt "$TREE/assets/cnn/sm0/"
else
    # Both draws ship. They use identical file names, so a plain copy would silently overwrite
    # half of them - rename to datnet_d{draw}_s{seed}_f{fold}.pt.
    draw=0
    for d in bn_n25 s2b_n25; do
        for f in "$CK/$d"/datnet_s*_f*.pt; do
            b=$(basename "$f" .pt); cp "$f" "$TREE/assets/cnn/sm0/datnet_d${draw}_${b#datnet_}.pt"
        done
        draw=$((draw + 1))
    done
    n=$(ls -1 "$TREE/assets/cnn/sm0"/*.pt | wc -l)
    [ "$n" -eq 30 ] || { echo "  !! expected 30 sm0 checkpoints, got $n"; exit 1; }
fi
cp "$CK/siamese"/*.pt  "$TREE/assets/cnn/siamese/"
cp "$CK/slice128"/*.pt "$TREE/assets/cnn/slicenet/"
cp "$CK/med3d"/*.pt    "$TREE/assets/cnn/med3d/"
dupes=$(find "$TREE/assets/cnn" -name '*.pt' -exec md5sum {} + | awk '{print $1}' | sort | uniq -d | wc -l)
[ "$dupes" -eq 0 ] || { echo "  !! $dupes checkpoints staged twice"; exit 1; }
echo "  $(find "$TREE/assets/cnn" -name '*.pt' | wc -l) CNN checkpoints, $(ls "$TREE/assets/models" | wc -l) boosters"

say "3. test suite"
$PY tests/run_tests.py | tail -1

if [ -n "${DEMO:-}" ]; then
    say "4. local inference on $DEMO"
    work=$(mktemp -d)
    (cd "$work" && DATPARK_DATA_DIR="$DEMO" $PY "$OLDPWD/$TREE/main.py")
    $PY - "$work/submission.csv" <<'EOF'
import sys, pandas as pd
s = pd.read_csv(sys.argv[1])
assert s.is_pathologic.notna().all() and s.is_pathologic.between(0, 1).all(), "bad predictions"
print(f"  rows={len(s)}  range [{s.is_pathologic.min():.4f}, {s.is_pathologic.max():.4f}]")
EOF
fi

say "5. zip"
mkdir -p dist
(cd "$TREE" && find . -name '__pycache__' -prune -o -type f -print | sort | zip -q -X "../../$OUT" -@)
ls -la "$OUT"
echo "  md5 $(md5sum "$OUT" | cut -c1-32)"
