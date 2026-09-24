#!/usr/bin/env bash
# Download the MedicalNet ResNet-10 initialisation used by the med3d family (MIT licence,
# Tencent; Chen, Ma & Zheng 2019, arXiv:1904.00625). 57 MB, verified by md5.
#
#     bash scripts/fetch_medicalnet.sh
#
# Only the fine-tuned checkpoints ship inside a submission; this file is needed for training only
# and is never committed.
set -euo pipefail
cd "$(dirname "$0")/.."

DEST=artifacts/pretrained/resnet_10_23dataset.pth
MD5=0d85648adead11c130fb7a4fb5f96812
URL=https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet10/resolve/main/resnet_10_23dataset.pth

mkdir -p "$(dirname "$DEST")"
[ -f "$DEST" ] || curl -sSL -m 900 -o "$DEST" "$URL"
got=$(md5sum "$DEST" | cut -d' ' -f1)
[ "$got" = "$MD5" ] || { echo "!! md5 mismatch for $DEST: got $got want $MD5"; exit 1; }
echo "MedicalNet r10 OK ($DEST, md5 $got)"
