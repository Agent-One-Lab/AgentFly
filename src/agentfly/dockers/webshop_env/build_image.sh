#!/bin/bash
# Build the two WebShop simulator images.
#
# Prereqs: stage the catalog data into the build context (gdown is dead for the
# canonical Drive IDs — copy from an existing image/checkout instead):
#   data/full/items_shuffle.json      data/full/items_ins_v2.json      (full 1.18M)
#   data/small/items_shuffle_1000.json data/small/items_ins_v2_1000.json (1,000 subset)
#   data/items_human_ins.json
#
# The Lucene index is built at image-build time (build_index.py + pyserini) and baked
# in — sized to match the catalog (full -> ~1.18M docs, small -> ~1,000). No runtime rebuild.
set -e

NS="${NS:-reasonwang}"

echo "Building webshop-env:full (1.18M catalog, human goals)..."
echo "data/small" > .dockerignore
docker build --build-arg DATASET=full  --build-arg HUMAN_GOALS=1 -t "${NS}/webshop-env:full"  .

echo "Building webshop-env:small (1,000 catalog, synthetic goals; aligned to verl-agent)..."
echo "data/full" > .dockerignore
docker build --build-arg DATASET=small --build-arg HUMAN_GOALS=0 -t "${NS}/webshop-env:small" .

rm -f .dockerignore
echo "Build completed. Push with: docker push ${NS}/webshop-env:full && docker push ${NS}/webshop-env:small"
