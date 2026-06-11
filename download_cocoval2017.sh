#!/usr/bin/env bash
set -e

mkdir -p data/coco2017
cd data/coco2017

echo "Downloading COCO 2017 val images..."
curl -L -C - -o val2017.zip http://images.cocodataset.org/zips/val2017.zip

echo "Downloading COCO 2017 train/val annotations..."
curl -L -C - -o annotations_trainval2017.zip http://images.cocodataset.org/annotations/annotations_trainval2017.zip

echo "Extracting val images..."
unzip -q val2017.zip

echo "Extracting annotations..."
unzip -q annotations_trainval2017.zip

echo "Removing zip files..."
rm -f val2017.zip annotations_trainval2017.zip

echo "Done."
echo "Images:      data/coco2017/val2017/"
echo "Annotations: data/coco2017/annotations/"