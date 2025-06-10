#!/usr/bin/env bash
make
mkdir -p output

find data -type f ! -name ".gitkeep" | while read -r filepath; do
  filename=$(basename "$filepath")           # e.g. tree.ascii.pgm
  base="${filename%%.*}"                     # e.g. tree
  output_filename="${base}.pgm"              # e.g. tree.pgm

  echo "Processing $filepath -> output/$output_filename"
  ./imageFacialDetectionNPP --input="$filepath" --output="output/$output_filename" >> output/output.txt
done