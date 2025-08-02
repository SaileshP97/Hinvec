#!/bin/bash

languages=("bengali" "burmese" "english" "gujarati" "hindi" "marathi" "nepali" "punjabi" "tamil" "telugu" "urdu")

# Create base directory if not exists
base_dir="./Bitext Mining/CrossSum"
mkdir -p "$base_dir"

for slang in "${languages[@]}"; do
  for tlang in "${languages[@]}"; do

    filename="${slang}-${tlang}_CrossSum.tar.bz2"
    url="https://huggingface.co/datasets/csebuetnlp/CrossSum/resolve/main/data/${filename}?download=true"
    out_dir="${base_dir}/${slang}-${tlang}"

    echo "Downloading $filename..."
    wget -O "$filename" "$url"

    echo "Extracting $filename to $out_dir..."
    mkdir -p "$out_dir"
    tar -xjf "$filename" -C "$out_dir"

    echo "Deleting $filename..."
    rm "$filename"

  done
done

echo "All downloads and extractions completed."
