#!/bin/bash

if [ "$(basename "$PWD")" != "Stimuli" ]; then
    echo "Go to the Stimuli directory first" >&2
    exit 1
fi
cp -r talia_20each talia_20each_tizi; cd talia_20each_tizi

# adds the dash to all the files that are missing it
for f in *.jpg; do
    newname=$(echo "$f" | sed 's/\([a-zA-Z]\)\([0-9]\)/\1_\2/')
    mv "$f" "$newname"
done

# Loop over all files in current folder and creates a folder for each unique filename
for f in *; do
    # Skip if not a file
    [ -f "$f" ] || continue
    [[ "$f" == *.db ]] && continue
    # Extract everything before first underscore
    folder="${f%%_*}"

    # Create folder if it doesn't exist
    mkdir -p "$folder"

    # Move file into folder
    mv "$f" "$folder/"
done
