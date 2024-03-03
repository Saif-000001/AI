#!/bin/bash

DATASET_FOLDER="dataset_folder"

find "$DATASET_FOLDER" -type f -name "*.wav" | while read -r FILENAME; do
    DIR_PATH=$(dirname "$FILENAME")
    BASE_NAME=$(basename "$FILENAME" .wav)
    SEGMENT_DURATION=3
    TOTAL_DURATION=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$FILENAME")
    NUM_SEGMENTS=$(echo "($TOTAL_DURATION + $SEGMENT_DURATION - 1)/$SEGMENT_DURATION" | bc)
    for ((i=0; i<NUM_SEGMENTS; i++)); do
        START_TIME=$(echo "$i * $SEGMENT_DURATION" | bc)
        OUTPUT_FILE="${DIR_PATH}/${BASE_NAME}segment${i}.wav"
        ffmpeg -y -loglevel error -i "$FILENAME" -ss "$START_TIME" -t "$SEGMENT_DURATION" -c copy "$OUTPUT_FILE"
    done
done

