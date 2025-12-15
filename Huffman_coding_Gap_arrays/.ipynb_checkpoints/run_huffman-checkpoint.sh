#!/bin/bash
set -e  # Exit immediately if any command fails


# Paths (adjust if needed)
ENCODER_DIR="./encoder"
DECODER_DIR="./decoder"

# chmod +x run_huffman.sh
#./run_huffman.sh
INPUT_FILE="../../data250_75.bin"
COMPRESSED_FILE="../encoder/compressed.huff"
OUTPUT_FILE="restored_data.bin"

echo "=============================="
echo " Building Encoder"
echo "=============================="
cd "$ENCODER_DIR"
make clean
make

echo "=============================="
echo " Running Encoder"
echo "=============================="
./bin/encoder "$INPUT_FILE" "$COMPRESSED_FILE"
#.encoder/bin/encoder ../data01_75.bin compressed.huff

echo "=============================="
echo " Building Decoder"
echo "=============================="
cd "../$DECODER_DIR"
make clean
make

echo "=============================="
echo " Running Decoder"
echo "=============================="
./bin/decoder "$COMPRESSED_FILE" "$OUTPUT_FILE"
#.decoder/bin/decoder compressed.huff restored_data.bin
echo "=============================="
echo " Done"
echo "=============================="
