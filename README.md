# CSE375-FinalProj-Huffman-Decoding

**magic server max threads: 16

Generate data:
Compile: g++ -std=c++17 -O2 generate.cpp -o generate
Run: ./generate (file size) (redundancy 0.0-1.0) 100000000

Sequential:
Compile: g++ -std=c++17 -O2 sequential.cpp -o sequential
Run: ./sequential

Parallel CPU:
Compile: g++ -std=c++17 -O2 -fopenmp parallel_cpu.cpp -o parallel_cpu
Run ./parallel_cpu

Parallel CPU w/ decompression - WORKS and TEST PASS FOR LOWER BIT FILES WHEN ONLY ONE THREAD USED, but once bit size is large then fails verification:
Compile: g++ -std=c++17 -O2 -fopenmp parallel_cpu_decomp.cpp -o parallel_decomp_cpu
Run: ./parallel_decomp_cpu

Parallel GPU:
Compile: nvcc -o huffman_gpu huffman_parallel_gpu.cpp
Run: ./huffman_gpu
*Note: The environment variable `CUDA_VISIBLE_DEVICES=0,1,2,3` mentioned in your prompt is still important and should be set externally to restrict which of the server's available GPUs the program can "see" and use.*

    
# ****TODO: FOCUS ON GETTING MULTIGPU TO WORK AND THEN MAKING OUR OWN IMPROVEMENTS TO IT 

Parallel GPU paper implemented - changed demo file WORKS AND PASSES ALL TESTS:
# Syntax: ./bin/demo <compute device index> <size of input in megabytes>
# Example: Running on the first GPU (ID 0) with a 100 MB data size.
# name input file data.bin for feeding our generated data in instead of using defualt generator
# Run in the gpuhd folder
make
./bin/demo 0 100

Parallel GPU paper implemented - changed demo file to MULTI-GPU BUT TESTS FAIL:
# Syntax: ./bin/demo <Number of GPUS to use> <size of input in megabytes>
# Run in the gpuhd folder
make
./bin/demo 0 100


GAP ARRAYS:
    Encoder- 
        if you can run the Makefile by GNU Make.
        You may modify variables defined in the file.
        Usage: ./bin/encoder inputfile encoded_file
        ./bin/encoder ../../data.bin compressed.huff
    Decoder- 
        You can generate Huffman decoder with Gap arrays, called by "bin/decoder", if you can run the Makefile by GNU Make.
        You may modify variables defined in the file.
        Usage: ./bin/decoder inputfile decoded_file
        ./bin/decoder ../encoder/compressed.huff restored_data.bin

grep -nP "[\x80-\xFF]" your_file.cpp
No output = no non-ASCII garbage