# CSE375-FinalProj-Huffman-Decoding

**magic server max threads: 16

Generate data:
Compile: g++ -std=c++17 -O2 generate.cpp -o generate
Run: ./generate (file size) (redundancy 0.0-1.0)

Sequential:
Compile: g++ -std=c++17 -O2 sequential.cpp -o sequential
Run: ./sequential

Parallel CPU:
Compile: g++ -std=c++17 -O2 -fopenmp parallel_cpu.cpp -o parallel_cpu
Run ./parallel_cpu

Parallel CPU w/ decompression:
Compile: g++ -std=c++17 -O2 -fopenmp parallel_cpu_decomp.cpp -o parallel_decomp_cpu
Run: ./parallel_decomp_cpu

Parallel GPU:
Compile: nvcc -o huffman_gpu huffman_parallel_gpu.cpp
Run: ./huffman_gpu
*Note: The environment variable `CUDA_VISIBLE_DEVICES=0,1,2,3` mentioned in your prompt is still important and should be set externally to restrict which of the server's available GPUs the program can "see" and use.*

grep -nP "[\x80-\xFF]" your_file.cpp
No output = no non-ASCII garbage