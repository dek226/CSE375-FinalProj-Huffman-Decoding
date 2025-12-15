// Usage: bin/encoder input output
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "../include/huffman.h"

void fatal(const char* str){
    fprintf(stderr, "%s\n", str);
    exit(EXIT_FAILURE);
}

int compare(const void *p, const void *q){
    struct Symbol *P = (struct Symbol *) p;
    struct Symbol *Q = (struct Symbol *) q;
    return P->num - Q->num;
}

static inline double now_us() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec / 1e3;
}

int main(int argc, char **argv){
    if(argc != 3){
        printf("Usage: bin/encoder input output\n");
        exit(EXIT_FAILURE);
    }

    FILE *input, *output;
    unsigned int *inputfile;
    unsigned int *outputfile;
    unsigned int *gap_array;

    unsigned int inputfilesize = 0;
    unsigned int outputfilesize = 0;
    unsigned long long outputfilesize_bits = 0;

    unsigned int gap_array_size = 0;
    unsigned int gap_array_elements_num = 0;

    unsigned int *num_of_symbols =
        (unsigned int *)malloc(sizeof(int) * MAX_CODE_NUM);

    struct Symbol symbols[MAX_CODE_NUM] = {};
    struct Codetable *codetable;

    input = fopen(argv[1], "rb");
    if (!input) fatal("Could not open input file");

    output = fopen(argv[2], "wb");
    if (!output) fatal("Could not open output file");

    // ------------------------------------------------------------
    // Read input file
    // ------------------------------------------------------------
    fseek(input, 0L, SEEK_END);
    inputfilesize = ftell(input);
    fseek(input, 0L, SEEK_SET);

    cudaMallocHost(&codetable,
                   sizeof(struct Codetable) * MAX_CODE_NUM);

    cudaMallocHost(&inputfile,
                   sizeof(int) * ((inputfilesize + 3) / 4));

    fread(inputfile, sizeof(char), inputfilesize, input);
    fsync(input->_fileno);

    // ------------------------------------------------------------
    // Device allocations
    // ------------------------------------------------------------
    unsigned int *d_inputfile;
    unsigned int *d_num_of_symbols;

    cudaMalloc((void **)&d_inputfile,
               sizeof(int) * ((inputfilesize + 3) / 4 + 1));

    cudaMalloc((void **)&d_num_of_symbols,
               sizeof(int) * MAX_CODE_NUM);

    // ============================================================
    // TOTAL ENCODE TIMER START (MATCHES DECODER)
    // ============================================================
    double total_start_us = now_us();

    // ------------------------------------------------------------
    // HtD copy
    // ------------------------------------------------------------
    cudaMemcpy(d_inputfile, inputfile,
               sizeof(int) * ((inputfilesize + 3) / 4),
               cudaMemcpyHostToDevice);

    // ------------------------------------------------------------
    // Histogram (GPU)
    // ------------------------------------------------------------
    histgram(d_num_of_symbols, d_inputfile, inputfilesize);

    cudaMemcpy(num_of_symbols, d_num_of_symbols,
               sizeof(int) * MAX_CODE_NUM,
               cudaMemcpyDeviceToHost);

    cudaFree(d_num_of_symbols);

    // ------------------------------------------------------------
    // Dictionary construction (CPU)
    // ------------------------------------------------------------
    int symbol_count = store_symbols(num_of_symbols, symbols);
    qsort(symbols, symbol_count, sizeof(struct Symbol), compare);
    boundary_PM(symbols, symbol_count, codetable);

    // ------------------------------------------------------------
    // Output size computation
    // ------------------------------------------------------------
    outputfilesize_bits =
        get_outputfilesize(symbols, symbol_count);

    gap_array_elements_num =
        (outputfilesize_bits + SEGMENTSIZE - 1) / SEGMENTSIZE;

    gap_array_size =
        gap_array_elements_num / GAP_ELEMENTS_NUM +
        ((gap_array_elements_num % GAP_ELEMENTS_NUM) != 0);

    outputfilesize =
        (outputfilesize_bits + MAX_BITS - 1) / MAX_BITS;

    cudaMallocHost(&outputfile,
                   sizeof(unsigned int) *
                   (outputfilesize + gap_array_size));

    gap_array = outputfile + outputfilesize;

    // ------------------------------------------------------------
    // Encode kernel timing (kernel-only)
    // ------------------------------------------------------------
    double kernel_start_us = now_us();

    encode(
        outputfile,
        outputfilesize,
        d_inputfile,
        inputfilesize,
        gap_array_elements_num,
        codetable
    );

    cudaDeviceSynchronize();

    double kernel_end_us = now_us();

    // ============================================================
    // TOTAL ENCODE TIMER END
    // ============================================================
    double total_end_us = now_us();

    // ------------------------------------------------------------
    // Timing results
    // ------------------------------------------------------------
    double kernel_us = kernel_end_us - kernel_start_us;
    double total_us  = total_end_us  - total_start_us;

    double total_s = total_us / 1e6;
    double throughput_MBps =
        (double)inputfilesize / (1024.0 * 1024.0) / total_s;

    // ------------------------------------------------------------
    // Print results (decoder-style)
    // ------------------------------------------------------------
    printf("Input file: %s\n", argv[1]);
    printf("Original size: %u bytes\n", inputfilesize);
    printf("Compressed size: %u bytes\n", outputfilesize * 4);
    printf("Encode kernel time: %.3f ms\n", kernel_us / 1000.0);
    printf("Total encode time: %.3f ms\n", total_us / 1000.0);
    printf("Throughput: %.2f MB/s\n", throughput_MBps);

    // ------------------------------------------------------------
    // Write output file
    // ------------------------------------------------------------
    size_t tmp_symbol_count = symbol_count;
    fwrite(&tmp_symbol_count, sizeof(size_t), 1, output);

    for(int i = symbol_count - 1; i >= 0; i--){
        unsigned char tmpsymbol = symbols[i].symbol;
        unsigned char tmplength = symbols[i].length;
        fwrite(&tmpsymbol, sizeof(tmpsymbol), 1, output);
        fwrite(&tmplength, sizeof(tmplength), 1, output);
    }

    fwrite(&inputfilesize, sizeof(inputfilesize), 1, output);
    fwrite(&outputfilesize, sizeof(outputfilesize), 1, output);
    fwrite(&gap_array_elements_num,
           sizeof(gap_array_elements_num), 1, output);

    fwrite(gap_array, sizeof(int), gap_array_size, output);
    fwrite(outputfile, sizeof(unsigned int), outputfilesize, output);

    fdatasync(output->_fileno);

    // ------------------------------------------------------------
    // Cleanup
    // ------------------------------------------------------------
    fclose(input);
    fclose(output);

    cudaFreeHost(inputfile);
    cudaFreeHost(outputfile);
    cudaFreeHost(codetable);
    cudaFree(d_inputfile);

    free(num_of_symbols);

    return 0;
}
