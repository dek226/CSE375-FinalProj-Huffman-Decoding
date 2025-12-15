#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "../include/huffman.h"

void fatal(const char* str){
    fprintf(stderr, "%s! at %s in %d\n", str, __FILE__, __LINE__);
    exit(EXIT_FAILURE);
}

static inline double now_us() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec / 1e3;
}

int main(int argc, char** argv){
    FILE *inputfile, *outputfile;
    size_t tmp_st;
    int symbol_count;
    struct Symbol *symbols;

    unsigned int* input;
    unsigned int* output;

    inputfile = fopen(argv[1], "rb");
    outputfile = fopen("decodedfile", "wb");

    if (!inputfile) fatal("Could not open input file");

    if (1 != fread(&tmp_st, sizeof(size_t), 1, inputfile))
        fatal("File read error 1");
    symbol_count = tmp_st;

    cudaMallocHost(&symbols, sizeof(struct Symbol)*symbol_count);

    for(int i=0; i<symbol_count; i++){
        unsigned char tmpsymbol;
        unsigned char tmplength;
        if (1 != fread(&tmpsymbol, sizeof(tmpsymbol), 1, inputfile))
            fatal("File read error 2");
        if (1 != fread(&tmplength, sizeof(tmplength), 1, inputfile))
            fatal("File read error 3");
        symbols[i].symbol = tmpsymbol;
        symbols[i].length = tmplength;
    }

    unsigned int prefix_bit = FIXED_PREFIX_BIT;
    struct TableInfo h_tableinfo;
    void* stage_table;
    int stage_table_size;

    for(int i=0; i<LOOP; i++){
        get_table_info(
            symbols,
            symbol_count,
            prefix_bit,
            h_tableinfo);
    }

    stage_table_size =
        sizeof(int)*h_tableinfo.ptrtable_size +
        sizeof(char)*(MAX_CODE_NUM +
        h_tableinfo.l1table_size +
        h_tableinfo.l2table_size);

    cudaMallocHost(&stage_table, stage_table_size);

    get_twolevel_table(
        stage_table,
        prefix_bit,
        symbols,
        symbol_count,
        h_tableinfo);

    int compressed_size, original_size, gaparray_size, gap_elements_num;

    if (1 != fread(&original_size, sizeof(int), 1, inputfile))
        fatal("File read error 4");
    if (1 != fread(&compressed_size, sizeof(int), 1, inputfile))
        fatal("File read error 5");
    if (1 != fread(&gap_elements_num, sizeof(int), 1, inputfile))
        fatal("File read error 6");

    gaparray_size = (gap_elements_num + GAP_FAC_NUM - 1) / GAP_FAC_NUM;

    cudaMallocHost(&input,
        sizeof(int)*(gaparray_size + compressed_size));
    cudaMallocHost(&output,
        sizeof(int)*((original_size+3)/4));

    if ((size_t)(gaparray_size + compressed_size) !=
        fread(input, sizeof(unsigned int),
              gaparray_size + compressed_size, inputfile))
        fatal("File read error 7");

    printf("Input file: %s\n", argv[1]);
    printf("Original size: %d bytes\n", original_size);
    printf("Compressed size: %d bytes\n", compressed_size);

    double decode_start_us = now_us();

    
    decoder_l1_l2(input,
        compressed_size,
        output,
        original_size,
        gap_elements_num,
        stage_table,
        stage_table_size,
        prefix_bit,
        symbol_count,
        h_tableinfo);

    cudaDeviceSynchronize();
    double decode_end_us = now_us();

    double decode_time_us = decode_end_us - decode_start_us;
    double decode_time_s  = decode_time_us / 1e6;
    double throughput_MBps =
        (double)original_size / (1024.0 * 1024.0) / decode_time_s;

    printf("Decode time: %.3f ms\n", decode_time_us / 1000.0);
    printf("Throughput: %.2f MB/s\n", throughput_MBps);

    fwrite(output, sizeof(char), original_size, outputfile);

    fclose(inputfile);
    fclose(outputfile);

    cudaFreeHost(symbols);
    cudaFreeHost(input);
    cudaFreeHost(output);
    cudaFreeHost(stage_table);

    return 0;
}
