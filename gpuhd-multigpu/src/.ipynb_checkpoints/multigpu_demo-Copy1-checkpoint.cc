#include <iostream>
#include <algorithm>
#include <random>
#include <fstream> 
#include <limits>  
#include <string>  
#include <vector>  // Required for multi-GPU contexts
#include <numeric> // Required for std::iota

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

#include <llhuff.h>       // encoder
#include <cuhd.h>         // decoder

// subsequence size
#define SUBSEQ_SIZE 4

// threads per block
#define NUM_THREADS 128

// performance parameter, high = high performance, low = low memory consumption
#define DEVICE_PREF 9

// ** ADDED: Constant for the data file name **
#define INPUT_FILENAME "input_data.bin"

// Structure to hold resources for a single GPU context
struct GpuContext {
    int device_id;
    long int uncompressed_elements; // SYMBOL_TYPE count
    long int compressed_units;      // UNIT_TYPE count
    long int uncompressed_offset;   // Offset in SYMBOL_TYPE elements
    long int compressed_offset;     // Offset in UNIT_TYPE units
    // cudaStream_t stream; // REMOVED: Using default stream per device
    
    // CUHD/CUDA resources
    std::shared_ptr<cuhd::CUHDInputBuffer> in_buf;
    std::shared_ptr<cuhd::CUHDOutputBuffer> out_buf;
    std::shared_ptr<cuhd::CUHDGPUInputBuffer> gpu_in_buf;
    std::shared_ptr<cuhd::CUHDGPUOutputBuffer> gpu_out_buf;
    std::shared_ptr<cuhd::CUHDCodetable> dec_table; // Decoder table copy
    std::shared_ptr<cuhd::CUHDGPUCodetable> gpu_table;
    std::shared_ptr<cuhd::CUHDGPUDecoderMemory> gpu_decoder_memory;
};


int main(int argc, char** argv) {
    // name of the binary file
    const char* bin = argv[0];
    
    // ** MODIFIED ARGUMENT PARSING **
    // argv[1] is now the number of GPUs to use
    // argv[2] is the total input size in MB
    if (argc < 3) {
        std::cout << "USAGE: " << bin << " <number of GPUs> "
        << "<total size of input in megabytes>" << std::endl;
        return 1;
    }

    const int num_gpus = atoi(argv[1]);
    const long int cli_size_bytes = atoi(argv[2]) * 1024 * 1024;
    long int size = cli_size_bytes / sizeof(SYMBOL_TYPE); // Total element count (SYMBOL_TYPE)
    
    int max_devices;
    cudaGetDeviceCount(&max_devices);
    CUERR

    if(num_gpus <= 0 || num_gpus > max_devices || size < 1) {
        std::cerr << "Invalid parameters." << std::endl;
        std::cerr << "Requested GPUs: " << num_gpus << ", Available: " << max_devices << std::endl;
        std::cerr << "Input element size: " << size << " (must be >= 1)" << std::endl;
        return 1;
    }
    // ** END MODIFIED ARGUMENT PARSING **


    // vector for storing time measurements
    std::vector<std::pair<std::string, size_t>> timings;
    
    // generate random data for testing
    // 'buffer' holds the original, uncompressed input data for final comparison
    std::vector<SYMBOL_TYPE> buffer; 

    // ** START MODIFIED DATA LOADING/GENERATION BLOCK (Retains size safety check) **
    
    std::ifstream infile(INPUT_FILENAME, std::ios::binary | std::ios::ate);

    if (infile.is_open()) {
        // File exists: Read data and update size
        long int file_size_bytes = infile.tellg();
        infile.seekg(0, std::ios::beg);
        
        if (file_size_bytes % sizeof(SYMBOL_TYPE) != 0) {
            std::cerr << "Error: File size (" << file_size_bytes 
                      << " bytes) is not a multiple of SYMBOL_TYPE size (" 
                      << sizeof(SYMBOL_TYPE) << " bytes). The file is likely corrupt." << std::endl;
            return 1;
        }

        size = file_size_bytes / sizeof(SYMBOL_TYPE); // size is now the correct number of elements
        
        buffer.resize(size);
        infile.read(reinterpret_cast<char*>(buffer.data()), file_size_bytes);
        infile.close();
        std::cout << "Input source: Read " << (file_size_bytes / (1024.0 * 1024.0)) << " MB from file: " << INPUT_FILENAME << std::endl;

    } else {
        // File missing: Generate random data (original logic), and save
        buffer.resize(size);

        std::independent_bits_engine<
            std::linear_congruential_engine<
            std::uint_fast32_t, 16807, 0, 2147483647>, 
            sizeof(SYMBOL_TYPE) * 8, SYMBOL_TYPE> gen;
        std::binomial_distribution<> dist(
            std::numeric_limits<SYMBOL_TYPE>::max(), 0.5);

        TIMER_START(timings, "generating random data")
            std::generate(begin(buffer), end(buffer), [&](){return dist(gen);});
        TIMER_STOP

        // Save to file
        std::ofstream outfile(INPUT_FILENAME, std::ios::binary);
        if (outfile.is_open()) {
            outfile.write(reinterpret_cast<const char*>(buffer.data()), size * sizeof(SYMBOL_TYPE));
            outfile.close();
            std::cout << "Data saved to '" << INPUT_FILENAME << "'." << std::endl;
        } else {
            std::cerr << "Warning: Could not save generated data to '" << INPUT_FILENAME << "'." << std::endl;
        }
    }
    // ** END MODIFIED DATA LOADING/GENERATION BLOCK **

    std::shared_ptr<std::vector<llhuff::LLHuffmanEncoder::Symbol>> lengths;
    std::shared_ptr<llhuff::LLHuffmanEncoderTable> enc_table;
    std::shared_ptr<cuhd::CUHDCodetable> dec_table;
    
    // compress the random data using length-limited Huffman coding
    TIMER_START(timings, "generating coding tables")

        // determine optimal lengths for the codewords to be generated
        lengths = llhuff::LLHuffmanEncoder::get_symbol_lengths(
            buffer.data(), size);
    
        // generate encoder table
        enc_table = llhuff::LLHuffmanEncoder::get_encoder_table(lengths);
    
        // generate decoder table
        dec_table = llhuff::LLHuffmanEncoder::get_decoder_table(enc_table);
    TIMER_STOP
    
    // buffer for compressed data
    std::unique_ptr<UNIT_TYPE[]> compressed
        = std::make_unique<UNIT_TYPE[]>(enc_table->compressed_size);

    // compress
    TIMER_START(timings, "encoding")
        llhuff::LLHuffmanEncoder::encode_memory(compressed.get(),
            enc_table->compressed_size, buffer.data(), size, enc_table);
    TIMER_STOP
        
    // Total compressed size (in UNIT_TYPE units)
    long int total_compressed_units = enc_table->compressed_size;
    // Total uncompressed size (in SYMBOL_TYPE elements)
    long int total_uncompressed_elements = size;
    
    // ** FIX: New buffer to hold the combined decoded output from all GPUs **
    std::vector<SYMBOL_TYPE> output_data(total_uncompressed_elements); 






    //possibly need separate arrays for each thread to write results to
    //make output buffer size uncompresseed/num_gpus


    //compressed = compressed data buffer
    
    
    // =======================
    // CUHD MULTI-GPU DECODE
    // =======================
    
    // ---------- Phase 0: Symbol partition ----------
    std::vector<long int> partitions(num_gpus);
    
    std::vector<long int> offsets(num_gpus);    //handle offsets later
    
    long int base = total_uncompressed_elements / num_gpus;
    long int rem  = total_uncompressed_elements % num_gpus;


    
    long int temp;
    for (int i = 0; i < num_gpus; ++i) {
       

        
        temp = i * base + (i + 1) *  base;
        
        partitions[i] = i * base + std::min<long int>(i, rem);
        offsets[i]  = base + (i < rem ? 1 : 0);
    }

    // ---------- Phase 1: Decode ----------
    TIMER_START(timings, "Multi-GPU Decode")

    std::vector<std::shared_ptr<cuhd::CUHDGPUDecoderMemory>> gpu_mem(num_gpus);
    std::vector<std::shared_ptr<cuhd::CUHDGPUInputBuffer>>  gpu_in_buf(num_gpus);
    std::vector<std::shared_ptr<cuhd::CUHDGPUOutputBuffer>> gpu_out_buf(num_gpus);
    std::vector<std::shared_ptr<cuhd::CUHDOutputBuffer>>    out_buf(num_gpus);
    std::vector<std::shared_ptr<cuhd::CUHDGPUCodetable>>    gpu_table(num_gpus);

    
    
    for (int i = 0; i < num_gpus; ++i) {
        cudaSetDevice(i);
        CUERR

        
        // load buffer with partition                   
        auto in_buf = std::make_shared<cuhd::CUHDInputBuffer>(
            reinterpret_cast<std::uint8_t*>(compressed.get()), 
        
            i * total_compressed_units/ * sizeof(UNIT_TYPE));   //CHANGE

        
    
        gpu_in_buf[i] = std::make_shared<cuhd::CUHDGPUInputBuffer>(in_buf);
    
        // Output buffer (exact symbol slice)
        out_buf[i] = std::make_shared<cuhd::CUHDOutputBuffer>(sym_count[i]);
        gpu_out_buf[i] = std::make_shared<cuhd::CUHDGPUOutputBuffer>(out_buf[i]);
    
        gpu_table[i] = std::make_shared<cuhd::CUHDGPUCodetable>(dec_table);
    
        gpu_mem[i] = std::make_shared<cuhd::CUHDGPUDecoderMemory>(
            in_buf->get_compressed_size_units(),
            SUBSEQ_SIZE,
            NUM_THREADS);

        // Allocate
        gpu_in_buf[i]->allocate();
        gpu_out_buf[i]->allocate();
        gpu_table[i]->allocate();
        gpu_mem[i]->allocate();
    
        gpu_table[i]->cpy_host_to_device();
        gpu_in_buf[i]->cpy_host_to_device();




        
        //partition 

        
        // Decode only this symbol range
        cuhd::CUHDGPUDecoder::decode(
            gpu_in_buf[i],
            in_buf->get_compressed_size_units(),
            gpu_out_buf[i],
            sym_count[i],
            gpu_table[i],
            gpu_mem[i],
            MAX_CODEWORD_LENGTH,
            SUBSEQ_SIZE,
            NUM_THREADS);


        std::cout << "test\n";
        
        
        // Copy back
        gpu_out_buf[i]->cpy_device_to_host();


        std::cout << "test\n";
        
    
        // Place into final output
        SYMBOL_TYPE* dst = output_data.data() + sym_offset[i];
        SYMBOL_TYPE* src = out_buf[i]->get_decompressed_data().get();

        std::copy(src, src + sym_count[i], dst);  
    }
    
    TIMER_STOP
    
    // =======================
    // END MULTI-GPU BLOCK
    // =======================



    

    // print timings
    for(auto &i: timings) {
        std::cout << i.first << ".. " << i.second << "µs" << std::endl;
    }
    
    // compare decoded output (output_data) to original input (buffer)
    cuhd::CUHDUtil::equals(buffer.data(),
        output_data.data(), total_uncompressed_elements) ? std::cout << std::endl
            : std::cout << std::endl << "mismatch" << std::endl;
    
    return 0;
}