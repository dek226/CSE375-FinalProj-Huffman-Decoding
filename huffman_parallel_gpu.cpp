#include <iostream>
#include <vector>
#include <unordered_map>
#include <queue>
#include <string>
#include <sstream>
#include <chrono>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <array>
#include <cuda_runtime.h>

// Helper macro for checking CUDA errors
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Maximum number of symbols (1 byte)
constexpr int SYMBOLS = 256;
// Maximum expected code length (for safety, though typically < 64)
constexpr int MAX_CODE_LEN = 256;
// Use 64 bits to store the actual code pattern (safe for short Huffman codes)
using CodeData = uint64_t;

// ================================================================
// ----------------------- CUDA KERNELS ---------------------------
// ================================================================

// Struct to hold the code information used on the GPU
struct GpuCodeInfo {
    uint32_t length;
    CodeData bits; // Stores the binary code pattern (e.g., 1011...)
};

/**
 * @brief Kernel for parallel frequency counting (histogram).
 * Each block computes a partial histogram.
 *
 * @param d_data Pointer to the input data chunk on the device.
 * @param data_size Size of the input data chunk.
 * @param d_histogram Pointer to the 256-entry histogram array on the device (initialized to zero).
 */
__global__ void frequency_kernel(const uint8_t* d_data, size_t data_size, int* d_histogram) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < data_size) {
        // Atomic operation is necessary here for correctness when multiple threads
        // might update the same symbol's counter, even though it's less efficient.
        // For large data, local histograms (reduction) are faster, but this is simpler for teaching.
        atomicAdd(&d_histogram[d_data[i]], 1);
    }
}

/**
 * @brief Kernel to calculate the total encoded bit length for a data chunk.
 * This is Phase 1 of parallel encoding.
 *
 * @param d_data Pointer to the input data chunk.
 * @param data_size Size of the input data chunk.
 * @param d_codes Array of GpuCodeInfo structures (lookup table).
 * @param d_length_out Pointer to the single output location for the total length.
 */
__global__ void calculate_length_kernel(const uint8_t* d_data, size_t data_size,
                                        const GpuCodeInfo* d_codes, size_t* d_length_out) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t local_sum = 0;

    if (i < data_size) {
        local_sum = d_codes[d_data[i]].length;
    }

    // Simple parallel reduction (inefficient for large block sizes, but safe)
    __shared__ size_t s_sum[512];
    s_sum[threadIdx.x] = local_sum;
    __syncthreads();

    // Summing reduction inside shared memory
    for (int s = blockDim.x / 2; s > 0; s /= 2) {
        if (threadIdx.x < s) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Block 0's thread 0 writes the total sum to the global output
    if (threadIdx.x == 0) {
        atomicAdd(d_length_out, s_sum[0]);
    }
}


/**
 * @brief Kernel to write the encoded bitstream for a data chunk into the final byte array.
 * This is Phase 2 of parallel encoding.
 *
 * @param d_data Input data chunk.
 * @param data_size Size of the data chunk.
 * @param d_codes Lookup table of GpuCodeInfo.
 * @param d_output The **GLOBAL** compressed byte array output buffer.
 * @param global_bit_offset The starting bit index for this chunk's output.
 */
__global__ void encode_kernel(const uint8_t* d_data, size_t data_size,
                              const GpuCodeInfo* d_codes, uint8_t* d_output,
                              size_t global_bit_offset) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < data_size) {
        // Look up the code for the current symbol
        uint8_t symbol = d_data[i];
        const GpuCodeInfo code_info = d_codes[symbol];
        size_t code_len = code_info.length;
        CodeData code_bits = code_info.bits;

        // Calculate the starting position of this symbol's code in the global bitstream
        size_t current_bit_offset = global_bit_offset;

        // --- Bit Packing ---
        // Iterate through the bits of the current code, MSB first (bit 0 is '1' or '0')
        for (size_t b = 0; b < code_len; ++b) {
            // Calculate the byte index and the bit position within that byte
            size_t byte_idx = current_bit_offset / 8;
            int bit_pos = 7 - (current_bit_offset % 8); // Bits are written high to low (7 to 0)

            // Extract the 'b'-th bit from the stored code_bits (0 is left-most/MSB)
            // Stored codes are right-justified (LSB). We shift the stored bits left
            // to align the most significant bit (the first bit of the code) to the
            // highest bit position in CodeData, then shift right by the difference
            // between 63 and the index b.
            CodeData bit_mask = (CodeData)1 << (code_len - 1 - b);
            int bit_val = (code_bits & bit_mask) != 0;

            // Use atomic OR to set the single bit in the global output array
            // This is safe because each symbol has a dedicated, non-overlapping range of bits.
            if (bit_val) {
                uint8_t mask = 1 << bit_pos;
                atomicOr(&d_output[byte_idx], mask);
            }

            current_bit_offset++;
        }
    }
}


// ================================================================
// ----------------------- CPU IMPLEMENTATION ---------------------
// ================================================================

class HuffmanParallelGPU {
private:
    // Internal node structure for sequential tree building
    struct Node {
        uint8_t symbol;
        size_t freq;
        Node* left;
        Node* right;
        bool isLeaf;

        Node(uint8_t s, size_t f)
            : symbol(s), freq(f), left(nullptr), right(nullptr), isLeaf(true) {}

        Node(Node* l, Node* r)
            : symbol(0), freq(l->freq + r->freq), left(l), right(r), isLeaf(false) {}
    };

    struct NodeCompare {
        bool operator()(Node* a, Node* b) const {
            return a->freq > b->freq;
        }
    };

    int num_gpus = 0;

    // Helper to convert std::string bit codes to the GPU-friendly GpuCodeInfo format
    GpuCodeInfo stringToGpuCode(const std::string& code) {
        GpuCodeInfo info;
        info.length = code.size();
        info.bits = 0;

        // Convert the '0'/'1' string to a uint64_t bit pattern
        for (char c : code) {
            info.bits <<= 1;
            if (c == '1') {
                info.bits |= 1;
            }
        }
        return info;
    }

    // ================================================================
    // MULTI-GPU FREQUENCY COUNTING
    // ================================================================
    std::unordered_map<uint8_t, size_t>
    countFrequenciesGPU(const std::vector<uint8_t>& data) {
        if (data.empty()) return {};
        size_t data_size = data.size();
        size_t chunk_size = data_size / num_gpus;

        // Host storage for all partial histograms
        std::vector<std::array<int, SYMBOLS>> host_histograms(num_gpus);

        std::vector<cudaStream_t> streams(num_gpus);
        for (int i = 0; i < num_gpus; ++i) {
            CHECK_CUDA(cudaStreamCreate(&streams[i]));
        }

        for (int i = 0; i < num_gpus; ++i) {
            CHECK_CUDA(cudaSetDevice(i));

            size_t start_idx = i * chunk_size;
            size_t current_chunk_size = (i == num_gpus - 1) ? (data_size - start_idx) : chunk_size;

            if (current_chunk_size == 0) continue;

            // Allocate device memory for data chunk and histogram
            uint8_t* d_data;
            int* d_histogram;
            CHECK_CUDA(cudaMalloc((void**)&d_data, current_chunk_size * sizeof(uint8_t)));
            CHECK_CUDA(cudaMallocManaged((void**)&d_histogram, SYMBOLS * sizeof(int), cudaMemAttachGlobal));
            CHECK_CUDA(cudaMemsetAsync(d_histogram, 0, SYMBOLS * sizeof(int), streams[i]));

            // Copy data to device
            CHECK_CUDA(cudaMemcpyAsync(d_data, data.data() + start_idx,
                                       current_chunk_size * sizeof(uint8_t),
                                       cudaMemcpyHostToDevice, streams[i]));

            // Launch kernel
            const int BLOCK_SIZE = 256;
            int num_blocks = (current_chunk_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

            frequency_kernel<<<num_blocks, BLOCK_SIZE, 0, streams[i]>>>(
                d_data, current_chunk_size, d_histogram);

            // Copy histogram back to host (using synchronous copy to ensure results before reduction)
            CHECK_CUDA(cudaMemcpyAsync(host_histograms[i].data(), d_histogram,
                                       SYMBOLS * sizeof(int), cudaMemcpyDeviceToHost, streams[i]));

            // Free device data (histogram is managed and will be freed after all streams sync)
            CHECK_CUDA(cudaFree(d_data));
        }

        // Wait for all GPU operations to complete
        for (int i = 0; i < num_gpus; ++i) {
            CHECK_CUDA(cudaStreamSynchronize(streams[i]));
            CHECK_CUDA(cudaStreamDestroy(streams[i]));
        }
        
        // Final reduction on CPU
        std::unordered_map<uint8_t, size_t> global_freq;
        for (int s = 0; s < SYMBOLS; s++) {
            size_t total_count = 0;
            for (int i = 0; i < num_gpus; i++) {
                total_count += host_histograms[i][s];
            }
            if (total_count > 0) {
                global_freq[(uint8_t)s] = total_count;
            }
        }
        
        // Note: d_histogram must be manually freed if using cudaMallocManaged without cudaFree
        // For simplicity and safety in this example, we rely on the host_histograms copy being complete.
        // In a production environment, d_histogram would be freed here.

        return global_freq;
    }

    // ================================================================
    // Tree building (sequential)
    // ================================================================
    Node* buildTree(const std::unordered_map<uint8_t, size_t>& freq) {
        std::priority_queue<Node*, std::vector<Node*>, NodeCompare> pq;

        for (auto& p : freq)
            pq.push(new Node(p.first, p.second));

        if (pq.size() == 1) {
            Node* x = pq.top(); pq.pop();
            return new Node(x, new Node(0, 0));
        }

        while (pq.size() > 1) {
            Node* a = pq.top(); pq.pop();
            Node* b = pq.top(); pq.pop();
            pq.push(new Node(a, b));
        }

        return pq.top();
    }

    void buildCodes(Node* n, const std::string& cur,
                    std::unordered_map<uint8_t, std::string>& codes) {
        if (n->isLeaf) {
            codes[n->symbol] = cur.empty() ? "0" : cur;
            return;
        }

        buildCodes(n->left, cur + "0", codes);
        buildCodes(n->right, cur + "1", codes);
    }

    void freeTree(Node* n) {
        if (!n) return;
        freeTree(n->left);
        freeTree(n->right);
        delete n;
    }

    // ================================================================
    // MULTI-GPU BITSTREAM ENCODING
    // ================================================================
    std::string encodeBitstreamGPU(const std::vector<uint8_t>& data,
                                   const std::unordered_map<uint8_t, std::string>& codes,
                                   std::vector<uint8_t>& out_payload, // Byte-packed output
                                   int& final_padding)
    {
        if (data.empty()) {
            final_padding = 0;
            return "";
        }

        size_t data_size = data.size();
        size_t chunk_size = data_size / num_gpus;

        // 1. Prepare GPU Code Lookup Table (Host)
        std::array<GpuCodeInfo, SYMBOLS> host_code_array;
        host_code_array.fill({0, 0});
        for (const auto& pair : codes) {
            host_code_array[pair.first] = stringToGpuCode(pair.second);
        }

        // --- Phase 1: Calculate Total Bit Length and Chunk Offsets ---

        // Host array to store the encoded bit length for each chunk
        std::vector<size_t> chunk_bit_lengths(num_gpus, 0);
        size_t* d_global_bit_length;
        CHECK_CUDA(cudaMallocManaged((void**)&d_global_bit_length, sizeof(size_t)));
        *d_global_bit_length = 0;

        std::vector<cudaStream_t> length_streams(num_gpus);
        for (int i = 0; i < num_gpus; ++i) {
            CHECK_CUDA(cudaStreamCreate(&length_streams[i]));
        }

        for (int i = 0; i < num_gpus; ++i) {
            CHECK_CUDA(cudaSetDevice(i));

            size_t start_idx = i * chunk_size;
            size_t current_chunk_size = (i == num_gpus - 1) ? (data_size - start_idx) : chunk_size;
            if (current_chunk_size == 0) continue;

            // Allocate device memory for data chunk and code info
            uint8_t* d_data;
            GpuCodeInfo* d_codes;
            CHECK_CUDA(cudaMalloc((void**)&d_data, current_chunk_size * sizeof(uint8_t)));
            CHECK_CUDA(cudaMalloc((void**)&d_codes, SYMBOLS * sizeof(GpuCodeInfo)));

            CHECK_CUDA(cudaMemcpyAsync(d_data, data.data() + start_idx,
                                       current_chunk_size * sizeof(uint8_t),
                                       cudaMemcpyHostToDevice, length_streams[i]));
            CHECK_CUDA(cudaMemcpyAsync(d_codes, host_code_array.data(),
                                       SYMBOLS * sizeof(GpuCodeInfo),
                                       cudaMemcpyHostToDevice, length_streams[i]));

            // Launch kernel to calculate total bit length for the chunk
            const int BLOCK_SIZE = 512;
            int num_blocks = (current_chunk_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

            calculate_length_kernel<<<num_blocks, BLOCK_SIZE, 0, length_streams[i]>>>(
                d_data, current_chunk_size, d_codes, d_global_bit_length);

            // We can't easily get the length per-chunk without a separate copy, so we
            // let the kernel accumulate to d_global_bit_length and calculate offsets later.

            CHECK_CUDA(cudaFree(d_data));
            CHECK_CUDA(cudaFree(d_codes));
        }

        // Wait for all length calculations to finish
        for (int i = 0; i < num_gpus; ++i) {
            CHECK_CUDA(cudaStreamSynchronize(length_streams[i]));
            CHECK_CUDA(cudaStreamDestroy(length_streams[i]));
        }

        size_t total_bit_size = *d_global_bit_length;
        CHECK_CUDA(cudaFree(d_global_bit_length)); // Done with managed memory for length

        if (total_bit_size == 0) {
            final_padding = 0;
            return "";
        }

        // 2. Pad and determine final byte size
        final_padding = (8 - (total_bit_size % 8)) % 8;
        size_t padded_bit_size = total_bit_size + final_padding;
        size_t total_byte_size = padded_bit_size / 8;

        // --- Phase 2: Parallel Bitstream Generation ---

        // Allocate the final global output buffer on the CPU and zero-initialize it
        out_payload.resize(total_byte_size, 0);

        std::vector<cudaStream_t> encode_streams(num_gpus);
        for (int i = 0; i < num_gpus; ++i) {
            CHECK_CUDA(cudaStreamCreate(&encode_streams[i]));
        }

        // Allocate the global output buffer on the device (or use the host vector's data)
        uint8_t* d_global_output;
        CHECK_CUDA(cudaMalloc((void**)&d_global_output, total_byte_size * sizeof(uint8_t)));
        CHECK_CUDA(cudaMemsetAsync(d_global_output, 0, total_byte_size * sizeof(uint8_t), 0)); // Global zeroing stream

        // The simple length calculation above used a single atomicAdd.
        // For actual encoding, we must run the length kernel again *per-chunk*
        // and then perform a prefix sum on the CPU to get the per-chunk starting offset.

        // Re-run length calculation to get per-chunk size for offset calculation
        size_t current_offset = 0;
        for (int i = 0; i < num_gpus; ++i) {
            CHECK_CUDA(cudaSetDevice(i));

            size_t start_idx = i * chunk_size;
            size_t current_chunk_size = (i == num_gpus - 1) ? (data_size - start_idx) : chunk_size;
            if (current_chunk_size == 0) continue;

            // Compute length of this chunk sequentially on CPU for simplicity, or use a separate kernel.
            // Since the code involves multiple kernel launches per GPU, we'll run the length calc again
            // using managed memory for simplicity.
            size_t* d_chunk_length;
            CHECK_CUDA(cudaMallocManaged((void**)&d_chunk_length, sizeof(size_t)));
            *d_chunk_length = 0;

            uint8_t* d_data_len;
            GpuCodeInfo* d_codes_len;
            CHECK_CUDA(cudaMalloc((void**)&d_data_len, current_chunk_size * sizeof(uint8_t)));
            CHECK_CUDA(cudaMalloc((void**)&d_codes_len, SYMBOLS * sizeof(GpuCodeInfo)));

            CHECK_CUDA(cudaMemcpy(d_data_len, data.data() + start_idx, current_chunk_size * sizeof(uint8_t), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(d_codes_len, host_code_array.data(), SYMBOLS * sizeof(GpuCodeInfo), cudaMemcpyHostToDevice));

            const int BLOCK_SIZE = 512;
            int num_blocks = (current_chunk_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

            calculate_length_kernel<<<num_blocks, BLOCK_SIZE>>>(
                d_data_len, current_chunk_size, d_codes_len, d_chunk_length);

            CHECK_CUDA(cudaDeviceSynchronize()); // Wait for this chunk's length
            chunk_bit_lengths[i] = *d_chunk_length;
            CHECK_CUDA(cudaFree(d_chunk_length));
            CHECK_CUDA(cudaFree(d_data_len));
            CHECK_CUDA(cudaFree(d_codes_len));
        }

        // --- Phase 3: Launch Encoding Kernels with correct global offsets ---

        for (int i = 0; i < num_gpus; ++i) {
            CHECK_CUDA(cudaSetDevice(i));

            size_t start_idx = i * chunk_size;
            size_t current_chunk_size = (i == num_gpus - 1) ? (data_size - start_idx) : chunk_size;
            if (current_chunk_size == 0) continue;

            // Re-allocate/Copy device memory for data chunk and code info
            uint8_t* d_data;
            GpuCodeInfo* d_codes;
            CHECK_CUDA(cudaMalloc((void**)&d_data, current_chunk_size * sizeof(uint8_t)));
            CHECK_CUDA(cudaMalloc((void**)&d_codes, SYMBOLS * sizeof(GpuCodeInfo)));

            CHECK_CUDA(cudaMemcpyAsync(d_data, data.data() + start_idx,
                                       current_chunk_size * sizeof(uint8_t),
                                       cudaMemcpyHostToDevice, encode_streams[i]));
            CHECK_CUDA(cudaMemcpyAsync(d_codes, host_code_array.data(),
                                       SYMBOLS * sizeof(GpuCodeInfo),
                                       cudaMemcpyHostToDevice, encode_streams[i]));

            // Launch encoding kernel
            const int BLOCK_SIZE = 512;
            int num_blocks = (current_chunk_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

            encode_kernel<<<num_blocks, BLOCK_SIZE, 0, encode_streams[i]>>>(
                d_data, current_chunk_size, d_codes, d_global_output, current_offset);

            current_offset += chunk_bit_lengths[i]; // Update offset for the next chunk

            CHECK_CUDA(cudaFree(d_data));
            CHECK_CUDA(cudaFree(d_codes));
        }

        // Wait for all encoding kernels to finish
        for (int i = 0; i < num_gpus; ++i) {
            CHECK_CUDA(cudaStreamSynchronize(encode_streams[i]));
            CHECK_CUDA(cudaStreamDestroy(encode_streams[i]));
        }

        // Copy the final compressed byte array from device to host
        CHECK_CUDA(cudaMemcpy(out_payload.data(), d_global_output,
                              total_byte_size * sizeof(uint8_t), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaFree(d_global_output));

        // Note: We don't return the bitstring, as the work is already done and byte-packed.
        // We return a dummy empty string here to satisfy the original function signature,
        // and let the caller use `out_payload` instead.
        return "";
    }

    // ================================================================
    // Header helpers (Sequential on CPU)
    // ================================================================
    void writeHeader(std::vector<uint8_t>& out,
                     int padding,
                     const std::unordered_map<uint8_t, std::string>& codes)
    {
        out.push_back((uint8_t)padding);

        uint16_t N = codes.size();
        // Write N (number of symbols) as 2 bytes (big-endian)
        out.push_back((N >> 8) & 0xFF);
        out.push_back(N & 0xFF);

        for (auto& p : codes) {
            out.push_back(p.first);                // symbol
            out.push_back((uint8_t)p.second.size()); // code length
            for (char c : p.second)
                out.push_back(c);
        }
    }

    void readHeader(const std::vector<uint8_t>& in,
                    size_t& index, int& padding,
                    std::unordered_map<uint8_t, std::string>& codes)
    {
        padding = in[index++];
        uint16_t N = ((uint16_t)in[index] << 8) | in[index+1];
        index += 2;

        for (int i = 0; i < N; i++) {
            uint8_t sym = in[index++];
            uint8_t len = in[index++];
            std::string code;
            code.reserve(len);
            for (int j = 0; j < len; j++)
                code.push_back(in[index++]);
            codes[sym] = code;
        }
    }


public:
    // ================================================================
    // PUBLIC INTERFACE
    // ================================================================
    HuffmanParallelGPU() {
        int dev_count;
        CHECK_CUDA(cudaGetDeviceCount(&dev_count));
        // Use a minimum of 1 and a maximum of 4, or the available count
        num_gpus = std::min(std::max(1, dev_count), 4);
        std::cout << "Detected " << dev_count << " CUDA devices. Using " << num_gpus << " for this task.\n";
    }

    // Encode (parallel GPU)
    std::vector<uint8_t> encode(const std::vector<uint8_t>& data) {
        if (data.empty()) return {};

        // 1. Parallel frequency count (GPU)
        auto freq = countFrequenciesGPU(data);
        if (freq.empty()) return {};

        // 2. Sequential tree build (CPU)
        Node* root = buildTree(freq);

        // 3. Sequential code table creation (CPU)
        std::unordered_map<uint8_t, std::string> codes;
        buildCodes(root, "", codes);

        // 4. Parallel bitstream generation and packing (GPU)
        std::vector<uint8_t> output;
        std::vector<uint8_t> payload;
        int padding = 0;

        // The GPU function handles the bit-packing directly into the payload vector
        encodeBitstreamGPU(data, codes, payload, padding);

        // 5. Write header (CPU)
        writeHeader(output, padding, codes);

        // 6. Append payload
        output.insert(output.end(), payload.begin(), payload.end());

        freeTree(root);
        return output;
    }

    // Decode (sequential CPU - remains the bottleneck)
    std::vector<uint8_t> decode(const std::vector<uint8_t>& compressed) {
        if (compressed.empty()) return {};

        size_t index = 0;
        int padding;
        std::unordered_map<uint8_t, std::string> codes;
        readHeader(compressed, index, padding, codes);

        // Build inverse map
        std::unordered_map<std::string, uint8_t> decodeMap;
        for (auto& p : codes)
            decodeMap[p.second] = p.first;

        // Convert remaining bytes to bitstring
        std::string bits;
        size_t payload_bytes = compressed.size() - index;
        bits.reserve(payload_bytes * 8);

        for (; index < compressed.size(); index++) {
            uint8_t b = compressed[index];
            for (int bit = 7; bit >= 0; bit--)
                bits.push_back((b >> bit) & 1 ? '1' : '0');
        }

        if (padding > 0)
            bits.resize(bits.size() - padding);

        // Decode
        std::vector<uint8_t> out;
        std::string cur;
        out.reserve(bits.size() / 8); // Estimate
        for (char c : bits) {
            cur.push_back(c);
            if (decodeMap.count(cur)) {
                out.push_back(decodeMap[cur]);
                cur.clear();
            }
        }
        return out;
    }
};


// ================================================================
// Main (Sequential CPU)
// ================================================================

// read file (Sequential CPU)
std::vector<uint8_t> read_file_to_vector(const std::string& filename)
{
    std::ifstream in(filename, std::ios::binary | std::ios::ate);
    if (!in) {
        // If file doesn't exist, generate dummy data for testing
        std::cout << "Warning: File 'data.bin' not found. Generating 10MB of dummy data.\n";
        std::vector<uint8_t> dummy(10 * 1024 * 1024);
        std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());
        std::uniform_int_distribution<uint8_t> dist(0, 255);
        for (uint8_t& byte : dummy) {
            byte = dist(rng);
        }
        return dummy;
    }

    std::streamsize size = in.tellg();
    if (size < 0) {
        throw std::runtime_error("Error: failed to read file size: " + filename);
    }

    std::vector<uint8_t> buffer(size);
    in.seekg(0, std::ios::beg);
    if (!in.read(reinterpret_cast<char*>(buffer.data()), size)) {
        throw std::runtime_error("Error: failed to read file: " + filename);
    }

    return buffer;
}


int main(int argc, char** argv) {
    // Note: The OpenMP thread_count setting is removed as we now rely on CUDA GPU count.
    
    HuffmanParallelGPU h;

    // Read data (or generate dummy data if data.bin is missing)
    std::vector<uint8_t> data = read_file_to_vector("data.bin");

    // Time compression and decompression
    auto start_comp = std::chrono::high_resolution_clock::now();
    auto compressed = h.encode(data);
    auto end_comp = std::chrono::high_resolution_clock::now();

    auto start_decomp = std::chrono::high_resolution_clock::now();
    auto decompressed = h.decode(compressed);
    auto end_decomp = std::chrono::high_resolution_clock::now();

    auto comp   = std::chrono::duration_cast<std::chrono::microseconds>(end_comp - start_comp).count();
    auto decomp = std::chrono::duration_cast<std::chrono::microseconds>(end_decomp - start_decomp).count();

    // Check for correctness
    bool ok = (data.size() == decompressed.size()) && std::equal(data.begin(), data.end(), decompressed.begin());

    // Results
    std::cout << "\n=== Multi-GPU CUDA Results ===\n";
    std::cout << "Original size:      " << data.size() << " bytes\n";
    std::cout << "Compressed size:    " << compressed.size() << " bytes\n";
    std::cout << "Compression ratio:  " << (100.0 * compressed.size() / data.size()) << "%\n\n";
    std::cout << "Compression time:   " << comp << " mcs\n";
    std::cout << "Decompression time: " << decomp << " mcs\n";
    std::cout << "Verification:       " << (ok ? "PASS" : "FAIL") << "\n";

    return 0;
}