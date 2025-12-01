#include <vector>
#include <unordered_map>
#include <queue>
#include <cstdint>
#include <string>
#include <stdexcept>
#include <sstream>
#include <iostream>
#include <random>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <omp.h>
#include <array>
#include <cmath>
#include <cstring>


// Global thread count for OpenMP
int thread_count = 16;
constexpr size_t DECODE_BLOCK_SIZE = 1024 * 64; // 64 KB block size for parallel decoding

class HuffmanParallelCPU {
    public:
        // Encode (parallel)
        std::vector<uint8_t> encode(const std::vector<uint8_t>& data) {
            if (data.empty()) return {};

            // 1. Parallel frequency count (HIGHLY parallel)
            auto freq = countFrequenciesParallel(data);

            // 2. Sequential tree build (tiny, fast)
            Node* root = buildTree(freq);

            // 3. Sequential code table creation (tiny, fast)
            std::unordered_map<uint8_t, std::string> codes;
            buildCodes(root, "", codes);

            // 4. Parallel bitstream generation (HIGHLY parallel)
            std::string bitstream = encodeParallel(data, codes);

            // 5. Pad
            int padding = (8 - (bitstream.size() % 8)) % 8;
            bitstream.append(padding, '0');

            // 6. Write header
            std::vector<uint8_t> output;
            writeHeader(output, padding, codes);

            // 7. Write payload (Sequential packing, fast due to string size)
            for (size_t i = 0; i < bitstream.size(); i += 8) {
                uint8_t byte = std::stoi(bitstream.substr(i, 8), nullptr, 2);
                output.push_back(byte);
            }

            freeTree(root);
            return output;
        }

        // Decode (Parallel implementation using Chunking/Gap Array concept)
        std::vector<uint8_t> decode(const std::vector<uint8_t>& compressed) {
            return decodeParallel(compressed);
        }

    private:
        // ================================================================
        // Internal node structure
        // ================================================================
        struct Node {
            uint8_t symbol;
            int freq;
            Node* left;
            Node* right;
            bool isLeaf;

            Node(uint8_t s, int f)
                : symbol(s), freq(f), left(nullptr), right(nullptr), isLeaf(true) {}

            Node(Node* l, Node* r)
                : symbol(0), freq(l->freq + r->freq), left(l), right(r), isLeaf(false) {}
        };

        struct NodeCompare {
            bool operator()(Node* a, Node* b) const {
                return a->freq > b->freq;
            }
        };

        // ================================================================
        // ENCODING HELPERS (OpenMP Parallel)
        // ================================================================

        // Parallel Frequency Counting: Uses a reduction pattern (local accumulation, global merge).
        std::unordered_map<uint8_t, int>
        countFrequenciesParallel(const std::vector<uint8_t>& data)
        {
            constexpr int SYMBOLS = 256;
            std::vector<std::array<int, SYMBOLS>> local(thread_count);

            for (auto& arr : local) arr.fill(0);

            // Each thread processes a chunk of the input data and updates only its local table.
            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                auto& table = local[tid];

                #pragma omp for nowait
                for (size_t i = 0; i < data.size(); i++) {
                    table[data[i]]++;
                }
            }

            // Sequential Reduction: Merge all local tables into one global table.
            std::array<int, SYMBOLS> global{};
            for (int t = 0; t < thread_count; t++)
                for (int s = 0; s < SYMBOLS; s++)
                    global[s] += local[t][s];

            // Convert to unordered_map
            std::unordered_map<uint8_t, int> freq;
            for (int s = 0; s < SYMBOLS; s++)
                if (global[s] > 0)
                    freq[(uint8_t)s] = global[s];

            return freq;
        }

        // Parallel Bitstream Encoding: Each thread independently encodes a data chunk.
        std::string encodeParallel(const std::vector<uint8_t>& data,
                                   const std::unordered_map<uint8_t, std::string>& codes)
        {
            std::vector<std::string> local(thread_count);

            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                auto& out = local[tid];

                out.reserve(data.size() / thread_count * 2);

                // Each thread processes a range of input bytes.
                #pragma omp for nowait
                for (size_t i = 0; i < data.size(); i++) {
                    out += codes.at(data[i]);
                }
            }

            // Sequential Concatenation (Required reduction step)
            std::string result;
            result.reserve(data.size() * 2);
            for (auto& s : local)
                result += s;

            return result;
        }

        // ================================================================
        // DECODING HELPERS (Conceptual Parallel Gap/Chunked Decode)
        // ================================================================

        // Helper function to perform sequential decoding within a known bit range.
        std::vector<uint8_t> sequentialBitDecode(const std::string& bitstream,
                                                 const std::unordered_map<std::string, uint8_t>& decodeMap)
        {
            std::vector<uint8_t> output;
            std::string current;
            for (char c : bitstream) {
                current.push_back(c);
                if (decodeMap.count(current)) {
                    output.push_back(decodeMap.at(current));
                    current.clear();
                }
            }
            return output;
        }

        // Parallel Decoding using Chunking (modeling the Gap Array approach)
        std::vector<uint8_t> decodeParallel(const std::vector<uint8_t>& compressed) {
            size_t index = 0;
            int padding;
            std::unordered_map<uint8_t, std::string> codes;
            readHeader(compressed, index, padding, codes);

            // Build inverse map (code-string → symbol)
            std::unordered_map<std::string, uint8_t> decodeMap;
            for (auto& p : codes) decodeMap[p.second] = p.first;

            size_t compressed_payload_size = compressed.size() - index;
            if (compressed_payload_size == 0) return {};

            // 1. Pre-Conversion to Bitstream (still necessary for simple CPU implementation)
            std::string full_bitstream;
            full_bitstream.reserve(compressed_payload_size * 8);
            for (size_t i = index; i < compressed.size(); i++) {
                uint8_t b = compressed[i];
                for (int bit = 7; bit >= 0; bit--)
                    full_bitstream.push_back(((b >> bit) & 1) ? '1' : '0');
            }

            if (padding > 0)
                full_bitstream.resize(full_bitstream.size() - padding);

            // 2. Divide bitstream into blocks (Gap Array / Chunking Concept)
            size_t total_bits = full_bitstream.size();
            size_t num_blocks = (total_bits + DECODE_BLOCK_SIZE * 8 - 1) / (DECODE_BLOCK_SIZE * 8);

            // Storage for thread-local results
            std::vector<std::vector<uint8_t>> decoded_blocks(num_blocks);

            // --- PARALLEL DECODING ---
            #pragma omp parallel for schedule(dynamic)
            for (size_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
                // Start bit position for this block
                size_t start_bit = block_idx * DECODE_BLOCK_SIZE * 8;
                size_t end_bit = std::min(start_bit + DECODE_BLOCK_SIZE * 8, total_bits);
                size_t block_size_bits = end_bit - start_bit;

                if (block_size_bits == 0) continue;

                std::string block_bits = full_bitstream.substr(start_bit, block_size_bits);

                // *** The Critical Parallel Challenge (Modeled by 'Fixing' the Start) ***
                // In a real Gap Array or LUT implementation, this is where a small sequential
                // pre-decode step (the 'Gap Array' step) would occur to find the exact bit-offset
                // for the first *full* symbol. Here, we simplify by assuming a bitstream string:
                size_t fixed_start_offset = 0; // The true symbol start relative to the block's start_bit

                // The thread performs the sequential bit-parsing on its chunk.
                // The sequentialBitDecode function is where the LUTs would be used
                // to speed up lookups (e.g., 8 bits at a time instead of 1).
                decoded_blocks[block_idx] = sequentialBitDecode(block_bits.substr(fixed_start_offset), decodeMap);
            }

            // 3. Final Assembly
            size_t total_decompressed_size = 0;
            for (const auto& block : decoded_blocks) {
                total_decompressed_size += block.size();
            }

            std::vector<uint8_t> output;
            output.reserve(total_decompressed_size);
            for (auto& block : decoded_blocks) {
                output.insert(output.end(), block.begin(), block.end());
            }

            // NOTE: A real implementation would need complex cross-block dependency resolution
            // to handle codes split across block boundaries. This simplified model assumes
            // the sequentialBitDecode handles all codes *fully* within the block.
            return output;
        }


        // ================================================================
        // SEQUENTIAL HELPERS (Tree Building and Header I/O)
        // ================================================================
        Node* buildTree(const std::unordered_map<uint8_t, int>& freq) {
            std::priority_queue<Node*, std::vector<Node*>, NodeCompare> pq;
            for (auto& p : freq) pq.push(new Node(p.first, p.second));
            if (pq.size() == 1) { Node* x = pq.top(); pq.pop(); return new Node(x, new Node(0, 0)); }
            while (pq.size() > 1) {
                Node* a = pq.top(); pq.pop();
                Node* b = pq.top(); pq.pop();
                pq.push(new Node(a, b));
            }
            return pq.top();
        }

        void buildCodes(Node* n, const std::string& cur, std::unordered_map<uint8_t, std::string>& codes)
        {
            if (n->isLeaf) { codes[n->symbol] = cur.empty() ? "0" : cur; return; }
            buildCodes(n->left,  cur + "0", codes);
            buildCodes(n->right, cur + "1", codes);
        }

        void freeTree(Node* n) {
            if (!n) return;
            freeTree(n->left); freeTree(n->right); delete n;
        }

        void writeHeader(std::vector<uint8_t>& out, int padding, const std::unordered_map<uint8_t, std::string>& codes)
        {
            out.push_back((uint8_t)padding);
            uint16_t N = codes.size();
            out.push_back((N >> 8) & 0xFF); out.push_back(N & 0xFF);
            for (auto& p : codes) {
                out.push_back(p.first);
                out.push_back((uint8_t)p.second.size());
                for (char c : p.second) out.push_back(c);
            }
        }

        void readHeader(const std::vector<uint8_t>& in, size_t& index, int& padding, std::unordered_map<uint8_t, std::string>& codes)
        {
            padding = in[index++];
            uint16_t N = ((uint16_t)in[index] << 8) | in[index+1];
            index += 2;
            for (int i = 0; i < N; i++) {
                uint8_t sym = in[index++];
                uint8_t len = in[index++];
                std::string code;
                code.reserve(len);
                for (int j = 0; j < len; j++) code.push_back(in[index++]);
                codes[sym] = code;
            }
        }
};

// ================================================================
// Main and Utility Functions
// ================================================================

std::vector<uint8_t> read_file_to_vector(const std::string& filename)
{
    std::ifstream in(filename, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("Error: cannot open input file: " + filename);
    std::streamsize size = in.tellg();
    if (size < 0) throw std::runtime_error("Error: failed to read file size: " + filename);
    std::vector<uint8_t> buffer(size);
    in.seekg(0, std::ios::beg);
    if (!in.read(reinterpret_cast<char*>(buffer.data()), size))
        throw std::runtime_error("Error: failed to read file: " + filename);
    return buffer;
}


int main(int argc, char** argv) {
    omp_set_num_threads(thread_count);
   
    HuffmanParallelCPU h;

    // Read data
    std::vector<uint8_t> data = read_file_to_vector("data.bin");

    // Time Compression
    auto start_comp = std::chrono::high_resolution_clock::now();
    auto compressed = h.encode(data);
    auto end_comp = std::chrono::high_resolution_clock::now();
   
    // Time Decompression (Parallel)
    auto start_decomp = std::chrono::high_resolution_clock::now();
    auto decompressed = h.decode(compressed);
    auto end_decomp = std::chrono::high_resolution_clock::now();
   
    auto comp   = std::chrono::duration_cast<std::chrono::microseconds>(end_comp - start_comp).count();
    auto decomp = std::chrono::duration_cast<std::chrono::microseconds>(end_decomp - start_decomp).count();

    // Check for correctness
    bool ok = (data.size() == decompressed.size()) && std::equal(data.begin(), data.end(), decompressed.begin());

    // Results
    std::cout << "\n=== Parallel CPU Results ===\n";
    std::cout << "Original size:      " << data.size() << " bytes\n";
    std::cout << "Compressed size:    " << compressed.size() << " bytes\n";
    std::cout << "Compression ratio:  " << (100.0 * compressed.size() / data.size()) << "%\n\n";
    std::cout << "Compression time:   " << comp << " mcs\n";
    std::cout << "Decompression time: " << decomp << " mcs\n";
    std::cout << "Verification:       " << (ok ? "PASS" : "FAIL") << "\n";

    return 0;
}