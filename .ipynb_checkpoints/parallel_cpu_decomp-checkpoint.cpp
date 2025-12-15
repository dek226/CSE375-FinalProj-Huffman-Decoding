// parallel_huffman_decoder.cpp
// Improved Huffman encode/decode with parallel decode boundary-fix + small LUT
// Uses OpenMP for parallel processing on CPUs.
//
// Compile: g++ -O3 -fopenmp -std=c++17 parallel_huffman_decoder.cpp -o parallel_decomp_cpu
// Run: ./parallel_decomp_cpu

#include <vector>
#include <unordered_map>
#include <queue>
#include <cstdint>
#include <string>
#include <stdexcept>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <omp.h>
#include <array>
#include <cmath>

// Global setting for the number of OpenMP threads
int thread_count = 1; // Increased default thread count for better parallel testing
// Block size for parallel decoding (in bytes)
constexpr size_t DECODE_BLOCK_SIZE = 1024 * 64; // 64KB block size
// Number of bits for the small lookup table
constexpr int LUT_BITS = 8; // small byte-wise LUT (256 entries)

// Structure to hold a decoded byte and the bit position where its code started.
struct DecodedSym {
    uint8_t sym;
    size_t start_bit;
    size_t end_bit;
};


// Custom print function for debugging within parallel regions (critical section)
void debug_print(const std::string& msg) {
    #pragma omp critical
    {
        std::cerr << "[Thread " << omp_get_thread_num() << "] " << msg << "\n";
    }
}

// Function to find the first difference between two vectors for verification
void find_first_difference(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
    size_t min_size = std::min(a.size(), b.size());
    size_t diff_index = min_size;
    for (size_t i = 0; i < min_size; ++i) {
        if (a[i] != b[i]) {
            diff_index = i;
            break;
        }
    }
    
    if (diff_index < min_size) {
        std::cerr << "Verification failed at byte index: " << diff_index << "\n";
        std::cerr << "Original byte: " << (int)a[diff_index] << "\n";
        std::cerr << "Decompressed byte: " << (int)b[diff_index] << "\n";
    } else if (a.size() != b.size()) {
        std::cerr << "Verification failed due to size mismatch.\n";
        std::cerr << "Original size: " << a.size() << ", Decompressed size: " << b.size() << "\n";
    }
}


class HuffmanParallelCPU {
public:
    std::vector<uint8_t> encode(const std::vector<uint8_t>& data) {
        if (data.empty()) return {};

        // 1. Parallel Frequency Counting
        auto freq = countFrequenciesParallel(data);
        
        // 2. Sequential Tree Building
        Node* root = buildTree(freq);
        std::unordered_map<uint8_t, std::string> codes;
        buildCodes(root, "", codes);

        // Calculate max code length for parallel boundary handling
        max_code_len_ = 0;
        for (auto &p : codes) if (p.second.size() > max_code_len_) max_code_len_ = p.second.size();
        
        // Debug
        debug_print("Max code length calculated as: " + std::to_string(max_code_len_));

        // 3. Parallel Encoding
        std::string bitstream = encodeParallel(data, codes);
        int padding = (8 - (bitstream.size() % 8)) % 8;
        bitstream.append(padding, '0');
        
        // Debug
        debug_print("Total bitstream size: " + std::to_string(bitstream.size()) + ", padding: " + std::to_string(padding));


        // 4. Serialize
        std::vector<uint8_t> output;
        writeHeader(output, padding, codes);
        // Convert bitstream string to byte vector
        for (size_t i = 0; i < bitstream.size(); i += 8) {
            uint8_t byte = 0;
            for (int j = 0; j < 8; ++j) {
                if (bitstream[i + j] == '1') {
                    // Correct: MSB first (index 0 is bit 7, index 7 is bit 0)
                    byte |= (1 << (7 - j));
                }
            }
            output.push_back(byte);
        }

        freeTree(root);
        return output;
    }

    std::vector<uint8_t> decode(const std::vector<uint8_t>& compressed) {
        return decodeParallel(compressed);
    }

private:
    // Huffman Tree Node for building the codes
    struct Node {
        uint8_t symbol;
        int freq;
        Node* left;
        Node* right;
        bool isLeaf;
        Node(uint8_t s, int f) : symbol(s), freq(f), left(nullptr), right(nullptr), isLeaf(true) {}
        Node(Node* l, Node* r) : symbol(0), freq(l->freq + r->freq), left(l), right(r), isLeaf(false) {}
    };

    // Comparator for the priority queue used in tree building
    struct NodeCompare { bool operator()(Node* a, Node* b) const { return a->freq > b->freq; } };

    // ---------------- ENCODE HELPERS ----------------

    // Counts symbol frequencies across threads
    std::unordered_map<uint8_t,int> countFrequenciesParallel(const std::vector<uint8_t>& data) {
        constexpr int SYMBOLS = 256;
        std::vector<std::array<int,SYMBOLS>> local(thread_count);
        for (auto &arr : local) arr.fill(0);

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            if (tid >= thread_count) tid = 0; // Guard against possible OMP weirdness if thread_count is lower than max
            auto& table = local[tid];
            #pragma omp for nowait
            for (size_t i = 0; i < data.size(); ++i) table[data[i]]++;
        }

        std::array<int,SYMBOLS> global{};
        for (int t=0;t<thread_count;++t) for (int s=0;s<SYMBOLS;++s) global[s] += local[t][s];

        std::unordered_map<uint8_t,int> freq;
        for (int s=0;s<SYMBOLS;++s) if (global[s] > 0) freq[static_cast<uint8_t>(s)] = global[s];
        return freq;
    }

    // Encodes the entire data in parallel by splitting the input data
    std::string encodeParallel(const std::vector<uint8_t>& data, const std::unordered_map<uint8_t,std::string>& codes) {
        std::vector<std::string> local(thread_count);
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            if (tid >= thread_count) tid = 0;
            auto &out = local[tid];
            // Reserve memory based on average code length (approx 1.5x)
            out.reserve(data.size()/thread_count * 1.5);
            #pragma omp for nowait
            for (size_t i=0;i<data.size();++i) out += codes.at(data[i]);
        }
        std::string result; 
        for (auto &s : local) result += s;
        return result;
    }

    // ---------------- DECODER STRUCTURES ----------------

    // simple binary trie node for decode
    struct TrieNode {
        int16_t symbol; // -1 = internal node, >=0 = leaf symbol
        TrieNode* child[2];
        TrieNode(): symbol(-1) { child[0]=child[1]=nullptr; }
    };

    // Builds the Huffman decoding trie from the symbol codes
    TrieNode* buildTrie(const std::unordered_map<uint8_t,std::string>& codes) {
        TrieNode* root = new TrieNode();
        for (auto &p : codes) {
            TrieNode* cur = root;
            const std::string &c = p.second;
            for (char ch : c) {
                int b = (ch == '1');
                if (!cur->child[b]) cur->child[b] = new TrieNode();
                cur = cur->child[b];
            }
            cur->symbol = static_cast<int16_t>(p.first);
        }
        return root;
    }
    
    // Recursively free the memory used by the trie
    void freeTrie(TrieNode* n) {
        if (!n) return;
        freeTrie(n->child[0]); freeTrie(n->child[1]);
        delete n;
    }

    // small LUT: for each byte value, if it resolves to a symbol in <= LUT_BITS bits
    struct LUTEntry { int16_t symbol; uint8_t bitsConsumed; bool valid; };

    // Builds a Look-Up Table for fast decoding of short codes (up to LUT_BITS)
    std::array<LUTEntry, 1<<LUT_BITS> buildLUT(const std::unordered_map<uint8_t,std::string>& codes) {
        std::array<LUTEntry, 1<<LUT_BITS> lut;
        for (size_t i=0;i<lut.size();++i) lut[i] = { -1, 0, false };

        // For each code, fill all possible bit combinations that start with that code
        for (auto &p : codes) {
            const std::string &c = p.second;
            if (c.size() <= LUT_BITS) {
                // Convert code bits to integer prefix
                int prefix = 0;
                for (char ch : c) { prefix = (prefix<<1) | (ch=='1'); }
                
                int suffixBits = LUT_BITS - static_cast<int>(c.size());
                int base = prefix << suffixBits;
                int repeat = 1 << suffixBits; // Number of ways to fill the remaining bits
                
                for (int r = 0; r < repeat; ++r) {
                    int idx = base | r;
                    // Only overwrite if current entry is invalid or the new code is shorter (shorter code takes precedence)
                    if (!lut[idx].valid || c.size() < lut[idx].bitsConsumed) {
                         lut[idx] = { static_cast<int16_t>(p.first), static_cast<uint8_t>(c.size()), true };
                    }
                }
            }
        }
        return lut;
    }

    // sequential bit-level decode within a bit-range using trie (with LUT speedup)
    // Returns symbols along with the bit position where they *started*
    std::vector<DecodedSym> sequentialDecodeRange(const std::string& fullBitstream,
                                                  size_t start_bit, size_t end_bit,
                                                  TrieNode* trie, const std::array<LUTEntry,1<<LUT_BITS>& lut)
    {
        std::vector<DecodedSym> out;
        size_t pos = start_bit;
        while (pos < end_bit) {
            size_t symbol_start = pos;
            
            // 1. Try LUT if enough bits are available in the stream
            if (end_bit - pos >= LUT_BITS) {
                
                // Read next LUT_BITS bits to form the index
                int idx = 0;
                for (int i=0;i<LUT_BITS;i++) {
                    idx = (idx<<1) | (fullBitstream[pos + i] - '0');
                }
                LUTEntry e = lut[idx];
                if (e.valid && pos + e.bitsConsumed <= end_bit) { // Ensure the symbol doesn't exceed the boundary
                    out.push_back({
                    static_cast<uint8_t>(e.symbol),
                    symbol_start,
                    symbol_start + e.bitsConsumed
                    });
                    pos += e.bitsConsumed;
                    continue; // Skip trie lookup
                }
            }
            
            // 2. Fallback to trie bit-by-bit
            TrieNode* cur = trie;
            size_t p = pos;
            bool found = false;
            
            // Limit trie search to avoid reading far beyond end_bit (up to max_code_len_ bits)
            size_t max_search_p = std::min(pos + max_code_len_, end_bit); 

            while (p < max_search_p && cur) {
                int b = fullBitstream[p] - '0';
                cur = cur->child[b];
                p++;
                if (cur && cur->symbol >= 0) {
                    out.push_back({
                    static_cast<uint8_t>(cur->symbol),
                    symbol_start,
                    p
                    });
                    pos = p;
                    found = true;
                    break;
                }
            }
            if (!found) {
                // incomplete code at end of range; stop
                break;
            }
        }
        return out;
    }

    // Align start bit: scan backwards up to max_code_len_ bits and try to find a boundary 
    // so that the first decoded symbol ends at or after nominal_start.
    size_t align_start_bit(const std::string& fullBitstream, size_t nominal_start, size_t total_bits, TrieNode* trie, const std::array<LUTEntry,1<<LUT_BITS>& lut, size_t bidx) {
        size_t max_back = std::min(static_cast<size_t>(max_code_len_), nominal_start);
        
        // Debug
        std::ostringstream dbg_msg;
        dbg_msg << "Block " << bidx << " nominal_start=" << nominal_start << ", max_back=" << max_back;
        debug_print(dbg_msg.str());

        // Try candidate starts from nominal_start - max_back up to nominal_start - 1
        for (size_t cand_offset = max_back; cand_offset > 0; --cand_offset) {
            size_t cand = nominal_start - cand_offset; // candidate start bit position
            
            // Decode up to max_code_len_ bits starting from cand
            // The try_end boundary needs to be large enough to contain at least one full symbol
            size_t try_end = std::min(cand + max_code_len_ + LUT_BITS, total_bits); 
            
            TrieNode* cur = trie;
            size_t p = cand;
            bool symbol_found = false;
            size_t next_p = p;

            // Inner loop for bit-by-bit trie traversal to find the first symbol
            while (p < try_end && cur) {
                // Optimization: Try LUT
                if (try_end - p >= LUT_BITS) {
                    int idx = 0;
                    for (int k=0; k<LUT_BITS; k++) { idx = (idx<<1) | (fullBitstream[p + k] - '0'); }
                    LUTEntry e = lut[idx];
                    if (e.valid && p + e.bitsConsumed <= try_end) {
                        next_p = p + e.bitsConsumed;
                        symbol_found = true;
                        break;
                    }
                }
                
                // Fallback to Trie
                TrieNode* inner_cur = cur;
                size_t q = p;
                bool inner_found = false;
                
                // Reset trie search to root for full symbol lookup
                inner_cur = trie;
                
                while (q < try_end && inner_cur) {
                    int b = fullBitstream[q] - '0';
                    inner_cur = inner_cur->child[b];
                    q++;
                    
                    if (inner_cur && inner_cur->symbol >= 0) {
                        inner_found = true;
                        next_p = q;
                        break; 
                    }
                }
                
                if(inner_found) {
                    symbol_found = true;
                    break;
                }
                
                // If we reach here, neither LUT nor Trie found a symbol starting at 'cand' quickly, break outer while.
                break;
            }
            
            if (symbol_found) { 
                if (next_p > nominal_start) {
                    // Found alignment: The symbol starting at 'cand' finishes after the nominal boundary.
                    dbg_msg.str("");
                    dbg_msg << "Block " << bidx << " aligned_start=" << cand << " (nominal=" << nominal_start << ") - first symbol ends at " << next_p;
                    debug_print(dbg_msg.str());
                    return cand;
                }
                // If it ends at or before nominal_start, the boundary is further in. Continue to the next candidate.
            } 
        }
        
        // If no pre-boundary start was found, fall back to nominal_start.
        dbg_msg.str("");
        dbg_msg << "Block " << bidx << " aligned_start failed, falling back to nominal_start=" << nominal_start;
        debug_print(dbg_msg.str());
        return nominal_start;
    }

    // ---------------- DECODE PARALLEL MAIN ----------------

    std::vector<uint8_t> decodeParallel(const std::vector<uint8_t>& compressed) {
        size_t index = 0;
        int padding = 0;
        std::unordered_map<uint8_t,std::string> codes;
        
        try {
            readHeader(compressed, index, padding, codes);
        } catch (const std::runtime_error& e) {
            std::cerr << "Error reading header: " << e.what() << "\n";
            return {};
        }
        
        // Need to set max_code_len_ from header if not set by encode
        if (max_code_len_ == 0) {
            for (auto &p : codes) if (p.second.size() > max_code_len_) max_code_len_ = p.second.size();
        }
        
        if (index > compressed.size()) return {};

        // build decode structures
        TrieNode* trie = buildTrie(codes);
        auto lut = buildLUT(codes);

        // convert payload to bitstring
        std::string fullBitstream;
        fullBitstream.reserve((compressed.size() - index) * 8);
        for (size_t i = index; i < compressed.size(); ++i) {
            uint8_t b = compressed[i];
            // MSB first (bit 7 to bit 0)
            for (int bit=7; bit>=0; --bit) fullBitstream.push_back(((b>>bit)&1) ? '1' : '0');
        }
        if (padding > 0 && fullBitstream.size() >= static_cast<size_t>(padding))
            fullBitstream.resize(fullBitstream.size() - padding);

        size_t total_bits = fullBitstream.size();
        if (total_bits == 0) {
            freeTrie(trie);
            return {};
        }

        size_t bits_per_block = DECODE_BLOCK_SIZE * 8;
        size_t num_blocks = (total_bits + bits_per_block - 1) / bits_per_block;
        std::vector<std::vector<uint8_t>> decoded_blocks(num_blocks);

        debug_print("Total bits: " + std::to_string(total_bits) + ", bits_per_block: " + std::to_string(bits_per_block) + ", num_blocks: " + std::to_string(num_blocks));

        auto t0 = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for schedule(dynamic)
        for (size_t bidx = 0; bidx < num_blocks; ++bidx) {
            size_t start_bit_nominal = bidx * bits_per_block;
            size_t end_bit_nominal = std::min(start_bit_nominal + bits_per_block, total_bits);
            
            // The decoding window end is extended by max_code_len_ to ensure the last symbol is captured.
            size_t decode_window_end = std::min(end_bit_nominal + max_code_len_, total_bits);

            std::ostringstream dbg_msg;

            // Align start bit: only apply alignment for blocks after the first one (bidx > 0)
            size_t aligned_start = start_bit_nominal;
            if (start_bit_nominal > 0) {
                aligned_start = align_start_bit(fullBitstream, start_bit_nominal, total_bits, trie, lut, bidx);
            }
            
            dbg_msg << "Block " << bidx << ": start_nominal=" << start_bit_nominal << ", end_nominal=" << end_bit_nominal 
                    << ", aligned_start=" << aligned_start << ", window_end=" << decode_window_end;
            debug_print(dbg_msg.str());

            // decode from aligned_start up to decode_window_end
            std::vector<DecodedSym> local_decoded_syms = sequentialDecodeRange(fullBitstream, aligned_start, decode_window_end, trie, lut);

            // Find the index of the first symbol that STARTS at or after the nominal start boundary.
            int symbols_to_skip = 0;
            dbg_msg.str("");
            dbg_msg << "Block " << bidx << " Decoded Syms (total=" << local_decoded_syms.size() << "): ";
            
            // FIX: Skip symbols that STARTED *before* the nominal boundary.
            for (size_t i = 0; i < local_decoded_syms.size(); ++i) {
                dbg_msg << "(" << (int)local_decoded_syms[i].sym << ", " << local_decoded_syms[i].start_bit << "-" << local_decoded_syms[i].end_bit << ") ";
                
                // If the symbol started *before* the nominal boundary, it belongs to the previous block (and must be skipped here).
                if (local_decoded_syms[i].start_bit < start_bit_nominal) { 
                    symbols_to_skip++;
                } else {
                    // This is the first symbol that started at or after the nominal boundary. KEEP it.
                    break;
                }
            }
            debug_print(dbg_msg.str());
            
            dbg_msg.str("");
            dbg_msg << "Block " << bidx << " symbols_to_skip: " << symbols_to_skip << ", first symbol to keep starts at: " 
                    << (symbols_to_skip < (int)local_decoded_syms.size() ? std::to_string(local_decoded_syms[symbols_to_skip].start_bit) : "N/A");
            debug_print(dbg_msg.str());

            // Filter and extract final symbols for this block
            std::vector<uint8_t> final_symbols;
            
            for (size_t i = symbols_to_skip; i < local_decoded_syms.size(); ++i) {
                size_t symbol_start = local_decoded_syms[i].start_bit;
                
                // Keep the symbol if it STARTS *before* the nominal end boundary.
                if (symbol_start < end_bit_nominal) {
                    final_symbols.push_back(local_decoded_syms[i].sym);
                } else {
                    // Symbol starts at or after the nominal end of the block. Stop.
                    break;
                }
            }
            
            dbg_msg.str("");
            dbg_msg << "Block " << bidx << " final_symbols size: " << final_symbols.size();
            debug_print(dbg_msg.str());

            decoded_blocks[bidx] = std::move(final_symbols);
        }
        auto t1 = std::chrono::high_resolution_clock::now();

        // assemble
        size_t total_out = 0;
        for (auto &v : decoded_blocks) total_out += v.size();
        std::vector<uint8_t> output;
        output.reserve(total_out);
        for (size_t i=0;i<decoded_blocks.size();++i) {
            output.insert(output.end(), decoded_blocks[i].begin(), decoded_blocks[i].end());
        }

        freeTrie(trie);

        auto micro = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        #pragma omp critical
        {
            std::cout << "Parallel decode elapsed: " << micro << " us (threads=" << omp_get_max_threads() << ")\n";
        }
        return output;
    }

    // ---------------- TREE & HEADER I/O ----------------

    // Builds the Huffman tree from symbol frequencies
    Node* buildTree(const std::unordered_map<uint8_t,int>& freq) {
        std::priority_queue<Node*, std::vector<Node*>, NodeCompare> pq;
        for (auto &p : freq) pq.push(new Node(p.first, p.second));
        if (pq.empty()) return nullptr;
        // Handle single symbol case: ensure a two-node root for a valid code ("0")
        if (pq.size() == 1) { 
             Node* x = pq.top(); pq.pop(); 
             // Create a dummy node with symbol 0, frequency 0 to complete the tree
             Node* dummy_leaf = new Node(0, 0); 
             return new Node(x, dummy_leaf); 
        }
        while (pq.size() > 1) {
            Node* a = pq.top(); pq.pop();
            Node* b = pq.top(); pq.pop();
            pq.push(new Node(a,b));
        }
        return pq.top();
    }

    // Generates the bit codes by traversing the Huffman tree
    void buildCodes(Node* n, const std::string& cur, std::unordered_map<uint8_t,std::string>& codes) {
        if (!n) return;
        if (n->isLeaf) { 
             // Single-symbol case: The only symbol is forced to "0" (or "0" and dummy "1").
             // If tree has >1 node, it must have two children, so cur will not be empty.
             // If single symbol, tree is: root -> (leaf | dummy_leaf). Leaf is 0, dummy_leaf is 1.
             if (n->freq > 0 && cur.empty()) { // Check for the root node if it's the only symbol (single code word case)
                codes[n->symbol] = "0"; 
             } else if (n->freq > 0) {
                 codes[n->symbol] = cur;
             }
             return; 
        }
        buildCodes(n->left,  cur + "0", codes);
        buildCodes(n->right, cur + "1", codes);
    }

    // Recursively free the memory used by the tree
    void freeTree(Node* n) {
        if (!n) return;
        freeTree(n->left); freeTree(n->right);
        delete n;
    }

    // Writes the header (padding, symbol count, and codes) to the output vector
    void writeHeader(std::vector<uint8_t>& out, int padding, const std::unordered_map<uint8_t,std::string>& codes) {
        out.push_back(static_cast<uint8_t>(padding));
        uint16_t N = static_cast<uint16_t>(codes.size());
        // Write N (symbol count) as two bytes (big-endian)
        out.push_back((N>>8)&0xFF);
        out.push_back(N & 0xFF);
        for (auto &p : codes) {
            out.push_back(p.first); // Symbol
            out.push_back(static_cast<uint8_t>(p.second.size())); // Code length
            // Write code bits as ASCII '0'/'1'
            for (char c : p.second) out.push_back(static_cast<uint8_t>(c));
        }
    }

    // Reads the header and reconstructs the symbol codes
    void readHeader(const std::vector<uint8_t>& in, size_t& index, int& padding, std::unordered_map<uint8_t,std::string>& codes) {
        if (index >= in.size()) throw std::runtime_error("Bad header: padding byte missing");
        padding = in[index++];
        if (index + 1 >= in.size()) throw std::runtime_error("Bad header size: N missing");
        // Read N (symbol count) as two bytes (big-endian)
        uint16_t N = (static_cast<uint16_t>(in[index])<<8) | static_cast<uint16_t>(in[index+1]);
        index += 2;
        for (uint16_t i=0;i<N;++i) {
            if (index + 1 >= in.size()) throw std::runtime_error("Bad header content: symbol or length missing");
            uint8_t sym = in[index++];
            uint8_t len = in[index++];
            std::string code; code.reserve(len);
            for (uint8_t j=0;j<len;++j) {
                if (index >= in.size()) throw std::runtime_error("Bad header code chars: code bit missing");
                code.push_back(static_cast<char>(in[index++]));
            }
            codes[sym] = code;
        }
    }

    // Stores the maximum code length for boundary alignment/checking
    size_t max_code_len_ = 0;
};

// ---------------- File IO utils ----------------
std::vector<uint8_t> read_file_to_vector(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("Cannot open: " + filename);
    std::streamsize size = in.tellg();
    if (size < 0) return {};
    std::vector<uint8_t> buf(static_cast<size_t>(size));
    in.seekg(0,std::ios::beg);
    in.read(reinterpret_cast<char*>(buf.data()), size);
    return buf;
}
void write_vector_to_file(const std::string& filename, const std::vector<uint8_t>& data) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) throw std::runtime_error("Cannot write: " + filename);
    out.write(reinterpret_cast<const char*>(data.data()), data.size());
}

// ---------------- MAIN ----------------
int main(int argc, char** argv) {
    // Set thread count for OpenMP based on the global variable
    omp_set_num_threads(thread_count);
    HuffmanParallelCPU h;

    try {
        // Create a dummy data.bin file if it doesn't exist for testing
        std::ifstream check_file("data.bin");
        if (!check_file.good() || check_file.peek() == std::ifstream::traits_type::eof()) {
            std::cout << "Creating dummy data.bin (256KB) for testing...\n";
            std::vector<uint8_t> dummy_data;
            // Create data that contains all 256 bytes repeatedly (good for huffman)
            for (int i = 0; i < 256; ++i) { 
                for (int j = 0; j < 1000; ++j) { 
                    dummy_data.push_back(static_cast<uint8_t>(i));
                }
            }
            write_vector_to_file("data.bin", dummy_data);
        }
        check_file.close();


        auto t_all_s = std::chrono::high_resolution_clock::now();
        auto data = read_file_to_vector("data.bin");
        
        if (data.empty()) {
            std::cout << "Input file data.bin is empty. Exiting.\n";
            return 1;
        }

        // --- Encode ---
        auto t0 = std::chrono::high_resolution_clock::now();
        auto compressed = h.encode(data);
        auto t1 = std::chrono::high_resolution_clock::now();
        
        // --- Decode ---
        auto t1_5 = std::chrono::high_resolution_clock::now(); 
        auto decompressed = h.decode(compressed);
        auto t2 = std::chrono::high_resolution_clock::now();

        write_vector_to_file("compressed.huff", compressed);
        write_vector_to_file("decompressed.bin", decompressed);

        auto enc_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        auto dec_us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1_5).count(); 

        bool ok = (data.size() == decompressed.size()) && std::equal(data.begin(), data.end(), decompressed.begin());

        std::cout << "\n=== Parallel CPU Huffman Results ===\n";
        std::cout << "Input File: data.bin\n";
        std::cout << "Original size:    " << data.size() << " bytes\n";
        std::cout << "Compressed size: " << compressed.size() << " bytes\n";
        std::cout << "Compression ratio: " << (100.0 * static_cast<double>(compressed.size()) / data.size()) << "%\n\n";
        std::cout << "Compression time (Parallel Encode):    " << enc_us << " us\n";
        std::cout << "Decompression time (Parallel Decode): " << dec_us << " us\n";
        std::cout << "Verification:          " << (ok ? "PASS" : "FAIL") << "\n";
        
        if (!ok) {
            find_first_difference(data, decompressed);
        }

    } catch (const std::exception &ex) {
        std::cerr << "Exception: " << ex.what() << "\n";
        return 2;
    }
    return 0;
}