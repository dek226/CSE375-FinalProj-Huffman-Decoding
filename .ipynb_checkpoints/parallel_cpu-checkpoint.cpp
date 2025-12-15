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


int thread_count = 16;  


class HuffmanParallelCPU {
    public:
        // Encode (parallel)
        std::vector<uint8_t> encode(const std::vector<uint8_t>& data) {
            if (data.empty()) return {};
    
            // 1. Parallel frequency count
            auto freq = countFrequenciesParallel(data);
    
            // 2. Sequential tree build (tiny)
            Node* root = buildTree(freq);
    
            // 3. Sequential code table creation (tiny)
            std::unordered_map<uint8_t, std::string> codes;
            buildCodes(root, "", codes);
    
            // 4. Parallel bitstream generation
            std::string bitstream = encodeParallel(data, codes);
    
            // 5. Pad
            int padding = (8 - (bitstream.size() % 8)) % 8;
            bitstream.append(padding, '0');
    
            // 6. Write header
            std::vector<uint8_t> output;
            writeHeader(output, padding, codes);
    
            // 7. Write payload
            for (size_t i = 0; i < bitstream.size(); i += 8) {
                uint8_t byte = std::stoi(bitstream.substr(i, 8), nullptr, 2);
                output.push_back(byte);
            }
    
            freeTree(root);
            return output;
        }



        // Decode (sequential)
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
            bits.reserve((compressed.size() - index) * 8);
    
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
    
            for (char c : bits) {
                cur.push_back(c);
                if (decodeMap.count(cur)) {
                    out.push_back(decodeMap[cur]);
                    cur.clear();
                }
            }
    
            return out;
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
        // PARALLEL FREQUENCY COUNTING
        // ================================================================
        std::unordered_map<uint8_t, int>
        countFrequenciesParallel(const std::vector<uint8_t>& data)
        {
            constexpr int SYMBOLS = 256;
    
            //int thread_count = omp_get_max_threads();
            std::vector<std::array<int, SYMBOLS>> local(thread_count);
    
            for (auto& arr : local)
                arr.fill(0);
    
            // Parallel chunk counting
            
            #pragma omp parallel 
            {
                int tid = omp_get_thread_num();
                auto& table = local[tid];
    
                #pragma omp for
                for (size_t i = 0; i < data.size(); i++) {
                    table[data[i]]++;
                }
            }
    
            // Reduce
            std::array<int, SYMBOLS> global{};
            global.fill(0);
    
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
    
        // ================================================================
        // Tree building (sequential)
        // ================================================================
        Node* buildTree(const std::unordered_map<uint8_t, int>& freq) {
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
                        std::unordered_map<uint8_t, std::string>& codes)
        {
            if (n->isLeaf) {
                codes[n->symbol] = cur.empty() ? "0" : cur;
                return;
            }
    
            buildCodes(n->left,  cur + "0", codes);
            buildCodes(n->right, cur + "1", codes);
        }
    
        void freeTree(Node* n) {
            if (!n) return;
            freeTree(n->left);
            freeTree(n->right);
            delete n;
        }
    
        // ================================================================
        // PARALLEL BITSTREAM ENCODING
        // ================================================================
        std::string encodeParallel(const std::vector<uint8_t>& data,
                                   const std::unordered_map<uint8_t, std::string>& codes)
        {
            //int thread_count = omp_get_max_threads();
            std::vector<std::string> local(thread_count);
    
            // Thread-local encode
            #pragma omp parallel 
            {
                int tid = omp_get_thread_num();
                auto& out = local[tid];
    
                out.reserve(data.size() / thread_count * 2);
    
                #pragma omp for nowait
                for (size_t i = 0; i < data.size(); i++) {
                    out += codes.at(data[i]);
                }
            }
    
            // Concatenate results
            std::string result;
            result.reserve(data.size() * 2);
    
            for (auto& s : local)
                result += s;
    
            return result;
        }
    
        // ================================================================
        // Header helpers
        // ================================================================
        void writeHeader(std::vector<uint8_t>& out,
                         int padding,
                         const std::unordered_map<uint8_t, std::string>& codes)
        {
            out.push_back((uint8_t)padding);
    
            uint16_t N = codes.size();
            out.push_back((N >> 8) & 0xFF);
            out.push_back(N & 0xFF);
    
            for (auto& p : codes) {
                out.push_back(p.first);              // symbol
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
};


//read file
std::vector<uint8_t> read_file_to_vector(const std::string& filename)
{
    // Open file at end to get size immediately
    std::ifstream in(filename, std::ios::binary | std::ios::ate);
    if (!in) {
        throw std::runtime_error("Error: cannot open input file: " + filename);
    }

    // Get file size
    std::streamsize size = in.tellg();
    if (size < 0) {
        throw std::runtime_error("Error: failed to read file size: " + filename);
    }

    // Allocate vector of correct size
    std::vector<uint8_t> buffer(size);

    // Seek back to start and read file
    in.seekg(0, std::ios::beg);
    if (!in.read(reinterpret_cast<char*>(buffer.data()), size)) {
        throw std::runtime_error("Error: failed to read file: " + filename);
    }

    return buffer;
}


int main(int argc, char** argv) {
    omp_set_num_threads(thread_count);   //set thread count
    
    HuffmanParallelCPU h;

    //read data
    std::vector<uint8_t> data = read_file_to_vector("data.bin");

    //time compression and decompression
    auto start_comp = std::chrono::high_resolution_clock::now();
    auto compressed = h.encode(data);
    auto end_comp = std::chrono::high_resolution_clock::now();
    
    auto start_decomp = std::chrono::high_resolution_clock::now();
    auto decompressed = h.decode(compressed);
    auto end_decomp = std::chrono::high_resolution_clock::now();
    
    auto comp   = std::chrono::duration_cast<std::chrono::microseconds>(end_comp - start_comp).count();
    auto decomp = std::chrono::duration_cast<std::chrono::microseconds>(end_decomp - start_decomp).count();

    //check for correctness
    bool ok = (data.size() == decompressed.size()) && std::equal(data.begin(), data.end(), decompressed.begin());

    //results
    std::cout << "\n=== Parallel CPU Results ===\n";
    std::cout << "Original size:      " << data.size() << " bytes\n";
    std::cout << "Compressed size:    " << compressed.size() << " bytes\n";
    std::cout << "Compression ratio:  " << (100.0 * compressed.size() / data.size()) << "%\n\n";
    std::cout << "Compression time:   " << comp << " mcs\n";
    std::cout << "Decompression time: " << decomp << " mcs\n";
    std::cout << "Verification:       " << (ok ? "PASS" : "FAIL") << "\n";

    return 0;
}


