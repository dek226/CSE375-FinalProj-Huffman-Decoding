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

class HuffmanSequential {
    public:
        // Encode (compress)
        std::vector<uint8_t> encode(const std::vector<uint8_t>& data) {
            if (data.empty()) return {};
    
            // 1. Build frequency table
            auto freq = countFrequencies(data);
    
            // 2. Build Huffman tree
            Node* root = buildTree(freq);
    
            // 3. Generate code table
            std::unordered_map<uint8_t, std::string> codes;
            buildCodes(root, "", codes);
    
            // 4. Encode bitstream
            std::string bitstream;
            bitstream.reserve(data.size() * 2);
            for (uint8_t b : data)
                bitstream += codes[b];
    
            // 5. Pad to byte boundary
            int padding = (8 - (bitstream.size() % 8)) % 8;
            bitstream.append(padding, '0');
    
            // 6. Build header (padding | code-table-size | code entries)
            std::vector<uint8_t> output;
            writeHeader(output, padding, codes);
    
            // 7. Write compressed bytes
            for (size_t i = 0; i < bitstream.size(); i += 8) {
                uint8_t byte = std::stoi(bitstream.substr(i, 8), nullptr, 2);
                output.push_back(byte);
            }
    
            freeTree(root);
            return output;
        }



        // Decode (decompress)
        std::vector<uint8_t> decode(const std::vector<uint8_t>& compressed) {
            if (compressed.empty())
                return {};
    
            size_t index = 0;
    
            // 1. Read header
            int padding;
            std::unordered_map<uint8_t, std::string> codes;
            readHeader(compressed, index, padding, codes);
    
            // 2. Build decode map (code-string → symbol)
            std::unordered_map<std::string, uint8_t> decodeMap;
            for (auto& p : codes) decodeMap[p.second] = p.first;
    
            // 3. Convert remaining bytes into bitstream
            std::string bitstream;
            bitstream.reserve((compressed.size() - index) * 8);
            for (; index < compressed.size(); index++) {
                uint8_t b = compressed[index];
                for (int bit = 7; bit >= 0; bit--)
                    bitstream.push_back(((b >> bit) & 1) ? '1' : '0');
            }
    
            // Remove padding bits
            if (padding > 0)
                bitstream.resize(bitstream.size() - padding);
    
            // 4. Decode bit-by-bit
            std::vector<uint8_t> output;
            std::string current;
            for (char c : bitstream) {
                current.push_back(c);
                if (decodeMap.count(current)) {
                    output.push_back(decodeMap[current]);
                    current.clear();
                }
            }
    
            return output;
        }

    private:
        // ----------------------------- Internal Structures -----------------------------
        struct Node {
            uint8_t symbol;
            int freq;
            Node* left;
            Node* right;
            bool isLeaf;
            Node(uint8_t s, int f) : symbol(s), freq(f), left(nullptr), right(nullptr), isLeaf(true) {}
            Node(Node* l, Node* r) : symbol(0), freq(l->freq + r->freq), left(l), right(r), isLeaf(false) {}
        };
    
        struct NodeCompare {
            bool operator()(Node* a, Node* b) const { return a->freq > b->freq; }
        };
    
        // ----------------------------- Encoding Helpers -----------------------------
        std::unordered_map<uint8_t, int> countFrequencies(const std::vector<uint8_t>& data) {
            std::unordered_map<uint8_t, int> freq;
            for (uint8_t b : data) 
                freq[b]++;
            return freq;
        }
    
        Node* buildTree(const std::unordered_map<uint8_t, int>& freq) {
            std::priority_queue<Node*, std::vector<Node*>, NodeCompare> pq;
    
            for (auto& p : freq)
                pq.push(new Node(p.first, p.second));
    
            if (pq.size() == 1) {
                // Special case: only one symbol
                Node* only = pq.top();
                pq.pop();
                Node* root = new Node(only, new Node(0, 0)); // fake second leaf
                return root;
            }
    
            while (pq.size() > 1) {
                Node* a = pq.top(); pq.pop();
                Node* b = pq.top(); pq.pop();
                pq.push(new Node(a, b));
            }
    
            return pq.top();
        }
    
        void buildCodes(Node* node, const std::string& current, std::unordered_map<uint8_t, std::string>& codes) {
            if (node->isLeaf) {
                codes[node->symbol] = current.empty() ? "0" : current;
                return;
            }
            buildCodes(node->left, current + "0", codes);
            buildCodes(node->right, current + "1", codes);
        }
    
        void freeTree(Node* node) {
            if (!node) return;
            freeTree(node->left);
            freeTree(node->right);
            delete node;
        }
    
        // ----------------------------- Header Encoding -----------------------------
        void writeHeader(std::vector<uint8_t>& out, int padding, const std::unordered_map<uint8_t, std::string>& codes)
        {
            // Format:
            // [padding:uint8] [num-codes:uint16] then N entries:
            // [symbol:uint8][codeLen:uint8][code bits as ASCII chars]
    
            out.push_back((uint8_t)padding);
    
            uint16_t N = codes.size();
            out.push_back((N >> 8) & 0xFF);
            out.push_back(N & 0xFF);
    
            for (auto& p : codes) {
                uint8_t symbol = p.first;
                const std::string& code = p.second;
    
                out.push_back(symbol);
                out.push_back((uint8_t)code.size());
    
                for (char c : code)
                    out.push_back((uint8_t)c);
            }
        }
    
        void readHeader(const std::vector<uint8_t>& in, size_t& index, int& padding, std::unordered_map<uint8_t, std::string>& codes) {
            padding = in[index++];
    
            uint16_t N = ((uint16_t)in[index] << 8) | in[index + 1];
            index += 2;
    
            for (int i = 0; i < N; i++) {
                uint8_t symbol = in[index++];
                uint8_t len = in[index++];
                std::string code;
                code.reserve(len);
    
                for (int j = 0; j < len; j++)
                    code.push_back((char)in[index++]);
    
                codes[symbol] = code;
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
    HuffmanSequential h;

    //read data
    std::vector<uint8_t> data = read_file_to_vector("data100_100.bin");

    //time compression and decompression
    auto start_comp = std::chrono::high_resolution_clock::now();
    auto compressed = h.encode(data);
    auto end_comp = std::chrono::high_resolution_clock::now();
    
    auto start_decomp = std::chrono::high_resolution_clock::now();
    auto decompressed = h.decode(compressed);
    auto end_decomp = std::chrono::high_resolution_clock::now();
    
    auto comp   = std::chrono::duration_cast<std::chrono::microseconds>(end_comp - start_comp).count();
    auto decomp = std::chrono::duration_cast<std::chrono::microseconds>(end_decomp - start_decomp).count();


    // throughput calculations (MB/s)
    double input_mb = data.size() / (1024.0 * 1024.0);

    double comp_throughput =
        input_mb / (comp * 1e-6);

    double decomp_throughput =
        input_mb / (decomp * 1e-6);

    
    //check for correctness
    bool ok = (data.size() == decompressed.size()) && std::equal(data.begin(), data.end(), decompressed.begin());

    //results
    std::cout << "\n=== Sequential Results ===\n";
    std::cout << "Original size:      " << data.size() << " bytes\n";
    std::cout << "Compressed size:    " << compressed.size() << " bytes\n";
    std::cout << "Compression ratio:  " << (100.0 * compressed.size() / data.size()) << "%\n\n";
    std::cout << "Compression time:   " << comp << " mcs\n";
    std::cout << "Decompression time: " << decomp << " mcs\n";
    std::cout << "Compression throughput (mb/s): " << comp_throughput << "\n";
    std::cout << "Decompression throughput (mb/s): " << decomp_throughput << "\n";
    std::cout << "Verification:       " << (ok ? "PASS" : "FAIL") << "\n";

    return 0;
}

