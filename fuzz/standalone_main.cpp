// Runs the fuzz target over files given on the command line (for platforms without libFuzzer, and
// for replaying crashes): fuzz_decode_replay file...
#include <cstdio>
#include <fstream>
#include <iterator>
#include <vector>
extern "C" int LLVMFuzzerTestOneInput(const unsigned char* data, size_t size);
int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        std::ifstream f(argv[i], std::ios::binary);
        std::vector<unsigned char> d((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        LLVMFuzzerTestOneInput(d.data(), d.size());
    }
    std::printf("replayed %d inputs\n", argc - 1);
    return 0;
}
