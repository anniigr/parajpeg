#pragma once
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include <dirent.h>

inline std::vector<uint8_t> read_file(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

inline std::vector<std::string> corpus_files(const char* suffix = ".jpg") {
    std::vector<std::string> out;
    DIR* d = opendir(PARAJPEG_CORPUS_DIR);
    if (!d) return out;
    while (dirent* e = readdir(d)) {
        std::string n = e->d_name;
        if (n.size() > 4 && n.compare(n.size() - 4, 4, suffix) == 0) out.push_back(std::string(PARAJPEG_CORPUS_DIR) + "/" + n);
    }
    closedir(d);
    std::sort(out.begin(), out.end());
    return out;
}
