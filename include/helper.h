#ifndef HELPER_H
#define HELPER_H

#include "lsh.h"
#include <vector>
#include <string>
#include <fstream>

namespace utils {

//δομή για τις παραμέτρους του προγράμματος
struct Params {
    int seed = 1;
    int k = 4;
    int L = 5;
    int w = 4;
    int N = 1;
    float R = 2000.0f;
    std::string input_file = "data/input.dat";
    std::string query_file = "data/query.dat";
    std::string output_file = "results.txt";
    std::string type = "mnist";
    bool do_range = true;
    bool use_lsh = false;
};

//συνάρτηση για την ανάλυση των παραμέτρων από τη γραμμή εντολών
Params parse_args(int argc, char* argv[]);
void print_params(const Params& p);

void run_queries(nn::LSH& lsh,
                 const std::vector<std::vector<float>>& data,
                 const std::vector<std::vector<float>>& queries,
                 const Params& params,
                 std::ofstream& out);

} //namespace utils

#endif //HELPER_H
