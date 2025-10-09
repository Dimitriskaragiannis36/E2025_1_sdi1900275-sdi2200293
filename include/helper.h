#ifndef HELPER_H
#define HELPER_H

#include <vector>
#include <string>
#include <fstream>
#include <functional>

using namespace std;

namespace nn { class LSH; } 

namespace utils {

using DistanceFunc = function<float(const vector<float>&, const vector<float>&)>;
//δομή για τις παραμέτρους του προγράμματος
struct Params {
    int seed = 1;
    int k = 4;
    int L = 5;
    int w = 4;
    int N = 1;
    float R = 2000.0f;
    string input_file = "data/input.dat";
    string query_file = "data/query.dat";
    string output_file = "results.txt";
    string type = "mnist";
    bool do_range = true;
    bool use_lsh = false;
};

//συνάρτηση για την ανάλυση των παραμέτρων από τη γραμμή εντολών
Params parse_args(int argc, char* argv[]);

//συνάρτηση για την εκτύπωση των παραμέτρων
void print_params(const Params& p);

//συνάρτηση για την ανάγνωση δεδομένων από αρχείο
void run_queries(nn::LSH& lsh,
                 const vector<vector<float>>& data,
                 const vector<vector<float>>& queries,
                 const Params& params,
                 ofstream& out);

//ευκλείδεια απόσταση
float euclidean_distance(const vector<float>& x, const vector<float>& y);

} //namespace utils

#endif //HELPER_H
