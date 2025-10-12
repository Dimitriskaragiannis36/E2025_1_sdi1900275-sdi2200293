#ifndef HELPER_H
#define HELPER_H

#include <vector>
#include <string>
#include <fstream>
#include <functional>
#include <cmath>
#include <unordered_set>


namespace utils {
//τύπος για συνάρτηση απόστασης
using DistanceFunc = std::function<float(const std::vector<float>&, const std::vector<float>&)>;

//παράμετροι από γραμμή εντολών
struct Params {
    //κοινές παράμετροι
    int seed = 1;
    int N = 1;
    float R = 2000.0f;
    std::string input_file = "data/input.dat";
    std::string query_file = "data/query.dat";
    std::string output_file = "results.txt";
    std::string type = "mnist";
    bool do_range = true;

    //LSH
    int k = 4;
    int L = 5;
    int w = 4;
    bool use_lsh = false;

    //IVFFlat
    bool use_ivfflat = false;
    int kclusters = 50;   // αριθμός συστάδων (nlist)
    int nprobe = 5;       // αριθμός clusters που εξετάζονται κατά την αναζήτηση
};

//ανάλυση παραμέτρων από γραμμή εντολών
Params parse_args(int argc, char* argv[]);

//εκτύπωση παραμέτρων
void print_params(const Params& p);

//template-based run_queries (λειτουργεί για LSH, Hypercube, IVF κ.λπ.)
template <typename IndexType>
void run_queries(IndexType& index,
                 const std::vector<std::vector<float>>& data,
                 const std::vector<std::vector<float>>& queries,
                 const Params& params,
                 std::ofstream& out);

//απλή ευκλείδεια απόσταση
float euclidean_distance(const std::vector<float>& x, const std::vector<float>& y);

} //namespace utils

#include "helper.tpp" //θα βάλουμε εδώ το template body

#endif //HELPER_H
