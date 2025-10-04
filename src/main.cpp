#include "helper.h"
#include "lsh.h"
#include "mnist.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
using namespace std;

int main(int argc, char* argv[]) {
    //διάβασμα παραμέτρων
    utils::Params params = utils::parse_args(argc, argv);
    utils::print_params(params);

    //φόρτωση δεδομένων
    vector<vector<float>> data, queries;
    if (params.type == "mnist") {
        data = load_mnist_images(params.input_file, -1);
        queries = load_mnist_images(params.query_file, 10);
    } else {
        cerr << "SIFT loader not implemented yet!\n";
        return 1;
    }

    //δημιουργία LSH
    nn::LSH lsh(data[0].size(), params.L, params.k, params.w, params.seed);
    lsh.build_index(data);

    //άνοιγμα εξόδου
    ofstream out(params.output_file);
    if (!out.is_open()) {
        cerr << "Error opening output file!\n";
        return 1;
    }
    out << fixed << setprecision(6);

    //εκτέλεση queries και metrics
    utils::run_queries(lsh, data, queries, params, out);

    //καθαρισμός
    out.close();
    lsh.clear_index();
    return 0;
}
