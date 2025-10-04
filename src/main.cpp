#include "lsh.h"
#include "mnist.h"
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
using namespace std;

int main(int argc, char* argv[]) {
    //default τιμές
    int seed = 1;
    int k = 4;
    int L = 5;
    float w = 4.0f;
    int N = 1;
    float R = 2000.0f; //default για MNIST
    string input_file = "data/input.dat";
    string query_file = "data/query.dat";
    string output_file = "results.txt";
    string type = "mnist";
    bool do_range = true;

    //διάβασμα παραμέτρων με flags
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "-d" && i+1 < argc) input_file = argv[++i];
        else if (arg == "-q" && i+1 < argc) query_file = argv[++i];
        else if (arg == "-o" && i+1 < argc) output_file = argv[++i];
        else if (arg == "-k" && i+1 < argc) k = atoi(argv[++i]);
        else if (arg == "-L" && i+1 < argc) L = atoi(argv[++i]);
        else if (arg == "-w" && i+1 < argc) w = atof(argv[++i]);
        else if (arg == "-N" && i+1 < argc) N = atoi(argv[++i]);
        else if (arg == "-R" && i+1 < argc) R = atof(argv[++i]);
        else if (arg == "-type" && i+1 < argc) type = argv[++i];
        else if (arg == "-range" && i+1 < argc) do_range = (string(argv[++i]) == "true");
        else if (arg == "-seed" && i+1 < argc) seed = atoi(argv[++i]);
    }

    if (type == "sift") {
        R = 2.0f; //default για SIFT
    }

    cout << "Parameters: "
         << "seed=" << seed << " "
         << "k=" << k << " "
         << "L=" << L << " "
         << "w=" << w << " "
         << "N=" << N << " "
         << "R=" << R << " "
         << "type=" << type << " "
         << "input=" << input_file << " "
         << "query=" << query_file << " "
         << "output=" << output_file << " "
         << "range=" << (do_range ? "true" : "false") << endl;

    //φόρτωσε δεδομένα
    vector<vector<float>> data, queries;
    if (type == "mnist") {
        data = load_mnist_images(input_file, -1);
        queries = load_mnist_images(query_file, -1);
    } else {
        cerr << "SIFT loader not implemented yet!" << endl;
        return 1;
    }

    //φτιάξε LSH object
    nn::LSH lsh(data[0].size(), L, k, w, seed);
    lsh.build_index(data);

    //άνοιξε output file
    ofstream out(output_file);
    if (!out.is_open()) {
        cerr << "Error opening output file!" << endl;
        return 1;
    }

    //τρέξε queries
    for (size_t qi = 0; qi < queries.size(); ++qi) {
        out << "Query " << qi << ":\n";

        int nn_idx = lsh.nn_query(queries[qi]);
        out << "  Nearest Neighbor: " << nn_idx << "\n";

        auto knn = lsh.knn_query(queries[qi], N);
        out << "  " << N << "-Nearest Neighbors:";
        for (int idx : knn) out << " " << idx;
        out << "\n";

        if (do_range) {
            auto range = lsh.range_search(queries[qi], R);
            out << "  Range (R=" << R << "): " << range.size() << " neighbors\n";
        }
        out << "\n";
    }

    out.close();
    lsh.clear_index();
    return 0;
}
