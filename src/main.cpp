#include "lsh.h"
#include "mnist.h"
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <limits>
#include <algorithm>

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
    auto t0_total = chrono::high_resolution_clock::now();
    double sum_AF = 0.0;
    int recall_count = 0;
    double sum_tApprox = 0.0;
    double sum_tTrue = 0.0;

    for (size_t qi = 0; qi < queries.size(); ++qi) {
        out << "LSH\n";
        out << "Query: " << qi << "\n";

        //approximate kNN query
        auto t0 = chrono::high_resolution_clock::now();
        auto knn = lsh.knn_query(queries[qi], N);
        auto t1 = chrono::high_resolution_clock::now();
        double tApprox_ms = chrono::duration<double, std::milli>(t1 - t0).count();
        sum_tApprox += tApprox_ms;

        //αποστάσεις approximate
        vector<double> approx_dists(N, 0.0);
        for (int i = 0; i < (int)knn.size(); ++i) {
            double dist = 0.0;
            for (size_t d = 0; d < queries[qi].size(); ++d)
                dist += (queries[qi][d] - data[knn[i]][d]) * (queries[qi][d] - data[knn[i]][d]);
            approx_dists[i] = sqrt(dist);
            out << "Nearest neighbor-" << (i+1) << ": " << knn[i] << "\n";
            out << "distanceApproximate: " << approx_dists[i] << "\n";
        }

        //ακριβές NN distance
        auto t0_true = chrono::high_resolution_clock::now();
        double best_dist = std::numeric_limits<double>::max();
        int true_nn = -1;
        for (size_t i = 0; i < data.size(); ++i) {
            double dist = 0.0;
            for (size_t d = 0; d < queries[qi].size(); ++d)
                dist += (queries[qi][d] - data[i][d]) * (queries[qi][d] - data[i][d]);
            dist = sqrt(dist);
            if (dist < best_dist) {
                best_dist = dist;
                true_nn = i;
            }
        }
        auto t1_true = chrono::high_resolution_clock::now();
        double tTrue_ms = chrono::duration<double, std::milli>(t1_true - t0_true).count();
        sum_tTrue += tTrue_ms;

        //εκτύπωση true distances
        for (int i = 0; i < (int)knn.size(); ++i) {
            out << "distanceTrue: " << best_dist << "\n";
        }

        //υπολογισμός AF και Recall@N
        sum_AF += approx_dists[0] / best_dist;  // πρώτος NN
        if (std::find(knn.begin(), knn.end(), true_nn) != knn.end()) recall_count++;

        //range search
        if (do_range) {
            auto range = lsh.range_search(queries[qi], R);
            out << "R-near neighbors:\n";
            for (int idx : range) out << idx << "\n";
        }
        out << "\n";
    }

    //μέσοι όροι και QPS
    auto t1_total = chrono::high_resolution_clock::now();
    double total_time_sec = chrono::duration<double>(t1_total - t0_total).count();
    double QPS = queries.size() / total_time_sec;
    double avg_AF = sum_AF / queries.size();
    double recall_at_N = (double)recall_count / queries.size();
    double tApproxAvg = sum_tApprox / queries.size();
    double tTrueAvg = sum_tTrue / queries.size();

    out << "Average AF: " << avg_AF << "\n";
    out << "Recall@N: " << recall_at_N << "\n";
    out << "QPS: " << QPS << "\n";
    out << "tApproximateAverage: " << tApproxAvg << " ms\n";
    out << "tTrueAverage: " << tTrueAvg << " ms\n";

    out.close();
    lsh.clear_index();
    return 0;
}
