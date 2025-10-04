#include "helper.h"
#include <iostream>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <unordered_set>
#include <iomanip>

using namespace std;

namespace utils {

    //διάβασμα παραμέτρων από command line
    Params parse_args(int argc, char* argv[]) {
        Params p;
        for (int i = 1; i < argc; i++) {
            string arg = argv[i];
            if (arg == "-d" && i+1 < argc) p.input_file = argv[++i];
            else if (arg == "-q" && i+1 < argc) p.query_file = argv[++i];
            else if (arg == "-o" && i+1 < argc) p.output_file = argv[++i];
            else if (arg == "-k" && i+1 < argc) p.k = atoi(argv[++i]);
            else if (arg == "-L" && i+1 < argc) p.L = atoi(argv[++i]);
            else if (arg == "-w" && i+1 < argc) p.w = atof(argv[++i]);
            else if (arg == "-N" && i+1 < argc) p.N = atoi(argv[++i]);
            else if (arg == "-R" && i+1 < argc) p.R = atof(argv[++i]);
            else if (arg == "-type" && i+1 < argc) p.type = argv[++i];
            else if (arg == "-lsh") {
                p.use_lsh = true;
            }
            else if (arg == "-range" && i+1 < argc) p.do_range = (string(argv[++i]) == "true");
            else if (arg == "-seed" && i+1 < argc) p.seed = atoi(argv[++i]);
        }
        if (p.type == "sift") p.R = 2.0f;
        return p;
    }

    //εκτύπωση παραμέτρων
    void print_params(const Params& p) {
        cout << "Parameters: "
            << "seed=" << p.seed << " "
            << "k=" << p.k << " "
            << "L=" << p.L << " "
            << "w=" << p.w << " "
            << "N=" << p.N << " "
            << "R=" << p.R << " "
            << "type=" << p.type << " "
            << "input=" << p.input_file << " "
            << "query=" << p.query_file << " "
            << "output=" << p.output_file << " "
            << "range=" << (p.do_range ? "true" : "false") << endl;
    }

    //εκτέλεση queries και μέτρηση metrics
    void run_queries(nn::LSH& lsh,
                    const vector<vector<float>>& data,
                    const vector<vector<float>>& queries,
                    const Params& params,
                    ofstream& out) {
        using clock = chrono::high_resolution_clock;

        auto t0_total = clock::now();
        double sum_AF = 0.0, sum_recall = 0.0, sum_tApprox = 0.0, sum_tTrue = 0.0;

        for (size_t qi = 0; qi < queries.size(); ++qi) {
            out << "LSH\nQuery: " << qi << "\n";

            // approximate kNN
            auto t0 = clock::now();
            auto knn = lsh.knn_query(queries[qi], params.N);  // τώρα pair<int,float>
            auto t1 = clock::now();
            double tApprox_ms = chrono::duration<double, milli>(t1 - t0).count();
            sum_tApprox += tApprox_ms;

            // true kNN
            auto t0_true = clock::now();
            vector<pair<double,int>> true_scores;
            true_scores.reserve(data.size());
            for (size_t i = 0; i < data.size(); ++i) {
                double dist = 0.0;
                for (size_t d = 0; d < queries[qi].size(); ++d)
                    dist += (queries[qi][d] - data[i][d]) * (queries[qi][d] - data[i][d]);
                true_scores.emplace_back(sqrt(dist), (int)i);
            }
            int denom_N = min(params.N, (int)true_scores.size());
            nth_element(true_scores.begin(), true_scores.begin()+denom_N, true_scores.end(),
                        [](auto&a,auto&b){return a.first<b.first;});
            true_scores.resize(denom_N);
            sort(true_scores.begin(), true_scores.end(),
                [](auto&a,auto&b){return a.first<b.first;});

            vector<int> true_ids;
            vector<double> true_dists;
            for (auto &p : true_scores) {
                true_ids.push_back(p.second);
                true_dists.push_back(p.first);
            }
            auto t1_true = clock::now();
            double tTrue_ms = chrono::duration<double, milli>(t1_true - t0_true).count();
            sum_tTrue += tTrue_ms;

            // εκτύπωση
            vector<double> approx_dists;
            int nn_counter = 1;
            for (auto [idx, dist] : knn) {
                approx_dists.push_back(dist);
                double trueDist = -1.0;
                auto it = find(true_ids.begin(), true_ids.end(), idx);
                if (it != true_ids.end()) {
                    //είναι στους top-N → πάρε την προϋπολογισμένη απόσταση
                    trueDist = true_dists[it - true_ids.begin()];
                } else {
                    //δεν είναι στους top-N → υπολόγισε απόσταση on the fly
                    double dsum = 0.0;
                    for (size_t d = 0; d < queries[qi].size(); ++d)
                        dsum += (queries[qi][d] - data[idx][d]) * (queries[qi][d] - data[idx][d]);
                    trueDist = sqrt(dsum);
                }
                out << "Nearest neighbor-" << nn_counter++ << ": " << idx << "\n";
                out << "distanceApproximate: " << dist << "\n";
                out << "distanceTrue: " << trueDist << "\n";
            }

            // metrics
            if (!approx_dists.empty() && !true_dists.empty() && true_dists[0] > 1e-12)
                sum_AF += approx_dists[0] / true_dists[0];
            unordered_set<int> true_set(true_ids.begin(), true_ids.end());
            int overlap = 0;
            for (auto [id, d] : knn)
                if (true_set.count(id))
                    overlap++;
            sum_recall += (denom_N > 0) ? (double)overlap / denom_N : 0.0;

            if (params.do_range) {
                auto range = lsh.range_search(queries[qi], params.R);
                out << "R-near neighbors:\n";
                for (int idx : range)
                    out << idx << "\n";
            }
            out << "\n";
        }

        // summary
        auto t1_total = clock::now();
        double total_sec = chrono::duration<double>(t1_total - t0_total).count();
        double QPS = queries.empty() ? 0.0 : queries.size() / total_sec;

        out << "Average AF: " << (sum_AF / queries.size()) << "\n";
        out << "Recall@N: " << (sum_recall / queries.size()) << "\n";
        out << "QPS: " << QPS << "\n";
        out << "tApproximateAverage: " << (sum_tApprox / queries.size()) << " ms\n";
        out << "tTrueAverage: " << (sum_tTrue / queries.size()) << " ms\n";
    }

} //namespace utils
