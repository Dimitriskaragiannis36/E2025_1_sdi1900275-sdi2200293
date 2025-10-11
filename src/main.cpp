#include "helper.h"
#include "lsh.h"
#include "mnist.h"
#include "kmeans.h"
#include "silhouette.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>

int main(int argc, char* argv[]) {
    //ανάγνωση παραμέτρων από τη γραμμή εντολών
    utils::Params params = utils::parse_args(argc, argv);
    utils::print_params(params);

    //φόρτωση δεδομένων
    std::vector<std::vector<float>> data;
    std::vector<std::vector<float>> queries;

    if (params.type == "mnist") {
        data = load_mnist_images(params.input_file, 1000);
        queries = load_mnist_images(params.query_file, 10);
    } else {
        std::cerr << "SIFT loader not implemented yet!\n";
        return 1;
    }

    //δημιουργία και εκπαίδευση του LSH
    nn::LSH lsh(
        static_cast<int>(data[0].size()),
        params.L,
        params.k,
        params.w,
        params.seed
    );
    lsh.build_index(data);

    //άνοιγμα αρχείου εξόδου
    std::ofstream out("results.txt");
    if (!out.is_open()) {
        std::cerr << "Error opening output file!\n";
        return 1;
    }
    out << std::fixed << std::setprecision(6);

    //εκτέλεση queries & υπολογισμός μετρικών
    utils::run_queries(lsh, data, queries, params, out);

    //καθαρισμός
    out.close();
    lsh.clear_index();

    
    //προαιρετικό: K-Means + Silhouette
    std::cout << "\nRunning K-Means clustering on the same dataset...\n";

    clustering::KMeans model(
        3, 100, 1e-4,
        clustering::KMeans::InitMethod::KMEANS_PLUS_PLUS,
        42, true, utils::euclidean_distance
    );

    model.fit(data);

    clustering::Silhouette sil;
    float score = sil.compute(data, model.labels(), model.k());

    std::cout << "\nSilhouette score = " << score << std::endl;
    

    return 0;
}
