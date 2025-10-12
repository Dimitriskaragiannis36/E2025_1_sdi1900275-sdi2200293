#include "helper.h"
#include "lsh.h"
#include "mnist.h"
#include "kmeans.h"
#include "silhouette.h"
#include "ivff.h"
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

    //δημιουργία αλγορίθμου αναζήτησης (LSH ή IVFFlat ή IVFPQ)
    std::ofstream out(params.output_file);
    if (!out.is_open()) {
        std::cerr << "Error opening output file!\n";
        return 1;
    }
    out << std::fixed << std::setprecision(6);

    auto dim = static_cast<int>(data[0].size());
    //εκτέλεση lsh
    if (params.use_lsh) {
        std::cout << "\n>> Using LSH index...\n";
        nn::LSH lsh(dim, params.L, params.k, params.w, params.seed);
        lsh.build_index(data);
        utils::run_queries(lsh, data, queries, params, out);
        lsh.clear_index();
    }

    //εκτέλεση ivfflat
    else if (params.use_ivfflat) {
        std::cout << "\n>> Using IVFFlat index...\n";
        ivf::IVFFlat index(params.kclusters, params.nprobe, params.seed, utils::euclidean_distance);
        index.build_index(data);
        utils::run_queries(index, data, queries, params, out);
        index.clear_index();
    }

    /*//εκτέλεση ivfpq (προς υλοποίηση)
    else if (params.use_ivfpq) {
        std::cout << "\n>> Using IVFPQ index (placeholder)...\n";
        // μελλοντικά: nn::IVFPQ ivfpq(...)
        // ivfpq.build_index(data);
        // utils::run_queries(ivfpq, data, queries, params, out);
    }*/

    else {
        std::cerr << "Error: No index type specified! Use -lsh or -ivfflat or -ivfpq.\n";
        return 1;
    }

    out.close();


    
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
