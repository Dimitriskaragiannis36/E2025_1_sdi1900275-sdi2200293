#include "helper.h"
#include "lsh.h"
#include "mnist.h"
#include "sift.h"
#include "kmeans.h"
#include "silhouette.h"
#include "ivff.h"
#include "hypercube.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>

int main(int argc, char *argv[])
{
    /*ανάγνωση παραμέτρων από τη γραμμή εντολών*/
    utils::Params params = utils::parse_args(argc, argv);
    utils::print_params(params);

    /*φόρτωση δεδομένων*/
    std::vector<std::vector<float>> data;
    std::vector<std::vector<float>> queries;

    /*έλεγχος κατάληξης αρχείου*/
    auto ends_with = [](const std::string &s, const std::string &suf)
    {
        return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
    };

    if (params.type == "mnist")
    {
        data = load_mnist_images(params.input_file, 1000);
        queries = load_mnist_images(params.query_file, 10);
    }
    else if ((!params.input_file.empty() && (ends_with(params.input_file, ".dat") || ends_with(params.input_file, ".fvecs"))) ||
             (!params.query_file.empty() && (ends_with(params.query_file, ".dat") || ends_with(params.query_file, ".fvecs"))))
    {
        /*φόρτωση SIFT (fvecs-like LE: [int32 dim][dim * float32])*/
        int maxN = params.N > 0 ? params.N : -1;
        int maxQ = params.Q > 0 ? params.Q : -1;

        if (!params.input_file.empty())
            data = sift::load_sift_dat(params.input_file, maxN, 128);
        if (!params.query_file.empty())
            queries = sift::load_sift_dat(params.query_file, maxQ, 128);

        std::cout << "Loaded SIFT dataset: " << data.size()
                  << " x " << (data.empty() ? 0 : data[0].size());
        if (!queries.empty())
            std::cout << " | queries: " << queries.size();
        std::cout << "\n";
    }
    else
    {
        std::cerr << "Unsupported dataset type. Use MNIST or provide .dat/.fvecs files.\n";
        return 1;
    }

    /*δημιουργία αλγορίθμου αναζήτησης (LSH ή IVFFlat ή IVFPQ)*/
    std::ofstream out(params.output_file);
    if (!out.is_open())
    {
        std::cerr << "Error opening output file!\n";
        return 1;
    }
    out << std::fixed << std::setprecision(6);

    auto dim = static_cast<int>(data[0].size());
    /*εκτέλεση lsh*/
    if (params.use_lsh)
    {
        std::cout << "\n>> Using LSH index...\n";
        nn::LSH lsh(dim, params.L, params.k, params.w, params.seed);
        lsh.build_index(data);
        utils::run_queries(lsh, data, queries, params, out);
        lsh.clear_index();
    }

    /*εκτέλεση hypercube*/
    else if (params.use_hypercube)
    {
        std::cout << "\n>> Using Hypercube index...\n";
        int dim = data[0].size();
        hcube::Hypercube index(dim,
                               params.kproj,
                               params.w,
                               params.max_candidates,
                               params.max_probes,
                               params.seed,
                               utils::euclidean_distance);

        index.build_index(data);
        utils::run_queries(index, data, queries, params, out);
        index.clear_index();
    }

    /*εκτέλεση ivfflat*/
    else if (params.use_ivfflat)
    {
        std::cout << "\n>> Using IVFFlat index...\n";
        int dim = static_cast<int>(data[0].size());
        nn::IVFFlat index(
            dim,
            params.nlist,
            params.nprobe,
            params.seed,
            utils::euclidean_distance);

        index.build_index(data);
        utils::run_queries(index, data, queries, params, out);

        /*προαιρετικό: K-Means + Silhouette*/
        std::cout << "\nComputing Silhouette score for IVFFlat clusters...\n";

        clustering::KMeans model(
            3, 100, 1e-4,
            clustering::KMeans::InitMethod::KMEANS_PLUS_PLUS,
            42, true, utils::euclidean_distance);

        model.fit(data);

        clustering::Silhouette sil;
        float score = sil.compute(data, model.labels(), model.k());

        std::cout << "\nSilhouette score = " << score << std::endl;

        index.clear_index();
    }

    /*εκτέλεση ivfpq (προς υλοποίηση)
    else if (params.use_ivfpq) {
        std::cout << "\n>> Using IVFPQ index (placeholder)...\n";
        // μελλοντικά: nn::IVFPQ ivfpq(...)
        // ivfpq.build_index(data);
        // utils::run_queries(ivfpq, data, queries, params, out);
    }*/

    else
    {
        std::cerr << "Error: No index type specified! Use -lsh or -ivfflat or -ivfpq.\n";
        return 1;
    }

    out.close();

    return 0;
}
