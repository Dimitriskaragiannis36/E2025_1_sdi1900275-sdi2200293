#include "helper.h" //για parse_args και print_params
#include "lsh.h" //για LSH
#include "mnist.h" //για load_mnist_images
#include "sift.h" //για load_sift_dat
#include "kmeans.h" //για KMeans
#include "ivff.h" //για IVFFlat
#include "hypercube.h" //για Hypercube
#include "ivfpq.h" //για IVFPQ
#include <iostream> //για είσοδο/έξοδο
#include <fstream> //std::ofstream
#include <iomanip> //std::setprecision

int main(int argc, char *argv[])
{
    /*ανάγνωση παραμέτρων από τη γραμμή εντολών*/
    utils::Params params = utils::parse_args(argc, argv);
    utils::print_params(params);
    /*φόρτωση δεδομένων*/
    std::vector<std::vector<float>> data;
    std::vector<std::vector<float>> queries;

    if (params.type == "mnist")
    {
        data = load_mnist_images(params.input_file, 1000); //φόρτωση έως 1000 εικόνες MNIST
        queries = load_mnist_images(params.query_file, 10); //φόρτωση έως 10 ερωτήματα
    }
    else if (params.type == "sift")
    {
        /*δεν περιορίζουμε την ανάγνωση (όλα τα vectors) — βάλτους όριο αν θέλεις*/
        int maxN = 1000; //π.χ. φόρτωση έως 1000 διανύσματα
        int maxQ = 10; //π.χ. φόρτωση έως 10 ερωτήματα

        if (!params.input_file.empty())
        {
            data = sift::load_sift_dat(params.input_file, maxN, 128); //φόρτωση SIFT δεδομένων
        }
        if (!params.query_file.empty())
        {
            queries = sift::load_sift_dat(params.query_file, maxQ, 128); //φόρτωση SIFT ερωτημάτων
        }

        std::cout << "Loaded SIFT dataset: " << data.size()
                  << " vectors x " << (data.empty() ? 0 : data[0].size()) << " dims";
        if (!queries.empty())
            std::cout << " | queries: " << queries.size(); //αριθμός ερωτημάτων
        std::cout << "\n";
    }
    else
    {
        std::cerr << "Unknown dataset type! Use -type mnist or -type sift (or provide appropriate filenames).\n";
        return 1;
    }
    /*δημιουργία αλγορίθμου αναζήτησης (LSH ή IVFFlat ή IVFPQ)*/
    std::ofstream out(params.output_file);
    if (!out.is_open())
    {
        std::cerr << "Error opening output file!\n";
        return 1;
    }
    out << std::fixed << std::setprecision(6); //6 δεκαδικά ψηφία στην έξοδο

    auto dim = static_cast<int>(data[0].size());
    /*εκτέλεση lsh*/
    if (params.use_lsh)
    {
        std::cout << "\n>> Using LSH index...\n";
        nn::LSH lsh(dim, params.L, params.k, params.w, params.seed); //δημιουργία LSH αντικειμένου
        lsh.build_index(data); //κατασκευή ευρετηρίου
        utils::run_queries(lsh, data, queries, params, out); //εκτέλεση ερωτημάτων
        lsh.clear_index(); //καθαρισμός ευρετηρίου
    }
    /*εκτέλεση hypercube*/
    else if (params.use_hypercube)
    {
        std::cout << "\n>> Using Hypercube index...\n";
        int dim = data[0].size(); //διάσταση δεδομένων
        hcube::Hypercube index(dim,
                               params.kproj,
                               params.w,
                               params.max_candidates,
                               params.max_probes,
                               params.seed,
                               utils::euclidean_distance); //δημιουργία Hypercube αντικειμένου

        index.build_index(data); //κατασκευή ευρετηρίου
        utils::run_queries(index, data, queries, params, out); //εκτέλεση ερωτημάτων
        index.clear_index(); //καθαρισμός ευρετηρίου
    }
    /*εκτέλεση ivfflat*/
    else if (params.use_ivfflat)
    {
        std::cout << "\n>> Using IVFFlat index...\n";
        ivf::IVFFlat index(params.kclusters, params.nprobe, params.seed, params.N, params.R); //δημιουργία IVFFlat αντικειμένου

        index.build_index(data); //κατασκευή ευρετηρίου
        utils::run_queries(index, data, queries, params, out); //εκτέλεση ερωτημάτων
        index.clear_index(); //καθαρισμός ευρετηρίου
    }
    /*εκτέλεση ivfpq*/
    else if (params.use_ivfpq)
    {
        std::cout << "\n>> Using IVFPQ index...\n";
        ivf::IVFPQ index(params.kclusters,
                         params.nprobe,
                         params.pq_M,
                         params.pq_nbits,
                         params.seed, params.N, params.R); //δημιουργία IVFPQ αντικειμένου

        index.build_index(data); //κατασκευή ευρετηρίου
        utils::run_queries(index, data, queries, params, out); //εκτέλεση ερωτημάτων
        index.clear_index(); //καθαρισμός ευρετηρίου
    }
    else
    {
        std::cerr << "Error: No index type specified! Use -lsh or -ivfflat or -ivfpq.\n";
        return 1; //σφάλμα
    }

    out.close();
    return 0;
}