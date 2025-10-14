#include "helper.h"
#include "mnist.h"
#include "kmeans.h"
#include "lsh.h"
#include "ivff.h"
#include "hypercube.h" // ΝΕΟ: για το A2
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
    int dim = 0;

    if (params.data_type == "mnist")
    {
        data = mnist::load_data(params.input_file, dim);
        queries = mnist::load_data(params.query_file, dim);
    }
    else if (params.data_type == "sift")
    {
        data = mnist::load_sift(params.input_file, dim);
        queries = mnist::load_sift(params.query_file, dim);
    }
    else
    {
        std::cerr << "Unknown data type: " << params.data_type << std::endl;
        return 1;
    }

    std::cout << "Loaded dataset: " << data.size()
              << " vectors, dim=" << dim << "\n";
    std::cout << "Loaded queries : " << queries.size() << "\n";

    /*A1: LSH*/
    if (params.use_lsh)
    {
        std::cout << "\n>> Using LSH index...\n";
        nn::LSH lsh(dim, params.k, params.L, params.w,
                    params.seed, utils::euclidean_distance);
        lsh.build_index(data);
        utils::run_queries(lsh, data, queries, params, params.output_file);
        lsh.clear_index();
    }

    /*A2: Hypercube*/
    else if (params.use_hypercube)
    {
        std::cout << "\n>> Using Hypercube index...\n";
        nn::Hypercube cube(dim, params.kproj, params.w, params.M_cap,
                           params.probes, params.seed,
                           utils::euclidean_distance);
        cube.build_index(data);
        utils::run_queries(cube, data, queries, params, params.output_file);
        cube.clear_index();
    }

    /*A3: IVFFlat*/
    else if (params.use_ivfflat)
    {
        std::cout << "\n>> Using IVFFlat index...\n";
        nn::IVFFlat ivf(dim, params.kclusters, params.nprobe,
                        params.seed, utils::euclidean_distance);
        ivf.build_index(data);
        utils::run_queries(ivf, data, queries, params, params.output_file);
        ivf.clear_index();
    }

    else
    {
        std::cerr << "No algorithm selected! Use -lsh, -hypercube, or -ivfflat.\n";
        return 1;
    }

    std::cout << "\nAll queries completed successfully.\n";
    return 0;
}
