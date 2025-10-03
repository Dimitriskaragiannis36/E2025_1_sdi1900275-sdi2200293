#include "lsh.h"
#include <iostream>

namespace nn {

LSH::LSH(int dim, int L, int k, float w, unsigned int seed)
    : dim_(dim), L_(L), k_(k), w_(w), seed_(seed) {
    //placeholder constructor
    tables_.resize(L_);
    std::cout << "LSH initialized with dim=" << dim_
              << ", L=" << L_
              << ", k=" << k_
              << ", w=" << w_
              << std::endl;
}

void LSH::build_index(const std::vector<std::vector<float>>& data) {
    data_ptr_ = &data;
    //εκκρεμεί hash function and bucket γέμισμα
    std::cout << "Building index on dataset of size " << data.size() << std::endl;
}

std::vector<int> LSH::query(const std::vector<float>& q, int num_neighbors) const {
    //εκκρεμεί LSH query λογική
    std::cout << "Querying for nearest neighbors (k=" << num_neighbors << ")" << std::endl;
    return {}; //εκκρεμεί
}

void LSH::clear_index() {
    for (auto& table : tables_) {
        table.clear();
    }
    std::cout << "Index cleared." << std::endl;
}

} //namespace nn
