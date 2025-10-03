#include "lsh.h"
#include <iostream>
#include <vector>
using namespace std;
namespace nn {

LSH::LSH(int dim, int L, int k, float w, unsigned int seed)
    : dim_(dim), L_(L), k_(k), w_(w), seed_(seed) {
    //placeholder constructor
    tables_.resize(L_);
    cout << "LSH initialized with dim=" << dim_
              << ", L=" << L_
              << ", k=" << k_
              << ", w=" << w_
              << endl;
}

void LSH::build_index(const vector<vector<float>>& data) {
    data_ptr_ = &data;
    //εκκρεμεί hash function and bucket γέμισμα
    cout << "Building index on dataset of size " << data.size() << endl;
}

vector<int> LSH::query(const vector<float>& q, int num_neighbors) const {
    //εκκρεμεί LSH query λογική
    cout << "Querying for nearest neighbors (k=" << num_neighbors << ")" << endl;
    return {}; //εκκρεμεί
}

void LSH::clear_index() {
    for (auto& table : tables_) {
        table.clear();
    }
    cout << "Index cleared." << endl;
}

} //namespace nn
