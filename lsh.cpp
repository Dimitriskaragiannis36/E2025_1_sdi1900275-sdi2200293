#include "lsh.h"
#include <iostream>
#include <vector>
#include <random>
using namespace std;

namespace nn {

LSH::LSH(int dim, int L, int k, float w, unsigned int seed)
    : dim_(dim), L_(L), k_(k), w_(w), seed_(seed) {
    tables_.resize(L_);
    v_.assign(L_, vector<vector<float>>(k_, vector<float>(dim_, 0.0f)));
    t_.assign(L_, vector<float>(k_, 0.0f));


    //random generators 
    mt19937 rng(seed_); 
    normal_distribution<float> normal_dist(0.0f, 1.0f); // N(0,1) 
    uniform_real_distribution<float> uniform_dist(0.0f, w_); // U[0,w)
    
    //γέμισμα v_ and t_ 
    for (int i = 0; i < L_; ++i) { 
        for (int j = 0; j < k_; ++j) { 
            for (int d = 0; d < dim_; ++d) { 
                v_[i][j][d] = normal_dist(rng); 
            } 
            t_[i][j] = uniform_dist(rng); 
        } 
    }

    cout << "LSH initialized with dim=" << dim_
              << ", L=" << L_
              << ", k=" << k_
              << ", w=" << w_
              << endl;
}

//h(p) = floor((p·v + t) / w) για κάθε hash function
vector<long long> LSH::compute_hashes_for_table(const vector<float>& p, int table_idx) const {
    vector<long long> hashes;
hashes.reserve(k_);

for (int j = 0; j < k_; ++j) {
    double dot = 0.0;
    for (int d = 0; d < dim_; ++d) {
        dot += static_cast<double>(p[d]) * static_cast<double>(v_[table_idx][j][d]);
    }
    double val = (dot + t_[table_idx][j]) / w_;
    long long h = static_cast<long long>(floor(val));
    hashes.push_back(h);
}

return hashes;
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
