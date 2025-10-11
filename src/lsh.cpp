#include "lsh.h"
#include "helper.h"

#include <iostream>
#include <vector>
#include <random>
#include <unordered_set>
#include <algorithm>

namespace nn {

using utils::euclidean_distance;

//κονστράκτορας με βασικές παραμέτρους
LSH::LSH(int dim, int L, int k, int w, unsigned int seed, utils::DistanceFunc dist_func)
    : dim_(dim), L_(L), k_(k), w_(w), seed_(seed), distance_func_(std::move(dist_func)) 
{   //αρχικοποίηση LSH παραμέτρων
    v_.assign(L_, std::vector<std::vector<float>>(k_, std::vector<float>(dim_, 0.0f)));
    t_.assign(L_, std::vector<float>(k_, 0.0f));
    r_.assign(L_, std::vector<uint32_t>(k_, 0));

    //default distance function
    if (!distance_func_)
        distance_func_ = euclidean_distance;

    //τυχαία διανύσματα v ∼ N(0,1)^d, t ∼ U[0,w), r ∼ [1,M-1]
    std::mt19937 rng(seed_);
    std::normal_distribution<float> normal_dist(0.0f, 1.0f);
    std::uniform_real_distribution<float> uniform_dist(0.0f, static_cast<float>(w_));
    std::uniform_int_distribution<uint32_t> int_dist(1, M_ - 1);

    //γέμισμα v, t, r
    for (int i = 0; i < L_; ++i) {
        for (int j = 0; j < k_; ++j) {
            r_[i][j] = int_dist(rng);
            for (int d = 0; d < dim_; ++d)
                v_[i][j][d] = normal_dist(rng);
            t_[i][j] = uniform_dist(rng);
        }
    }

    //αρχικοποίηση hash tables
    std::cout << "LSH initialized with dim=" << dim_
              << ", L=" << L_
              << ", k=" << k_
              << ", w=" << w_ << std::endl;
}

//h(p) = floor((p·v + t) / w)
std::vector<long long> LSH::compute_hashes_for_table(const std::vector<float>& p, int table_idx) const {
    std::vector<long long> hashes;
    hashes.reserve(k_);

    //υπολογισμός k hash values
    for (int j = 0; j < k_; ++j) {
        double dot = 0.0;
        for (int d = 0; d < dim_; ++d)
            dot += static_cast<double>(p[d]) * static_cast<double>(v_[table_idx][j][d]);

        double val = (dot + t_[table_idx][j]) / w_;
        hashes.push_back(static_cast<long long>(std::floor(val)));
    }
    return hashes;
}

//ID(p) = (Σ r_j * h_j(p)) mod M
uint64_t LSH::compute_id(const std::vector<long long>& hashes, int table_idx) const {
    uint64_t acc = 0;
    for (int j = 0; j < k_; j++) {
        long long hval = hashes[j] % static_cast<long long>(M_);
        if (hval < 0) hval += M_;
        acc = (acc + static_cast<uint64_t>(r_[table_idx][j]) * static_cast<uint64_t>(hval)) % M_;
    }
    return acc;
}

//g(p) = ID(p) mod TableSize
int LSH::compute_bucket_id(uint64_t id) const {
    return static_cast<int>(id % table_size_);
}

//χτίσιμο index
void LSH::build_index(const std::vector<std::vector<float>>& data) {
    data_ptr_ = &data;
    int n = static_cast<int>(data.size());
    table_size_ = std::max(1, n / 4);

    tables_.assign(L_, std::vector<std::vector<std::pair<uint64_t, int>>>(table_size_));
    //γέμισμα hash tables
    std::cout << "Building index on dataset of size " << n << std::endl;
    for (int id = 0; id < n; id++) {
        const auto& p = data[id];
        for (int i = 0; i < L_; i++) {
            auto hashes = compute_hashes_for_table(p, i);
            uint64_t obj_id = compute_id(hashes, i);
            int bucket = compute_bucket_id(obj_id);
            tables_[i][bucket].push_back({obj_id, id});
        }
    }
}

//αναζήτηση υποψηφίων
std::vector<int> LSH::query(const std::vector<float>& q, int num_neighbors) const {
    if (!data_ptr_) {
        std::cerr << "Error: index not built!" << std::endl;
        return {};
    }
    //συλλογή υποψηφίων
    std::unordered_set<int> candidates;
    //για κάθε πίνακα
    for (int i = 0; i < L_; ++i) {
        auto hashes = compute_hashes_for_table(q, i);
        uint64_t q_id = compute_id(hashes, i);
        int bucket = compute_bucket_id(q_id);

        for (const auto& [obj_id, idx] : tables_[i][bucket]) {
            if (obj_id == q_id)
                candidates.insert(idx);
        }
    }
    //υπολογισμός αποστάσεων και επιλογή των num_neighbors καλύτερων
    std::vector<std::pair<float, int>> dists;
    dists.reserve(candidates.size());
    for (int id : candidates)
        dists.emplace_back(distance_func_(q, (*data_ptr_)[id]), id);
    //επιλογή num_neighbors μικρότερων αποστάσεων
    if (static_cast<int>(dists.size()) > num_neighbors) {
        std::nth_element(dists.begin(), dists.begin() + num_neighbors, dists.end());
        dists.resize(num_neighbors);
        std::sort(dists.begin(), dists.end());
    } else {
        std::sort(dists.begin(), dists.end());
    }

    //επιστροφή μόνο των IDs
    std::vector<int> result;
    result.reserve(std::min(num_neighbors, static_cast<int>(dists.size())));
    for (int i = 0; i < num_neighbors && i < static_cast<int>(dists.size()); ++i)
        result.push_back(dists[i].second);

    return result;
}

//εκκαθάριση index
void LSH::clear_index() {
    for (auto& table : tables_)
        for (auto& bucket : table)
            bucket.clear();
    tables_.clear();
    std::cout << "Index cleared.\n";
}

//(α) Ένας πλησιέστερος γείτονας (ε-approximate NN)
int LSH::nn_query(const std::vector<float>& q, float epsilon) const {
    auto result = query(q, 1);
    if (result.empty()) return -1;
    
    int candidate = result[0];
    float best_dist = distance_func_(q, (*data_ptr_)[candidate]);
    //έλεγχος όλων των υποψηφίων για καλύτερο αποτέλεσμα
    for (int i = 0; i < static_cast<int>(data_ptr_->size()); i++) {
        if (i == candidate) continue;
        float dist = distance_func_(q, (*data_ptr_)[i]);
        if (dist < best_dist / (1 + epsilon)) {
            candidate = i;
            best_dist = dist;
        }
    }
    return candidate;
}

//(β) N πλησιέστεροι γείτονες
std::vector<std::pair<int, float>> LSH::knn_query(const std::vector<float>& q, int N) const {
    if (!data_ptr_) {
        std::cerr << "Error: index not built!" << std::endl;
        return {};
    }
    //συλλογή υποψηφίων
    std::unordered_set<int> candidates;
    //για κάθε πίνακα
    for (int i = 0; i < L_; ++i) {
        auto hashes = compute_hashes_for_table(q, i);
        uint64_t q_id = compute_id(hashes, i);
        int bucket = compute_bucket_id(q_id);
        //συλλογή υποψηφίων από το bucket
        for (const auto& [obj_id, idx] : tables_[i][bucket]) {
            if (obj_id == q_id)
                candidates.insert(idx);
        }
    }
    //υπολογισμός αποστάσεων και επιλογή των N καλύτερων
    std::vector<std::pair<float, int>> dists;
    dists.reserve(candidates.size());
    for (int id : candidates)
        dists.emplace_back(distance_func_(q, (*data_ptr_)[id]), id);

    if (dists.empty()) return {};
    //επιλογή N μικρότερων αποστάσεων
    if (static_cast<int>(dists.size()) > N) {
        std::nth_element(dists.begin(), dists.begin() + N, dists.end());
        dists.resize(N);
        std::sort(dists.begin(), dists.end());
    } else {
        std::sort(dists.begin(), dists.end());
    }
    //επιστροφή (ID, απόσταση)
    std::vector<std::pair<int, float>> result;
    result.reserve(dists.size());
    for (auto& [dist, id] : dists)
        result.emplace_back(id, dist);

    return result;
}

//(γ) Αναζήτηση εντός ακτίνας R
std::vector<int> LSH::range_search(const std::vector<float>& q, float R) const {
    if (!data_ptr_) {
        std::cerr << "Error: index not built!" << std::endl;
        return {};
    }
    //συλλογή υποψηφίων
    std::unordered_set<int> candidates;
    //για κάθε πίνακα
    for (int i = 0; i < L_; ++i) {
        auto hashes = compute_hashes_for_table(q, i);
        uint64_t q_id = compute_id(hashes, i);
        int bucket = compute_bucket_id(q_id);
        //συλλογή υποψηφίων από το bucket
        for (const auto& [obj_id, idx] : tables_[i][bucket]) {
            if (obj_id == q_id)
                candidates.insert(idx);
        }
    }
    //φιλτράρισμα υποψηφίων με απόσταση <= R
    std::vector<int> result;
    for (int id : candidates) {
        float dist = distance_func_(q, (*data_ptr_)[id]);
        if (dist <= R)
            result.push_back(id);
    }

    return result;
}

} //namespace nn
