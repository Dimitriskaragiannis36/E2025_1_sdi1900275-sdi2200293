#include "ivff.h"
#include <algorithm>
#include <iostream>
#include <cmath>
#include <numeric>
#include <queue>

namespace ivf {

//κονστράκτορας
IVFFlat::IVFFlat(int nlist, int nprobe, unsigned int seed,
                 utils::DistanceFunc dist_func)
    : nlist_(nlist),
      nprobe_(nprobe),
      seed_(seed),
      dist_func_(::std::move(dist_func)),
      data_ptr_(nullptr) {}

//υπολογισμός τετραγωνικής απόστασης
float IVFFlat::distance_sq(const std::vector<float>& a, const std::vector<float>& b) const {
    float d = dist_func_(a, b);
    return d * d;
}

//επιλογή καλύτερου k με χρήση Silhouette
static int select_best_k_by_silhouette(
    const std::vector<std::vector<float>>& data,
    int k_min, int k_max,
    utils::DistanceFunc dist_func,
    unsigned int seed)
{
    using namespace clustering;
    float best_score = -1.0f;
    int best_k = k_min;

    for (int k = k_min; k <= k_max; ++k) {
        KMeans kmeans(k, 100, 1e-4f,
                      KMeans::InitMethod::KMEANS_PLUS_PLUS,
                      seed, false, dist_func);
        kmeans.fit(data);

        Silhouette sil(dist_func);
        float score = sil.compute(data, kmeans.labels(), k);

        std::cerr << "[Silhouette] k=" << k
                  << " score=" << score << std::endl;

        if (score > best_score) {
            best_score = score;
            best_k = k;
        }
    }

    std::cerr << "[Silhouette] Best k=" << best_k
              << " (score=" << best_score << ")\n";
    return best_k;
}

//κατασκευή του index
void IVFFlat::build_index(const std::vector<std::vector<float>>& data) {
    if (data.empty()) return;
    data_ptr_ = &data;
    std::size_t n = data.size();

    //1) Επιλογή του αριθμού συστάδων k με silhouette (αν nlist_ <= 0)
    int k_opt = nlist_;
    if (nlist_ <= 0) {
        int k_min = 2;
        int k_max = std::min<int>(10, std::sqrt(n)); // μην το παρακάνουμε
        std::cerr << "[IVF] Selecting best k via Silhouette in range ["
                  << k_min << "," << k_max << "]...\n";

        using namespace clustering;
        float best_score = -1.0f;
        int best_k = k_min;

        for (int k = k_min; k <= k_max; ++k) {
            KMeans kmeans(k, 100, 1e-4f,
                          KMeans::InitMethod::KMEANS_PLUS_PLUS,
                          seed_, false, dist_func_);
            kmeans.fit(data);

            Silhouette sil(dist_func_);
            float score = sil.compute(data, kmeans.labels(), k);

            std::cerr << "[Silhouette] k=" << k
                      << " score=" << score << std::endl;

            if (score > best_score) {
                best_score = score;
                best_k = k;
            }
        }

        k_opt = best_k;
        std::cerr << "[Silhouette] Best k=" << k_opt
                  << " (score=" << best_score << ")\n";
    }

    //2) Επιλογή υποσυνόλου X'
    std::size_t subset_size = static_cast<std::size_t>(std::sqrt(n));
    if (subset_size < static_cast<std::size_t>(k_opt))
        subset_size = k_opt;

    std::vector<std::vector<float>> subset;
    subset.reserve(subset_size);

    std::mt19937 rng(seed_);
    std::uniform_int_distribution<std::size_t> dist_idx(0, n - 1);
    std::unordered_set<std::size_t> picked;

    while (subset.size() < subset_size) {
        std::size_t idx = dist_idx(rng);
        if (picked.insert(idx).second)
            subset.push_back(data[idx]);
    }

    //3) Εκτέλεση K-Means με το βέλτιστο k
    clustering::KMeans kmeans(k_opt, 100, 1e-4f,
                              clustering::KMeans::InitMethod::KMEANS_PLUS_PLUS,
                              seed_, false, dist_func_);
    kmeans.fit(subset);
    centroids_ = kmeans.centroids();
    nlist_ = k_opt;

    //4) Ανάθεση σημείων στο κοντινότερο centroid
    inverted_lists_.assign(nlist_, {});
    for (std::size_t i = 0; i < n; ++i) {
        const auto& x = data[i];
        float bestd = std::numeric_limits<float>::max();
        int bestk = -1;
        for (int k = 0; k < nlist_; ++k) {
            float d = distance_sq(x, centroids_[k]);
            if (d < bestd) {
                bestd = d;
                bestk = k;
            }
        }
        inverted_lists_[bestk].push_back(static_cast<int>(i));
    }

    std::cerr << "[IVF] Built index with " << nlist_
              << " clusters (chosen by silhouette)." << std::endl;
}

//εύρεση υποψηφίων (βήμα coarse search)
std::vector<int> IVFFlat::query_candidates(const std::vector<float>& q) const {
    std::vector<std::pair<float,int>> centroid_dists;
    centroid_dists.reserve(nlist_);
    for (int k = 0; k < nlist_; ++k) {
        float d = distance_sq(q, centroids_[k]);
        centroid_dists.emplace_back(d, k);
    }

    //κρατάμε τα nprobe κοντινότερα centroids
    std::nth_element(centroid_dists.begin(),
                     centroid_dists.begin() + std::min(nprobe_, nlist_),
                     centroid_dists.end());
    centroid_dists.resize(std::min(nprobe_, nlist_));

    //ενώνουμε τα inverted lists από τα πιο κοντινά centroids
    std::vector<int> candidates;
    for (auto& [_, cid] : centroid_dists)
        for (int idx : inverted_lists_[cid])
            candidates.push_back(idx);

    return candidates;
}

//αναζήτηση Ν κοντινότερων γειτόνων
std::vector<std::pair<int,float>> IVFFlat::knn_query(const std::vector<float>& q, int N) const {
    std::vector<std::pair<int,float>> results;
    if (!data_ptr_ || centroids_.empty()) return results;

    auto candidates = query_candidates(q);
    if (candidates.empty()) return results;

    std::vector<std::pair<float,int>> dists;
    dists.reserve(candidates.size());
    for (int idx : candidates) {
        float d = dist_func_(q, (*data_ptr_)[idx]);
        dists.emplace_back(d, idx);
    }

    if (dists.size() > static_cast<std::size_t>(N))
        std::nth_element(dists.begin(), dists.begin() + N, dists.end());
    else
        N = static_cast<int>(dists.size());

    std::sort(dists.begin(), dists.begin() + N);

    for (int i = 0; i < N; ++i)
        results.emplace_back(dists[i].second, dists[i].first);

    return results;
}

//range search
std::vector<int> IVFFlat::range_search(const std::vector<float>& q, float R) const {
    std::vector<int> result;
    if (!data_ptr_) return result;

    auto candidates = query_candidates(q);
    for (int idx : candidates) {
        float d = dist_func_(q, (*data_ptr_)[idx]);
        if (d <= R)
            result.push_back(idx);
    }
    return result;
}

//καθαρισμός
void IVFFlat::clear() {
    centroids_.clear();
    inverted_lists_.clear();
    data_ptr_ = nullptr;
}

} //namespace ivf
