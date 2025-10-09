#include "kmeans.h"
#include <iostream>
#include <limits>
#include <cmath>
#include <algorithm>
#include <unordered_set>

namespace clustering {

//κατασκευαστής της κλάσης KMeans
KMeans::KMeans(int k, int max_iters, float tol,
               InitMethod init, unsigned int seed, bool verbose)
    : k_(k),
      max_iters_(max_iters),
      tol_(tol),
      init_method_(init),
      seed_(seed),
      verbose_(verbose) {}


void KMeans::clear() {
    centroids_.clear();
    labels_.clear();
}

//υλοποίηση του αλγορίθμου k-means
float KMeans::squared_distance(const std::vector<float>& a,
                               const std::vector<float>& b) const {
    float dist = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return dist;
}


//τυχαία αρχικοποίηση των κεντροειδών από τα δεδομένα
void KMeans::init_random(const std::vector<std::vector<float>>& data,
                         std::mt19937& rng) {
    std::uniform_int_distribution<size_t> dist(0, data.size() - 1);
    std::unordered_set<size_t> chosen;

    centroids_.clear();
    while (centroids_.size() < static_cast<size_t>(k_)) {
        size_t idx = dist(rng);
        if (chosen.insert(idx).second) {
            centroids_.push_back(data[idx]);
        }
    }

    if (verbose_) {
        std::cout << "Initialized " << k_ << " centroids randomly." << std::endl;
    }
}

//k-means++ αρχικοποίηση των κεντροειδών
int KMeans::fit(const std::vector<std::vector<float>>& data) {
    if (data.empty()) return 0;
    size_t n = data.size();
    size_t d = data[0].size();
    std::mt19937 rng(seed_);

    //αρχικοποίηση κέντρων
    if (init_method_ == InitMethod::RANDOM)
        init_random(data, rng);
    else
        init_kmeans_pp(data, rng);

    labels_.assign(n, -1);
    std::vector<std::vector<float>> new_centroids(k_, std::vector<float>(d, 0.0f));
    std::vector<int> counts(k_, 0);

    int iter = 0;
    for (; iter < max_iters_; ++iter) {
        bool changed = false;

        //βήμα ανάθεσης
        for (size_t i = 0; i < n; ++i) {
            float best_dist = std::numeric_limits<float>::max();
            int best_cluster = -1;

            for (int c = 0; c < k_; ++c) {
                float dist = squared_distance(data[i], centroids_[c]);
                if (dist < best_dist) {
                    best_dist = dist;
                    best_cluster = c;
                }
            }

            if (labels_[i] != best_cluster) {
                labels_[i] = best_cluster;
                changed = true;
            }
        }

        //βήμα ενημέρωσης
        for (int c = 0; c < k_; ++c) {
            std::fill(new_centroids[c].begin(), new_centroids[c].end(), 0.0f);
            counts[c] = 0;
        }

        for (size_t i = 0; i < n; ++i) {
            int c = labels_[i];
            counts[c]++;
            for (size_t j = 0; j < d; ++j)
                new_centroids[c][j] += data[i][j];
        }

        for (int c = 0; c < k_; ++c) {
            if (counts[c] > 0) {
                for (size_t j = 0; j < d; ++j)
                    new_centroids[c][j] /= counts[c];
            } else {
                //αν κάποια ομάδα άδειασε, επανατοποθέτησε τυχαία
                std::uniform_int_distribution<size_t> dist(0, n - 1);
                new_centroids[c] = data[dist(rng)];
            }
        }

        //έλεγχος σύγκλισης
        float shift = 0.0f;
        for (int c = 0; c < k_; ++c)
            shift += squared_distance(centroids_[c], new_centroids[c]);

        if (verbose_) {
            std::cout << "Iteration " << iter + 1
                      << ": centroid shift = " << shift << std::endl;
        }

        centroids_ = new_centroids;

        if (shift < tol_) break; //σύγκλιση
        if (!changed) break; //δεν αλλάζουν οι ετικέτες
    }

    if (verbose_) {
        std::cout << "Converged after " << iter + 1 << " iterations." << std::endl;
    }

    return iter + 1;
}


} //namespace clustering
