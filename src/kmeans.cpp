#include "kmeans.h"
#include <iostream>
#include <limits>
#include <cmath>
#include <algorithm>

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

} //namespace clustering
