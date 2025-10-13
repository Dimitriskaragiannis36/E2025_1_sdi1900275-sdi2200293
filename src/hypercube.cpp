#include "hypercube.h"
#include <iostream>
#include <cmath> // floor

namespace nn
{

    /*προσθήκη υπολογισμού h_i(p) = floor((v_i·p + t_i)/w)*/
    Hypercube::Hypercube(int dim,
                         int kproj,
                         int w,
                         int M,
                         int probes,
                         unsigned int seed,
                         utils::DistanceFunc dist_func)
        : dim_(dim),
          kproj_(kproj),
          w_(w),
          M_(M),
          probes_(probes),
          seed_(seed),
          dist_func_(dist_func ? dist_func : utils::euclidean_distance),
          rng_(seed_),
          normal_(0.0f, 1.0f),
          uni_(0.0f, static_cast<float>(w)),
          v_(kproj_, std::vector<float>(dim_, 0.0f)),
          t_(kproj_, 0.0f),
          data_ptr_(nullptr)
    {
        /*γέμισμα προβολών v_ με N(0,1) και μετατοπίσεων t_ με U(0,w)*/
        for (int i = 0; i < kproj_; ++i)
        {
            for (int d = 0; d < dim_; ++d)
            {
                v_[i][d] = normal_(rng_);
            }
            t_[i] = uni_(rng_);
        }

        std::cout << "Hypercube ctor (commit 4): "
                  << "dim=" << dim_
                  << " kproj=" << kproj_
                  << " w=" << w_
                  << " M=" << M_
                  << " probes=" << probes_
                  << " seed=" << seed_
                  << " [projections & shifts ready, hashes enabled]\n";
    }

    /*σύνδεση dataset*/
    void Hypercube::build_index(const std::vector<std::vector<float>> &data)
    {
        data_ptr_ = &data;
        std::cout << "[Hypercube] build_index(): dataset attached (commit 4)\n";
    }

    /*υπολογισμός των k' ακέραιων hash τιμών για διάνυσμα p*/
    /*h_i(p) = floor( (v_i · p + t_i) / w )*/
    std::vector<long long> Hypercube::compute_hashes(const std::vector<float> &p) const
    {
        std::vector<long long> h(kproj_);
        for (int i = 0; i < kproj_; ++i)
        {
            double dot = 0.0;
            /*υπολογισμός εσωτερικού γινομένου v_i · p*/
            for (int d = 0; d < dim_; ++d)
            {
                dot += static_cast<double>(v_[i][d]) * static_cast<double>(p[d]);
            }
            /*εφαρμογή μετατόπισης και κανονικοποίηση με w*/
            const double val = (dot + static_cast<double>(t_[i])) / static_cast<double>(w_);
            h[i] = static_cast<long long>(std::floor(val));
        }
        return h;
    }

    std::vector<std::pair<int, float>> Hypercube::knn_query(const std::vector<float> &, int) const
    {
        return {};
    }

    std::vector<int> Hypercube::range_search(const std::vector<float> &, float) const
    {
        return {};
    }

    void Hypercube::clear_index()
    {
        std::cout << "[Hypercube] clear_index():\n";
    }

} /*namespace nn*/