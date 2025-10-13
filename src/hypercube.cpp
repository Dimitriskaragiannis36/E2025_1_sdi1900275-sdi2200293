#include "hypercube.h"
#include <iostream>
#include <cmath>
#include <algorithm>

namespace nn
{

    /*mapping hash -> bit και υπολογισμός vertex id*/
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
          data_ptr_(nullptr),
          bit_maps_(kproj_) /*k' ανεξάρτητοι πίνακες hash -> bit*/
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

        std::cout << "Hypercube ctor (commit 5): "
                  << "dim=" << dim_
                  << " kproj=" << kproj_
                  << " w=" << w_
                  << " M=" << M_
                  << " probes=" << probes_
                  << " seed=" << seed_
                  << " [hash→bit mapping & vertex id enabled]\n";
    }

    /*σύνδεση dataset*/
    void Hypercube::build_index(const std::vector<std::vector<float>> &data)
    {
        data_ptr_ = &data;
        std::cout << "[Hypercube] build_index(): dataset attached (commit 5)\n";
    }

    /*υπολογισμός των k' ακέραιων hash τιμών για διάνυσμα p*/
    /*h_i(p) = floor( (v_i · p + t_i) / w )*/
    std::vector<long long> Hypercube::compute_hashes(const std::vector<float> &p) const
    {
        std::vector<long long> h(kproj_);
        for (int i = 0; i < kproj_; ++i)
        {
            double dot = 0.0;
            for (int d = 0; d < dim_; ++d)
            {
                dot += static_cast<double>(v_[i][d]) * static_cast<double>(p[d]);
            }
            const double val = (dot + static_cast<double>(t_[i])) / static_cast<double>(w_);
            h[i] = static_cast<long long>(std::floor(val));
        }
        return h;
    }

    /*επιστρέφει bit ∈ {0,1} για (proj_id, hval) με συνεπή, ντετερμινιστική ανάθεση*/
    /*χρησιμοποιούμε ελαφρύ 64-bit mix ώστε η αντιστοίχιση να είναι σταθερή και "τυχαία".*/
    int Hypercube::bit_for(int proj_id, long long hval) const
    {
        /*προσπέλαση του πίνακα για την προβολή proj_id (δημιουργείται lazy)*/
        auto &mp = const_cast<std::unordered_map<long long, int> &>(bit_maps_[proj_id]);
        auto it = mp.find(hval);
        if (it != mp.end())
            return it->second;

        /*64-bit μείξη (xorshifted-murmur-like) και χρήση του ελάχιστου bit*/
        std::uint64_t x = static_cast<std::uint64_t>(hval);
        x ^= x >> 33;
        x *= 0xff51afd7ed558ccdULL;
        x ^= x >> 33;
        x *= 0xc4ceb9fe1a85ec53ULL;
        x ^= x >> 33;
        const int bit = static_cast<int>(x & 1ULL);

        mp.emplace(hval, bit);
        return bit;
    }

    /*συσκευάζει τα k' bits σε 64-bit αναγνωριστικό κορυφής.
    bit i -> θέση i στο αποτέλεσμα. Υπόθεση: kproj_ <= 64.*/
    std::uint64_t Hypercube::compute_vertex_id(const std::vector<long long> &hashes) const
    {
        std::uint64_t bits = 0ULL;
        for (int i = 0; i < kproj_; ++i)
        {
            const int b = bit_for(i, hashes[i]);
            if (b)
                bits |= (1ULL << i);
        }
        return bits;
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
