#include "hypercube.h"
#include <iostream>
#include <cmath>         // floor
#include <algorithm>     // (μελλοντικά για nth_element/sort)
#include <queue>         // BFS
#include <unordered_set> // αποφυγή διπλοεπίσκεψης

namespace nn
{

    /*προσθήκη vertices_to_probe() για BFS σε χώρο Hamming.*/
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

        std::cout << "Hypercube ctor (commit 6): "
                  << "dim=" << dim_
                  << " kproj=" << kproj_
                  << " w=" << w_
                  << " M=" << M_
                  << " probes=" << probes_
                  << " seed=" << seed_
                  << " [vertex probing ready]\n";
    }

    /*σύνδεση dataset (δεν χτίζουμε δομή ακόμα)*/
    void Hypercube::build_index(const std::vector<std::vector<float>> &data)
    {
        data_ptr_ = &data;
        std::cout << "[Hypercube] build_index(): dataset attached (commit 6)\n";
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

    /*επιστροφή bit ∈ {0,1} για (proj_id, hval) μεντετερμινιστική ανάθεση.*/
    int Hypercube::bit_for(int proj_id, long long hval) const
    {
        auto &mp = const_cast<std::unordered_map<long long, int> &>(bit_maps_[proj_id]);
        auto it = mp.find(hval);
        if (it != mp.end())
            return it->second;

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
    bit i -> θέση i στο αποτέλεσμα. Υπόθεση: kproj_ ≤ 64.*/
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

    /*επιστρέφει μέχρι 'probes_' κορυφές ξεκινώντας από τη βάση 'base'.
    Διασχίζει τον χώρο κορυφών με BFS παράγοντας γείτονες Hamming-1.
    Η σειρά επιστροφής ακολουθεί αύξουσα απόσταση Hamming (base πρώτα).*/
    std::vector<std::uint64_t> Hypercube::vertices_to_probe(std::uint64_t base) const
    {
        std::vector<std::uint64_t> out;
        out.reserve(probes_ > 0 ? probes_ : 1);

        /*αν το πλήθος ζητούμενων κορυφών είναι 0, δεν χρειάζεται δουλειά.*/
        if (probes_ <= 0)
            return out;

        /*τοποθετούμε την αρχική κορυφή*/
        out.push_back(base);
        if (probes_ == 1)
            return out;

        /*BFS με ουρά και αποφυγή διπλοεπίσκεψης*/
        std::queue<std::uint64_t> q;
        std::unordered_set<std::uint64_t> seen;
        q.push(base);
        seen.insert(base);

        while (!q.empty() && static_cast<int>(out.size()) < probes_)
        {
            const std::uint64_t cur = q.front();
            q.pop();

            /*γεννάμε όλους τους γείτονες Hamming-1 (flip κάθε bit 0..kproj_-1)*/
            for (int i = 0; i < kproj_; ++i)
            {
                const std::uint64_t nei = cur ^ (1ULL << i);
                if (seen.insert(nei).second)
                {
                    out.push_back(nei);
                    if (static_cast<int>(out.size()) >= probes_)
                        break;
                    q.push(nei);
                }
            }
        }

        /*σε περίπτωση υπερχείλισης (θεωρητικά), περικόπτουμε στα probes_*/
        if (static_cast<int>(out.size()) > probes_)
        {
            out.resize(probes_);
        }
        return out;
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
