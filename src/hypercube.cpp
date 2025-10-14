#include "hypercube.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <queue>
#include <unordered_set>

namespace nn
{

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
          bit_maps_(kproj_)
    {
        /*δημιουργία των προβολών v_ και των μετατοπίσεων t_*/
        for (int i = 0; i < kproj_; ++i)
        {
            for (int d = 0; d < dim_; ++d)
                v_[i][d] = normal_(rng_);
            t_[i] = uni_(rng_);
        }

        std::cout << "[Hypercube] Initialized:"
                  << " dim=" << dim_
                  << " kproj=" << kproj_
                  << " w=" << w_
                  << " M=" << M_
                  << " probes=" << probes_
                  << " seed=" << seed_
                  << std::endl;
    }

    /*δημιουργία του δείκτη (build)*/
    void Hypercube::build_index(const std::vector<std::vector<float>> &data)
    {
        data_ptr_ = &data;
        cube_.clear();

        const int n = static_cast<int>(data.size());
        for (int idx = 0; idx < n; ++idx)
        {
            const auto h = compute_hashes(data[idx]);
            const std::uint64_t vtx = compute_vertex_id(h);
            cube_[vtx].push_back(idx);
        }

        std::cout << "[Hypercube] build_index(): "
                  << "Inserted " << n << " points into "
                  << cube_.size() << " non-empty vertices.\n";
    }

    /*υπολογισμός hash τιμών*/
    std::vector<long long> Hypercube::compute_hashes(const std::vector<float> &p) const
    {
        std::vector<long long> h(kproj_);
        for (int i = 0; i < kproj_; ++i)
        {
            double dot = 0.0;
            for (int d = 0; d < dim_; ++d)
                dot += static_cast<double>(v_[i][d]) * static_cast<double>(p[d]);
            const double val = (dot + static_cast<double>(t_[i])) / static_cast<double>(w_);
            h[i] = static_cast<long long>(std::floor(val));
        }
        return h;
    }

    /*bit mapping*/
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

    /*υπολογισμός vertex id*/
    std::uint64_t Hypercube::compute_vertex_id(const std::vector<long long> &hashes) const
    {
        std::uint64_t bits = 0ULL;
        for (int i = 0; i < kproj_; ++i)
            if (bit_for(i, hashes[i]))
                bits |= (1ULL << i);
        return bits;
    }

    /*παραγωγή λίστας κορυφών για probing*/
    std::vector<std::uint64_t> Hypercube::vertices_to_probe(std::uint64_t base) const
    {
        std::vector<std::uint64_t> out;
        out.reserve(probes_ > 0 ? probes_ : 1);
        if (probes_ <= 0)
            return out;
        out.push_back(base);
        if (probes_ == 1)
            return out;

        std::queue<std::uint64_t> q;
        std::unordered_set<std::uint64_t> seen;
        q.push(base);
        seen.insert(base);

        while (!q.empty() && static_cast<int>(out.size()) < probes_)
        {
            const std::uint64_t cur = q.front();
            q.pop();
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

        if (static_cast<int>(out.size()) > probes_)
            out.resize(probes_);
        return out;
    }

    /*k-NN query*/
    std::vector<std::pair<int, float>> Hypercube::knn_query(const std::vector<float> &q, int N) const
    {
        std::vector<std::pair<int, float>> out;
        if (!data_ptr_ || data_ptr_->empty() || N <= 0)
            return out;

        const auto hq = compute_hashes(q);
        const auto vq = compute_vertex_id(hq);
        const auto list = vertices_to_probe(vq);

        std::vector<std::pair<float, int>> cand;
        cand.reserve(std::min(M_, static_cast<int>(data_ptr_->size())));
        int examined = 0;

        for (std::uint64_t vtx : list)
        {
            auto it = cube_.find(vtx);
            if (it == cube_.end())
                continue;
            for (int idx : it->second)
            {
                float d = dist_func_(q, (*data_ptr_)[idx]);
                cand.emplace_back(d, idx);
                ++examined;
                if (examined >= M_)
                    break;
            }
            if (examined >= M_)
                break;
        }

        if (cand.empty())
        {
            std::cout << "[Hypercube::knn_query] No candidates found.\n";
            return out;
        }

        const int take = std::min(N, static_cast<int>(cand.size()));
        std::nth_element(cand.begin(), cand.begin() + take, cand.end(),
                         [](auto &a, auto &b)
                         { return a.first < b.first; });
        cand.resize(take);
        std::sort(cand.begin(), cand.end(),
                  [](auto &a, auto &b)
                  { return a.first < b.first; });

        out.reserve(cand.size());
        for (const auto &p : cand)
            out.emplace_back(p.second, p.first);

        std::cout << "[Hypercube::knn_query] Examined " << examined
                  << " candidates across " << list.size()
                  << " vertices; returned top " << out.size() << ".\n";

        return out;
    }

    /*range search*/
    std::vector<int> Hypercube::range_search(const std::vector<float> &q, float R) const
    {
        std::vector<int> res;
        if (!data_ptr_ || data_ptr_->empty() || R < 0.0f)
            return res;

        const auto hq = compute_hashes(q);
        const auto vq = compute_vertex_id(hq);
        const auto list = vertices_to_probe(vq);

        int examined = 0;
        for (std::uint64_t vtx : list)
        {
            auto it = cube_.find(vtx);
            if (it == cube_.end())
                continue;
            for (int idx : it->second)
            {
                float d = dist_func_(q, (*data_ptr_)[idx]);
                if (d <= R)
                    res.push_back(idx);
                ++examined;
                if (examined >= M_)
                    break;
            }
            if (examined >= M_)
                break;
        }

        std::cout << "[Hypercube::range_search] Found " << res.size()
                  << " points within R=" << R
                  << " after checking " << examined
                  << " candidates.\n";

        return res;
    }

    /*καθαρισμός όλων των πόρων*/
    void Hypercube::clear_index()
    {
        std::cout << "[Hypercube] Clearing index and releasing resources...\n";
        cube_.clear();
        bit_maps_.clear();
        v_.clear();
        t_.clear();
        data_ptr_ = nullptr;
        std::cout << "[Hypercube] Resources successfully released.\n";
    }

} /*namespace nn*/
