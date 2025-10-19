#include "ivfpq.h"
#include <algorithm>
#include <iostream>
#include <numeric>
#include <cmath>
#include <unordered_set>
#include <queue>
#include <cassert>

namespace ivf
{

  ProductQuantizer::ProductQuantizer(int M, int nbits, utils::DistanceFunc dist)
      : M_(M), nbits_(nbits), Ks_(1 << nbits), dist_func_(dist)
  {
    if (M_ <= 0)
      M_ = 1;
    if (nbits_ <= 0)
    {
      nbits_ = 8;
      Ks_ = 256;
    }
  }

  void ProductQuantizer::set_dimension_splits(int D)
  {
    starts_.assign(M_, 0);
    sizes_.assign(M_, 0);

    int base = D / M_;
    int rem = D % M_;

    int start = 0;
    for (int m = 0; m < M_; ++m)
    {
      int sz = base + (m < rem ? 1 : 0);
      starts_[m] = start;
      sizes_[m] = sz;
      start += sz;
    }
  }

  void ProductQuantizer::train(const std::vector<std::vector<float>> &residuals)
  {
    if (residuals.empty())
      return;
    int D = static_cast<int>(residuals[0].size());
    set_dimension_splits(D);

    codebooks_.clear();
    codebooks_.resize(M_);

    std::mt19937 rng(1234567);
    /*εκπαίδευση ενός KMeans σε κάθε υποχώρο ανεξάρτητα*/
    for (int m = 0; m < M_; ++m)
    {
      int d_m = sizes_[m];
      int s_m = starts_[m];

      /*συλλογή δειγμάτων εκπαίδευσης προβαλλόμενων στον υποχώρο m*/
      std::vector<std::vector<float>> subtrain;
      subtrain.reserve(residuals.size());
      for (const auto &r : residuals)
      {
        std::vector<float> sub(d_m);
        for (int j = 0; j < d_m; ++j)
          sub[j] = r[s_m + j];
        subtrain.emplace_back(std::move(sub));
      }

      clustering::KMeans kmeans(Ks_, /*max_iters*/ 100, /*tol*/ 1e-4f,
                                clustering::KMeans::InitMethod::KMEANS_PLUS_PLUS,
                                /*seed*/ 1337 + m, /*verbose*/ false, dist_func_);
      /*σε πολύ μικρά σύνολα εκπαίδευσης, περιορίζουμε το Ks (αποφυγή κατάρρευσης λόγω κενών συσταδοποιήσεων)*/
      if ((int)subtrain.size() < Ks_)
      {
        /*εναλλακτική: αντιγραφή δειγμάτων ή μείωση του Ks*/
        int newKs = std::max(2, (int)subtrain.size());
        if (newKs < Ks_)
        {
          std::cerr << "[PQ] Reducing Ks in subspace " << m
                    << " to " << newKs << " (insufficient data)\n";
          /*μικρός KMeans με λιγότερα κεντροειδή*/
          clustering::KMeans small(newKs, 100, 1e-4f,
                                   clustering::KMeans::InitMethod::KMEANS_PLUS_PLUS,
                                   1337 + m, false, dist_func_);
          small.fit(subtrain);
          const auto &C = small.centroids();
          /*υπερδειγματοληψία έως Ks επαναλαμβάνοντας δείγματα & προσθέτοντας θόρυβο*/
          codebooks_[m].assign(Ks_, std::vector<float>(d_m, 0.0f));
          for (int k = 0; k < Ks_; ++k)
          {
            const auto &src = C[k % newKs];
            codebooks_[m][k] = src;
          }
          continue;
        }
      }

      kmeans.fit(subtrain);
      codebooks_[m] = kmeans.centroids();
      /*σε σπάνιες ιδιάζουσες περιπτώσεις, το kmeans μπορεί να επιστρέψει <Ks_ κεντροειδή>*/
      if ((int)codebooks_[m].size() < Ks_)
      {
        int have = codebooks_[m].size();
        codebooks_[m].resize(Ks_, std::vector<float>(d_m, 0.0f));
        for (int k = have; k < Ks_; ++k)
        {
          codebooks_[m][k] = codebooks_[m][k % have];
        }
      }
    }
  }

  std::vector<uint16_t> ProductQuantizer::encode(const std::vector<float> &residual) const
  {
    std::vector<uint16_t> code(M_, 0);
    for (int m = 0; m < M_; ++m)
    {
      int d_m = sizes_[m];
      int s_m = starts_[m];

      float best = std::numeric_limits<float>::max();
      int bestk = 0;
      for (int k = 0; k < Ks_; ++k)
      {
        const auto &c = codebooks_[m][k];
        float dist = 0.0f;
        for (int j = 0; j < d_m; ++j)
        {
          float diff = residual[s_m + j] - c[j];
          dist += diff * diff;
        }
        dist = std::sqrt(dist);
        if (dist < best)
        {
          best = dist;
          bestk = k;
        }
      }
      code[m] = static_cast<uint16_t>(bestk);
    }
    return code;
  }

  std::vector<std::vector<float>> ProductQuantizer::compute_LUT(const std::vector<float> &residual_q) const
  {
    std::vector<std::vector<float>> lut(M_, std::vector<float>(Ks_, 0.0f));
    for (int m = 0; m < M_; ++m)
    {
      int d_m = sizes_[m];
      int s_m = starts_[m];
      for (int k = 0; k < Ks_; ++k)
      {
        const auto &c = codebooks_[m][k];
        float dist = 0.0f;
        for (int j = 0; j < d_m; ++j)
        {
          float diff = residual_q[s_m + j] - c[j];
          dist += diff * diff;
        }
        lut[m][k] = std::sqrt(dist);
      }
    }
    return lut;
  }

  float ProductQuantizer::adc_distance(const std::vector<std::vector<float>> &lut,
                                       const std::vector<uint16_t> &code)
  {
    float d = 0.0f;
    for (size_t m = 0; m < code.size(); ++m)
      d += lut[m][code[m]];
    return d;
  }

  /*IVFPQ*/

  IVFPQ::IVFPQ(int kclusters, int nprobe, int M, int nbits,
               unsigned int seed, int N, float R, utils::DistanceFunc dist_func)
      : kclusters_(kclusters),
        nprobe_(nprobe),
        M_(M),
        nbits_(nbits),
        Ks_(1 << nbits),
        seed_(seed),
        N_(N),
        R_(R),
        dist_func_(dist_func ? dist_func : utils::euclidean_distance),
        pq_(M, nbits, dist_func)
  {
    std::cout << "IVFPQ αρχικοποιήθηκε με kclusters=" << kclusters_
              << ", nprobe=" << nprobe_
              << ", M=" << M_
              << ", nbits=" << nbits_
              << ", seed=" << seed_
              << ", N=" << N_
              << ", R=" << R_ << std::endl;
  }

  int IVFPQ::nearest_centroid(const std::vector<float> &x) const
  {
    float best = std::numeric_limits<float>::max();
    int bestk = 0;
    for (int k = 0; k < kclusters_; ++k)
    {
      float d = dist_func_(x, centroids_[k]);
      if (d < best)
      {
        best = d;
        bestk = k;
      }
    }
    return bestk;
  }

  std::vector<int> IVFPQ::top_nprobe_centroids(const std::vector<float> &q) const
  {
    std::vector<std::pair<float, int>> ds;
    ds.reserve(kclusters_);
    for (int k = 0; k < kclusters_; ++k)
      ds.emplace_back(dist_func_(q, centroids_[k]), k);

    int take = std::min(nprobe_, kclusters_);
    std::nth_element(ds.begin(), ds.begin() + take, ds.end());
    ds.resize(take);
    std::vector<int> ids;
    ids.reserve(take);
    for (auto &p : ds)
      ids.push_back(p.second);
    return ids;
  }

  std::vector<float> IVFPQ::residual_of(const std::vector<float> &x,
                                        const std::vector<float> &c)
  {
    std::vector<float> r(x.size());
    for (size_t i = 0; i < x.size(); ++i)
      r[i] = x[i] - c[i];
    return r;
  }

  void IVFPQ::build_index(const std::vector<std::vector<float>> &data)
  {
    if (data.empty())
      return;
    data_ptr_ = &data;
    size_t n = data.size();
    int D = static_cast<int>(data[0].size());

    /*1) επιλογή k μέσω silhouette αν kclusters_ <= 0 (όπως στο IVFFlat)*/
    int k_opt = kclusters_;
    if (k_opt <= 0)
    {
      int k_min = 2;
      int k_max = std::min<int>(10, std::sqrt(n));
      using namespace clustering;
      float best_score = -1.0f;
      int best_k = k_min;
      for (int k = k_min; k <= k_max; ++k)
      {
        KMeans kmeans(k, 100, 1e-4f, KMeans::InitMethod::KMEANS_PLUS_PLUS, seed_, false, dist_func_);
        kmeans.fit(data);
        Silhouette sil(dist_func_);
        float s = sil.compute(data, kmeans.labels(), k);
        if (s > best_score)
        {
          best_score = s;
          best_k = k;
        }
      }
      k_opt = best_k;
      std::cerr << "[IVFPQ] Selected k=" << k_opt << " via silhouette.\n";
    }
    kclusters_ = k_opt;

    /*2) k-means σε υποσύνολο για να ληφθούν κεντροειδή (χονδρικός κβαντιστής)*/
    size_t subset_size = std::max<size_t>(kclusters_, (size_t)std::sqrt(n));
    std::mt19937 rng(seed_);
    std::uniform_int_distribution<size_t> rnd(0, n - 1);
    std::unordered_set<size_t> pick;
    std::vector<std::vector<float>> subset;
    subset.reserve(subset_size);
    while (subset.size() < subset_size)
    {
      size_t i = rnd(rng);
      if (pick.insert(i).second)
        subset.push_back(data[i]);
    }

    clustering::KMeans km(kclusters_, 100, 1e-4f,
                          clustering::KMeans::InitMethod::KMEANS_PLUS_PLUS,
                          seed_, false, dist_func_);
    km.fit(subset);
    centroids_ = km.centroids();

    /*3) Προετοιμασία λιστών & συλλογή residuals για εκπαίδευση του PQ*/
    invlists_.assign(kclusters_, {});
    std::vector<std::vector<float>> residuals_for_pq;
    residuals_for_pq.reserve(subset_size);

    for (size_t i = 0; i < n; ++i)
    {
      int cid = nearest_centroid(data[i]);
      auto r = residual_of(data[i], centroids_[cid]);
      /*δειγματοληψία ορισμένων residuals για εκπαίδευση του PQ*/
      if (i % std::max<size_t>(1, n / (size_t)std::min(5000, (int)n)))
      {
        /*υποδειγματοληψία, διατήρηση επαρκούς πλήθους*/
      }
      residuals_for_pq.emplace_back(r);
    }

    /*προαιρετική υποδειγματοληψία residuals (αποφυγή τεράστιων KMeans ανά υποχώρο)*/
    if ((int)residuals_for_pq.size() > 20000)
    {
      std::vector<std::vector<float>> small;
      small.reserve(20000);
      std::unordered_set<size_t> seen;
      while (small.size() < 20000)
      {
        size_t idx = rnd(rng) % residuals_for_pq.size();
        if (seen.insert(idx).second)
          small.push_back(residuals_for_pq[idx]);
      }
      residuals_for_pq.swap(small);
    }

    /*4) Εκπαίδευση του παγκόσμιου residual PQ*/
    pq_.train(residuals_for_pq);

    /*5) Κωδικοποίηση residuals στις λίστες*/
    for (size_t i = 0; i < n; ++i)
    {
      int cid = nearest_centroid(data[i]);
      auto r = residual_of(data[i], centroids_[cid]);
      auto code = pq_.encode(r);
      invlists_[cid].push_back({static_cast<int>(i), std::move(code)});
    }

    std::cerr << "[IVFPQ] Δημιουργήθηκε index με " << kclusters_
              << " λίστες; PQ(M=" << M_ << ", nbits=" << nbits_ << ").\n";
  }

  std::vector<int> IVFPQ::query_candidates(const std::vector<float> &q) const
  {
    std::vector<int> ids;
    auto topc = top_nprobe_centroids(q);
    for (int cid : topc)
    {
      for (const auto &e : invlists_[cid])
        ids.push_back(e.id);
    }
    return ids;
  }

  std::vector<std::pair<int, float>> IVFPQ::knn_query(const std::vector<float> &q, int N) const
  {
    std::vector<std::pair<int, float>> out;
    if (!data_ptr_ || centroids_.empty())
      return out;

    /*για κάθε επιλεγμένη λίστα, υπολογίζουμε το residual του ερωτήματος, το LUT, και στη συνέχεια ADC*/
    struct Pair
    {
      float dist;
      int id;
    };
    std::vector<Pair> heap;
    heap.reserve(1024);

    auto topc = top_nprobe_centroids(q);
    for (int cid : topc)
    {
      auto rq = residual_of(q, centroids_[cid]);
      auto lut = pq_.compute_LUT(rq);
      for (const auto &e : invlists_[cid])
      {
        float d = ProductQuantizer::adc_distance(lut, e.code);
        heap.push_back({d, e.id});
      }
    }

    if (heap.empty())
      return out;

    /*διατήρηση των N μικρότερων*/
    if ((int)heap.size() > N)
    {
      std::nth_element(heap.begin(), heap.begin() + N, heap.end(),
                       [](const Pair &a, const Pair &b)
                       { return a.dist < b.dist; });
      heap.resize(N);
    }
    std::sort(heap.begin(), heap.end(),
              [](const Pair &a, const Pair &b)
              { return a.dist < b.dist; });

    out.reserve(heap.size());
    for (auto &p : heap)
      out.emplace_back(p.id, p.dist);
    return out;
  }

  std::vector<int> IVFPQ::range_search(const std::vector<float> &q, float R) const
  {
    std::vector<int> ids;
    if (!data_ptr_)
      return ids;

    auto topc = top_nprobe_centroids(q);
    for (int cid : topc)
    {
      auto rq = residual_of(q, centroids_[cid]);
      auto lut = pq_.compute_LUT(rq);
      for (const auto &e : invlists_[cid])
      {
        float d = ProductQuantizer::adc_distance(lut, e.code);
        if (d <= R)
          ids.push_back(e.id);
      }
    }
    return ids;
  }

  void IVFPQ::clear_index()
  {
    centroids_.clear();
    invlists_.clear();
    data_ptr_ = nullptr;
  }

} /*namespace ivf*/
