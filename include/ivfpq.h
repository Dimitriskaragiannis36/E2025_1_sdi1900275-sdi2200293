#ifndef IVFPQ_H
#define IVFPQ_H

#include "helper.h"
#include "kmeans.h"
#include "silhouette.h"
#include <vector>
#include <unordered_set>
#include <random>
#include <limits>
#include <cstdint>

namespace ivf
{
  /*Product Quantizer*/
  class ProductQuantizer
  {
  public:
    ProductQuantizer(int M = 16, int nbits = 8, utils::DistanceFunc dist = utils::euclidean_distance);

    /*εκπαίδευση codebooks πάνω σε residuals (υποσύνολο για αποδοτικότητα)*/
    void train(const std::vector<std::vector<float>> &residuals);

    /*κωδικοποίηση ενός residual διανύσματος -> code (μήκους M, δείκτης κεντροειδούς ανά υποχώρο)*/
    std::vector<uint16_t> encode(const std::vector<float> &residual) const;

    /*κατασκευή LUT για το residual(αποστάσεις ανά υποχώρο προς τα codewords)*/
    std::vector<std::vector<float>> compute_LUT(const std::vector<float> &residual_q) const;

    /*απόσταση ADC χρησιμοποιώντας LUT και code*/
    static float adc_distance(const std::vector<std::vector<float>> &lut,
                              const std::vector<uint16_t> &code);

    /*μέθοδοι πρόσβασης*/
    int M() const { return M_; }
    int nbits() const { return nbits_; }
    int Ks() const { return Ks_; }
    const std::vector<int> &split_starts() const { return starts_; }
    const std::vector<int> &split_sizes() const { return sizes_; }

    /*(επανα)διαχωρισμός ενός D-διαστατικού χώρου σε M μπλοκ (σχεδόν ίσα)*/
    void set_dimension_splits(int D);

  private:
    int M_;
    int nbits_;
    int Ks_; /*2^nbits*/
    utils::DistanceFunc dist_func_;

    /*M sub-codebooks, το καθένα είναι Ks x d_m*/
    std::vector<std::vector<std::vector<float>>> codebooks_;
    /*subspace layout*/
    std::vector<int> starts_;
    std::vector<int> sizes_;
  };

  /*IVFPQ Index(coarse IVF + residual PQ ADC)*/
  class IVFPQ
  {
  public:
    IVFPQ(int kclusters, int nprobe, int M, int nbits,
          unsigned int seed = 1, int N = 1, float R = 2000.0f,
          utils::DistanceFunc dist_func = utils::euclidean_distance);

    void build_index(const std::vector<std::vector<float>> &data);

    /*αναζήτηση KNN χρησιμοποιώντας ADC*/
    std::vector<std::pair<int, float>> knn_query(const std::vector<float> &q, int N) const;

    /*αναζήτηση περιοχής (range) χρησιμοποιώντας ADC*/
    std::vector<int> range_search(const std::vector<float> &q, float R) const;

    /*Προαιρετικό: λήψη ταυτοτήτων υποψηφίων (από τις λίστες nprobe)*/
    std::vector<int> query_candidates(const std::vector<float> &q) const;

    void clear_index();

    /*μέθοδοι ανάκτησης (συμβατότητα με IVFFlat)*/
    int nlist() const { return kclusters_; }
    int nprobe() const { return nprobe_; }
    const std::vector<std::vector<float>> &centroids() const { return centroids_; }
    const std::vector<std::vector<float>> &data() const { return *data_ptr_; }
    utils::DistanceFunc distance_func() const { return dist_func_; }

  private:
    struct ListEntry
    {
      int id;
      std::vector<uint16_t> code; /*μήκος M*/
    };

    int kclusters_;
    int nprobe_;
    int M_;     /*PQ sub-vectors*/
    int nbits_; /*bits per sub-vector*/
    int Ks_;    /*2^nbits_*/
    unsigned int seed_;
    int N_;
    float R_;
    utils::DistanceFunc dist_func_;

    /*χονδρικός κβαντιστής (Coarse quantizer)*/
    std::vector<std::vector<float>> centroids_;    /*k x D*/
    std::vector<std::vector<ListEntry>> invlists_; /*k λίστες με κωδικοποιημένα residuals*/
    const std::vector<std::vector<float>> *data_ptr_ = nullptr;

    /*Residual Product Quantizer (κβαντιστής για residuals)*/
    ProductQuantizer pq_;

    /*helpers*/
    int nearest_centroid(const std::vector<float> &x) const;
    std::vector<int> top_nprobe_centroids(const std::vector<float> &q) const;
    static std::vector<float> residual_of(const std::vector<float> &x,
                                          const std::vector<float> &c);
  };

} /*namespace ivf*/

#endif /*IVFPQ_H*/