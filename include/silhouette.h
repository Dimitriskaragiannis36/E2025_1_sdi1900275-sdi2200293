#ifndef SILHOUETTE_H
#define SILHOUETTE_H

#include "helper.h"
#include <vector>
#include <cstddef>

namespace clustering {

class Silhouette {
public:
    //κονστρακτορας που δέχεται τη συνάρτηση απόστασης (όπως και το KMeans)
    explicit Silhouette(utils::DistanceFunc dist_func = utils::euclidean_distance);

    //υπολογίζει το συνολικό μέσο silhouette score
    float compute(
        const std::vector<std::vector<float>>& data,
        const std::vector<int>& labels,
        int k
    ) const;

    //υπολογίζει και επιστρέφει τα επιμέρους s(i)
    std::vector<float> compute_per_point(
        const std::vector<std::vector<float>>& data,
        const std::vector<int>& labels,
        int k
    ) const;

private:
    utils::DistanceFunc dist_func_;
};

} //namespace clustering

#endif //SILHOUETTE_H
