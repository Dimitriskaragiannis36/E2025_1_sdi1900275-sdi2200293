#ifndef SIFT_H
#define SIFT_H

#include <vector>
#include <string>

/*φόρτωση διανυσμάτων SIFT από δυαδικό αρχείο Little-Endian (.dat ή .fvecs).
   Μορφή εγγραφής (επαναλαμβανόμενη ανά διάνυσμα, τύπου fvecs):
     - 32-bit signed int: dimension (π.χ. 128)
     - 128 * 32-bit float: οι συντεταγμένες του διανύσματος
   Επιστρέφει: vector<vector<float>> (float32).
*/
namespace sift
{
  std::vector<std::vector<float>> load_sift_dat(const std::string &filename,
                                                int max_vectors = -1,
                                                int expected_dim = 128);
}

#endif /* SIFT_H */
