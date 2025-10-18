#include "sift.h"
#include <fstream>
#include <iostream>
#include <cstdint>
#include <stdexcept>
#include <limits>

/* ΒΟΗΘΗΤΙΚΑ (commit 5): endianness & byte-swaps
   - is_little_endian(): true αν η μηχανή είναι Little-Endian.
   - bswap32(): αντιστροφή byte-order για 32-bit τιμές.
*/
static bool is_little_endian()
{
  const uint16_t x = 1;
  return *reinterpret_cast<const uint8_t *>(&x) == 1;
}

static inline uint32_t bswap32(uint32_t v)
{
  return ((v & 0x000000FFu) << 24) |
         ((v & 0x0000FF00u) << 8) |
         ((v & 0x00FF0000u) >> 8) |
         ((v & 0xFF000000u) >> 24);
}

namespace sift
{
  /* φόρτωση αρχείου .dat/.fvecs (LE).
     Διαβάζουμε επαναλαμβανόμενες εγγραφές:
       [int32 dim][dim * float32]
     Edge cases:
     - Αν το αρχείο δεν ανοίγει -> exception.
     - Αν διαβάσουμε dim != expected_dim -> exception (κατεστραμμένα/λάθος δεδομένα).
     - Αν κοπεί η ανάγνωση στη μέση των float (truncated) -> exception.
     - Αν δεν διαβαστεί κανένα διάνυσμα -> warning στο stderr.
     - Big-endian: γίνεται αντιστροφή byte-order για dim και για κάθε float.
     - max_vectors > 0: ανάγνωση μέχρι αυτό το όριο.
     Σημείωση: Υποστηρίζονται αρχεία με κατάληξη .dat και .fvecs (ίδια διάταξη).
  */
  std::vector<std::vector<float>> load_sift_dat(const std::string &filename,
                                                int max_vectors,
                                                int expected_dim)
  {
    std::ifstream f(filename, std::ios::binary);
    if (!f)
      throw std::runtime_error("Cannot open SIFT file: " + filename);

    std::vector<std::vector<float>> data;
    data.reserve(10000); /*συντηρητική αρχικοποίηση, θα μεγαλώσει δυναμικά*/

    const bool sys_le = is_little_endian();

    /*Διαβάζουμε συνεχόμενες εγγραφές μέχρι EOF.
       Κάθε εγγραφή = [int32 dim][dim * float32].*/
    while (true)
    {
      /*(1) διάσταση (int32 Little-Endian)*/
      int32_t dim_raw = 0;
      f.read(reinterpret_cast<char *>(&dim_raw), 4);

      if (!f)
      {
        /*EOF πριν από νέα έγκυρη εγγραφή -> ολοκλήρωση χωρίς σφάλμα*/
        break;
      }

      if (!sys_le)
      {
        /*Big-endian μηχανή: μετατροπή του dim από LE */
        uint32_t u = bswap32(static_cast<uint32_t>(dim_raw));
        dim_raw = static_cast<int32_t>(u);
      }

      if (dim_raw != expected_dim)
      {
        throw std::runtime_error(
            "SIFT record with unexpected dimension: " +
            std::to_string(dim_raw) + " (expected " +
            std::to_string(expected_dim) + ")");
      }

      /*(2) διαβάζουμε dim floats (Little-Endian)*/
      std::vector<float> v(expected_dim);
      f.read(reinterpret_cast<char *>(v.data()),
             sizeof(float) * static_cast<size_t>(expected_dim));

      if (!f)
        throw std::runtime_error(
            "Truncated SIFT vector payload in file: " + filename);

      if (!sys_le)
      {
        /*Big-endian: αντιστροφή bytes στα float*/
        for (int i = 0; i < expected_dim; ++i)
        {
          uint32_t *pi = reinterpret_cast<uint32_t *>(&v[i]);
          *pi = bswap32(*pi);
        }
      }

      data.emplace_back(std::move(v));

      if (max_vectors > 0 &&
          static_cast<int>(data.size()) >= max_vectors)
      {
        break;
      }
    }

    if (data.empty())
      std::cerr << "Warning: no vectors read from " << filename << "\n";

    return data;
  }

} /*namespace sift*/
