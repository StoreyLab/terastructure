#ifndef POP_HH
#define POP_HH

#include <string>
#include <vector>
#include <queue>
#include <map>
#include <stdint.h>
#include "matrix.hh"
#include "env.hh"
#include "lib.hh"
#include <string.h>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sf_psi.h>
#include <gsl/gsl_sf.h>

#include <algorithm>

using namespace std;

class SNP {
public:
  SNP(Env &env);
  ~SNP() { }

  int read(string s);
  int read_bed(string s);
  int read_idfile(string s);
  const AdjMatrix &y() const { assert(_y); return *_y; }
  AdjMatrix &y() { assert(_y); return *_y; }
  //const map<KV, bool> &missing_snps() const { return _missing_snps; }
  bool is_missing(uint32_t indiv, uint32_t loc) const;
  string label(uint32_t id) const;

  uint32_t n() const;
  uint32_t l() const;
  yval_t dad(uint32_t i, uint32_t j) const;
  yval_t mom(uint32_t i, uint32_t j) const;

  double maf(uint32_t l) const { return _maf[l]; }

  uint32_t thrown() const { return _thrown; }

  double symmetrized_kl(Array &p1, Array &p2);
  double js_divergence(Array &p1, Array &p2);
  double kl(Array &p1, Array &p2);
  
private:
  Env &_env;
  AdjMatrix *_y;
  //map<KV, bool> _missing_snps;
  map<uint32_t, string> _labels;
  uint32_t _thrown;
  Array _maf;
  IDMap _loc_to_idx;
  gsl_rng *_r;


};

inline
SNP::SNP(Env &env):
  _env(env),
  _y(NULL),
  _thrown(0),
  _maf(_env.l),
  _r(NULL)
{
  gsl_rng_env_setup();
  const gsl_rng_type *T = gsl_rng_default;
  _r = gsl_rng_alloc(T);
  if (_env.seed) //is a local version of env in other code, ask prem
    gsl_rng_set(_r, _env.seed);
}

inline uint32_t
SNP::n() const
{
  assert(_y);
  return _y->m();
}

inline uint32_t
SNP::l() const
{
  assert(_y);
  return _y->n();
}

inline yval_t
SNP::mom(uint32_t a, uint32_t b) const
{
  assert(_y);
  assert (a < _y->m() && b < _y->n());
  const yval_t **yd = _y->const_data();
  //return (yd[a][b] == 2) ? 1 : 0;
  if (yd[a][b] == 2)
    return 1;
  else if (yd[a][b] == 1 && b % 2 == 0)
    return 1;
  return 0;
}

inline yval_t
SNP::dad(uint32_t a, uint32_t b) const
{
  assert(_y);
  assert (a < _y->m() && b < _y->n());
  const yval_t **yd = _y->const_data();
  //return (yd[a][b] == 1 || yd[a][b] == 2) ? 1 : 0;
  if (yd[a][b] == 2)
    return 1;
  else if (yd[a][b] == 1 && b % 2 == 1)
    return 1;
  return 0;
}

inline bool
SNP::is_missing(uint32_t indiv, uint32_t loc) const
{
  assert(_y);
  const yval_t **yd = _y->const_data();
  if (yd[indiv][loc] == 3)
    return true;
  return false;
}

inline string
SNP::label(uint32_t id) const
{
  map<uint32_t, string>::const_iterator i = _labels.find(id);
  if (i == _labels.end())
    return "";
  return i->second;
}

inline double
SNP::symmetrized_kl(Array &p1, Array &p2)
{
  return kl(p1, p2) + kl(p2, p1);
}

inline double
SNP::js_divergence(Array &p1, Array &p2)
{
  Array m(3);
  m.copy_from(p1);
  m.add_to(p2);
  m.scale(0.5);
  return 0.5 * kl(p1,m)  + 0.5 * kl(p2, m);
}

inline double
SNP::kl(Array &p1, Array &p2)
{
  assert (p1.n() == 3 && p2.n() == 3);
  double s = .0;
  for (uint32_t x = 0; x < 3; ++x) {
    if (p1[x] != .0 && p2[x] != .0)
      s += log(p1[x] / p2[x]) * p1[x];
    else
      _thrown++;
  }
  return s;
}



#endif
