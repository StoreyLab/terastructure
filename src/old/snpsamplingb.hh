#ifndef SNPSAMPLINGB_HH
#define SNPSAMPLINGB_HH

#include <list>
#include <utility>
#include <unistd.h>
#include <stdio.h>
#include <stdint.h>

#include "env.hh"
#include "matrix.hh"
#include "lib.hh"
#include "snp.hh"

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sf_psi.h>
#include <gsl/gsl_sf.h>

class SNPSamplingB;
class SNPSamplingBPhi {
public:
  SNPSamplingBPhi(const Env &env, gsl_rng **r, 
		  const uint32_t &iter,
		  uint32_t n, uint32_t k, 
		  uint32_t loc, uint32_t t, 
		  const SNP &snp, 
		  const SNPSamplingB &pop)
    : _env(env), _r(r), _iter(iter),
      _n(n), _k(k), _loc(loc), _t(t),
      _phidad(_n,_k), _phimom(_n,_k),
      _phinext(_k), 
      _snp(snp), 
      _pop(pop)
  { }
  ~SNPSamplingBPhi() { }

  void reset(uint32_t loc);  
  const Matrix& phimom()   const { return _phimom; }
  const Matrix& phidad()   const { return _phidad; }
  uint32_t iter()          const { return _iter; }

  void update_phis();
  void update_phis_all();
  void update_phimom(uint32_t n);
  void update_phidad(uint32_t n);
  void save_beta();

private:
  const Env &_env;
  gsl_rng **_r;
  const uint32_t &_iter;
  
  uint32_t _n;
  uint32_t _k;
  uint32_t _loc;
  uint32_t _t;

  Matrix _phidad;
  Matrix _phimom;
  Array _phinext;
  const SNP &_snp;
  const SNPSamplingB &_pop;
};

class SNPSamplingB {
public:
  SNPSamplingB(Env &env, SNP &snp);
  ~SNPSamplingB();

  void infer();
  bool kv_ok(uint32_t indiv, uint32_t loc) const;
  void load_model(string betafile = "", string thetafile = "");
  void snp_likelihood(uint32_t loc, uint32_t n, Array &p);

  const uArray& shuffled_nodes() const { return _shuffled_nodes; }

  const Matrix &Elogtheta() const   { return _Elogtheta; }
  const D3 &Elogbeta() const        { return _Elogbeta;  }
  const vector<uint32_t> &indivs() const { return _indivs;   }

private:
  void init_heldout_sets();
  void set_test_sample();
  void set_validation_sample();
  void set_validation_sample2();
  void estimate_beta(uint32_t loc);
  double logl();

  void save_beta();
  void save_gamma();
  void save_model();

  double compute_likelihood(bool first, bool validation);

  void init_gamma();
  void init_lambda();

  void update_gamma();
  void update_lambda();

  void get_subsample();
  uint32_t duration() const;

  void estimate_theta();
  void estimate_beta();
  
  double approx_log_likelihood();
  double logcoeff(yval_t x);
  
  double snp_likelihood(uint32_t loc, vector<uint32_t> &indiv, bool first = false);
  void estimate_pi(uint32_t p, Array &pi_p) const;
  void shuffle_nodes();

  void estimate_theta(uint32_t n, Array &theta) const;
  void estimate_all_theta();
  string add_iter_suffix(const char *c);
  void update_rho_indiv(uint32_t n);
  void update_rho_loc(uint32_t l);

  Env &_env;
  SNP &_snp;
  
  SNPMap _test_map;
  SNPMap _validation_map;

  uint64_t _n;
  uint32_t _k;
  uint64_t _l;
  uint32_t _t;
  uint32_t _iter;
  Array _alpha;
  uint32_t _loc;

  Matrix _eta;

  vector<uint32_t> _heldout_loc;
  vector<uint32_t> _validation_loc;
  gsl_rng *_r;

  Matrix _gamma;
  D3 _lambda;
  Matrix _lambdat;

  Matrix *_gamma_ag;
  D3 *_lambda_ag;

  double _tau0;
  double _kappa;
  double _nodetau0;
  double _nodekappa;

  Array _rho_indiv;
  Array _rho_loc;
  uArray _c_indiv;
  uArray _c_loc;
  
  double _rhot;
  double _noderhot;
  uint32_t _nodec;
  Array _nodeupdatec;

  time_t _start_time;
  struct timeval _last_iter;
  FILE *_lf;

  Matrix _Elogtheta;
  D3 _Elogbeta;
  Matrix _Etheta;
  Matrix _Ebeta;
  
  FILE *_vf;
  FILE *_tf;
  FILE *_trf;
  FILE *_hef;
  FILE *_vef;
  FILE *_tef;

  SNPSamplingBPhi _pcomp;
  uArray _shuffled_nodes;
  vector<uint32_t> _indivs;

  double _max_t, _max_h, _max_v, _prev_h, _prev_w, _prev_t;
  mutable uint32_t _nh, _nt;
  uint32_t _sampled_loc;
  uint64_t _total_locations;
};

inline void
SNPSamplingBPhi::reset(uint32_t loc)
{
  _phidad.zero();
  _phimom.zero();
  _loc = loc;
}

inline void
SNPSamplingBPhi::update_phimom(uint32_t n)
{
  //_phinext.zero();
  const double ** const elogthetad = _pop.Elogtheta().const_data();
  const double *** const elogbetad = _pop.Elogbeta().const_data(); 
  for (uint32_t k = 0; k < _k; ++k)
    _phinext[k] = elogthetad[n][k] + elogbetad[_loc][k][0];
  _phinext.lognormalize();
  _phimom.set_elements(n, _phinext);
}

inline void
SNPSamplingBPhi::update_phidad(uint32_t n)
{
  //_phinext.zero();
  const double ** const elogthetad = _pop.Elogtheta().const_data();
  const double *** const elogbetad = _pop.Elogbeta().const_data();
  for (uint32_t k = 0; k < _k; ++k)
    _phinext[k] = elogthetad[n][k] + elogbetad[_loc][k][1];
  _phinext.lognormalize();
  _phidad.set_elements(n, _phinext);
}

inline void
SNPSamplingBPhi::update_phis()
{
  double u = 1./_k;
  for (uint32_t i = 0; i < _pop.indivs().size(); ++i) {
    uint32_t n = _pop.indivs()[i];
    _phimom.set_elements(n, u);
    _phidad.set_elements(n, u);
    update_phimom(n);
    update_phidad(n);
  }
  debug("iter = %d, phimom = %s", _iter, _phimom.s().c_str());
  debug("iter = %d, phidad = %s", _iter, _phidad.s().c_str());
}

inline void
SNPSamplingBPhi::update_phis_all()
{
  double u = 1./_k;
  _phimom.set_elements(u);
  _phidad.set_elements(u);

  for (uint32_t i = 0; i < _n; ++i) {
    update_phimom(i);
    update_phidad(i);
  }
}

inline uint32_t
SNPSamplingB::duration() const
{
  time_t t = time(0);
  return t - _start_time;
}

inline void
SNPSamplingB::estimate_beta(uint32_t loc)
{
  // lambda for location _loc
  const double ***const ld = _lambda.const_data();
  double **betad = _Ebeta.data();
  for (uint32_t k = 0; k < _k; ++k) {
    double s = .0;
    for (uint32_t t = 0; t < _t; ++t)
      s += ld[loc][k][t];
    betad[loc][k] = ld[loc][k][0] / s;
  }
}

inline double
SNPSamplingB::snp_likelihood(uint32_t loc, vector<uint32_t> &indivs, bool first)
{
  const double ** const thetad = _Etheta.const_data();
  const double ** const betad = _Ebeta.const_data();
  const yval_t ** const snpd = _snp.y().const_data();

  D1Array<yval_t> a(_n);
  _snp.y().slice(1, loc, a);

  if (first)
    estimate_beta(loc);

  double lsum = .0;
  for (uint32_t i = 0; i < indivs.size(); ++i)  {
    uint32_t n = indivs[i];
    assert (!_snp.is_missing(n, loc));

    double sum = .0;

    yval_t x = snpd[n][loc];
    double q = .0;
    double v = gsl_sf_fact(2) / 
      (gsl_sf_fact(x) * gsl_sf_fact(2 - x));
    
    for (uint32_t k = 0; k < _k; ++k)
      q += betad[loc][k] * thetad[n][k];
    
    sum = v * pow(q, x) *  pow(1 - q, 2 - x);
    if (sum < 1e-30)
      sum = 1e-30;
    lsum += log(sum);
  }
  tst("logsum=%.5f\t%.5f\n", lsum / indivs.size(), exp(lsum / indivs.size()));
  return lsum;
}


inline void
SNPSamplingB::snp_likelihood(uint32_t loc, uint32_t n, Array &p)
{
  assert (p.n() == 3);
  assert (loc < _l);
  assert (n < _n);

  const double ** const thetad = _Etheta.const_data();
  const double ** const betad = _Ebeta.const_data();

  assert (!_snp.is_missing(n, loc));
  
  for (uint32_t x = 0; x < 3; ++x) {
    double q = .0;
    double v = gsl_sf_fact(2) / 
      (gsl_sf_fact(x) * gsl_sf_fact(2 - x));
    
    for (uint32_t k = 0; k < _k; ++k)
      q += betad[loc][k] * thetad[n][k];
    
    double m = v * pow(q, x) *  pow(1 - q, 2 - x);
    p[x] = m;
  }
  return;
}

inline bool
SNPSamplingB::kv_ok(uint32_t indiv, uint32_t loc) const
{
  assert (indiv < _n && loc < _l);

  KV kv(indiv, loc);
  
  const SNPMap::const_iterator u = _test_map.find(kv);
  if (u != _test_map.end())
    return false;

  const SNPMap::const_iterator w = _validation_map.find(kv);
  if (w != _validation_map.end())
    return false;

  if (_snp.is_missing(indiv, loc))
    return false;

  return true;
}

inline double
SNPSamplingB::logcoeff(yval_t x) {
  uint32_t c = 2;
  return log(gsl_sf_fact(c)) - log(gsl_sf_fact(x) * gsl_sf_fact(c - x));
}


#endif

