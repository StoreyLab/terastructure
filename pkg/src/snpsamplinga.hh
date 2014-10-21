#ifndef SNPSAMPLINGA_HH
#define SNPSAMPLINGA_HH

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

class SNPSamplingA;
class SNPSamplingAPhi {
public:
  SNPSamplingAPhi(const Env &env, gsl_rng **r, 
	     const uint32_t &iter,
	     uint32_t n, uint32_t k, uint32_t loc,
	     uint32_t t, const Matrix &Elogtheta, 
	     const SNP &snp, 
	     const Matrix &eta, 
	     const SNPSamplingA &pop)
    : _env(env), _r(r), _iter(iter),
      _n(n), _k(k), _loc(loc), _t(t),
      _Elogtheta(Elogtheta), 
      _v(_k,_t), _phidad(_n,_k), _phimom(_n,_k),
      _phinext(_k), 
      _snp(snp), 
      _lambda(_k,_t),
      _lambdaold(_k,_t), _Elogbeta(_k,_t), 
      _beta(_k), _eta(eta), _pop(pop)
  { }
  ~SNPSamplingAPhi() { }

  int init_lambda();
  void reset(uint32_t loc);
  
  const Matrix& phimom() const { return _phimom; }
  const Matrix& phidad() const { return _phidad; }
  const Matrix& Elogbeta() const { return _Elogbeta;}
  const Matrix& lambda() const { return _lambda;}
  const Array& beta() const { return _beta;}
  const vector<uint32_t> &indiv() const { return _indivs; }

  uint32_t iter() const     { return _iter; }

  void update_phis_until_conv();
  void update_phimom(uint32_t n);
  void update_phidad(uint32_t n);
  void update_lambda();
  double estimate_mean_rate(uint32_t k) const;
  void estimate_beta();

private:
  const Env &_env;
  gsl_rng **_r;
  const uint32_t &_iter;
  
  uint32_t _n;
  uint32_t _k;
  uint32_t _loc;
  uint32_t _t;
 
  const Matrix &_Elogtheta;
  Matrix _v;

  Matrix _phidad;
  Matrix _phimom;
  Array _phinext;
  const SNP &_snp;
  
  Matrix _lambda;
  Matrix _lambdaold;
  Matrix _Elogbeta;
  Array _beta;
  vector<uint32_t> _indivs;
  
  const Matrix &_eta;
  const SNPSamplingA &_pop;
};

class SNPSamplingA {
public:
  SNPSamplingA(Env &env, SNP &snp);
  ~SNPSamplingA();

  void infer();
  bool kv_ok(uint32_t indiv, uint32_t loc) const;
  void load_model(string betafile = "", string thetafile = "");
  void snp_likelihood(uint32_t loc, uint32_t n, Array &p);

  const uArray& shuffled_nodes() const { return _shuffled_nodes; }

private:
  void init_heldout_sets();
  void set_test_sample();
  void set_validation_sample();

  void save_beta();
  void save_gamma();
  void save_model();

  double compute_likelihood(bool first, bool validation);

  void init_gamma();
  uint32_t duration() const;
  void estimate_theta(const vector<uint32_t> &indivs);
  
  double approx_log_likelihood();
  double logcoeff(yval_t x);
  
  double snp_likelihood(uint32_t loc, vector<uint32_t> &indiv, bool first = false);
  void estimate_pi(uint32_t p, Array &pi_p) const;
  void shuffle_nodes();

  void estimate_theta(uint32_t n, Array &theta) const;
  void estimate_all_theta();
  string add_iter_suffix(const char *c);

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

  yval_t _family;
  Matrix _eta;

  vector<uint32_t> _heldout_loc;
  vector<uint32_t> _validation_loc;
  gsl_rng *_r;

  Matrix _gamma;
  Matrix _theta;
  Matrix _beta;

  double _tau0;
  double _kappa;
  double _nodetau0;
  double _nodekappa;
  
  double _rhot;
  Array _noderhot;
  uArray _nodec;
  Array _nodeupdatec;

  time_t _start_time;
  struct timeval _last_iter;
  FILE *_lf;

  Matrix _Elogtheta;
  Matrix _Etheta;
  
  FILE *_vf;
  FILE *_tf;
  FILE *_trf;
  FILE *_hef;
  FILE *_vef;
  FILE *_tef;

  SNPSamplingAPhi _pcomp;
  uArray _shuffled_nodes;

  double _max_t, _max_h, _max_v, _prev_h, _prev_w, _prev_t;
  mutable uint32_t _nh, _nt;
  uint32_t _sampled_loc;
  uint64_t _total_locations;
};



inline void
SNPSamplingAPhi::reset(uint32_t loc)
{
  _v.zero();
  _phidad.zero();
  _phimom.zero();
  _lambdaold.zero();
  _loc = loc;
  _indivs.clear();
  
  init_lambda();
}

inline int
SNPSamplingAPhi::init_lambda()
{
  if (_lambda.copy_from(_eta) < 0) {
    lerr("init lambda failed");
    return -1;
  }
  double **ld = _lambda.data();
  for (uint32_t k = 0; k < _k; ++k)
    for (uint32_t t = 0; t < _t; ++t) {
      double v = (_k <= 100) ? 1.0 : (double)100.0 / _k;
      ld[k][t] += gsl_ran_gamma(*_r, 100 * v, 0.01);
    }
  PopLib::set_dir_exp(_lambda, _Elogbeta);
  return 0;
}

inline void
SNPSamplingAPhi::update_lambda()
{
  double **lambdad = _lambda.data();
  double **phimomd = _phimom.data();
  double **phidadd = _phidad.data();
  const yval_t ** const snpd = _snp.y().data();

  for (uint32_t k = 0; k < _k; ++k) {
    lambdad[k][0] = .0;
    lambdad[k][1] = .0;
    for (uint32_t i = 0; i < _indivs.size(); ++i)  {
      uint32_t n = _indivs[i];
      if (!_pop.kv_ok(n, _loc))
	continue;
      lambdad[k][0] += phimomd[n][k] * snpd[n][_loc];
      lambdad[k][1] += phidadd[n][k] * (2 - snpd[n][_loc]);
    }
  }
  for (uint32_t k = 0; k < _k; ++k) {
    lambdad[k][0] = _env.eta0 + ((double)_n / _indivs.size()) * lambdad[k][0];
    lambdad[k][1] = _env.eta1 + ((double)_n / _indivs.size()) * lambdad[k][1];
  }
  PopLib::set_dir_exp(_lambda, _Elogbeta);
}

inline void
SNPSamplingAPhi::update_phimom(uint32_t n)
{
  //_phinext.zero();
  const double ** const elogthetad = _Elogtheta.const_data();
  const double ** const elogbetad = _Elogbeta.const_data();
  for (uint32_t k = 0; k < _k; ++k)
    _phinext[k] = elogthetad[n][k] + elogbetad[k][0];
  _phinext.lognormalize();
  _phimom.set_elements(n, _phinext);
}

inline void
SNPSamplingAPhi::update_phidad(uint32_t n)
{
  //_phinext.zero();
  const double ** const elogthetad = _Elogtheta.const_data();
  const double ** const elogbetad = _Elogbeta.const_data();
  for (uint32_t k = 0; k < _k; ++k)
    _phinext[k] = elogthetad[n][k] + elogbetad[k][1];
  _phinext.lognormalize();
  _phidad.set_elements(n, _phinext);
}

inline void
SNPSamplingAPhi::update_phis_until_conv()
{
  double u = 1./_k;
  _phimom.set_elements(u);
  _phidad.set_elements(u);

  const uArray &shuffled_nodes = _pop.shuffled_nodes();

  // get subsample of individuals
  double v = (double)(gsl_rng_uniform_int(*_r, _n)) / _env.indiv_sample_size;
  uint32_t q = ((int)v) * _env.indiv_sample_size;
  _indivs.clear();
  while (_indivs.size() < _env.indiv_sample_size) {
    uint32_t n = shuffled_nodes[q];
    if (!_pop.kv_ok(n, _loc)) {
      q = (q + 1) % _n;
      continue;
    }
    _indivs.push_back(n);
    q = (q + 1) % _n;
    continue;
  }

  for (uint32_t i = 0; i < _env.online_iterations; ++i) {
    for (uint32_t m = 0; m < _indivs.size(); ++m) {
      uint32_t n = _indivs[m];
      update_phimom(n);
      update_phidad(n);
    }
    _lambdaold.copy_from(_lambda);
    update_lambda();
  
    //_v.zero();
    sub(_lambda, _lambdaold, _v);
    tst("v = %s", _v.s().c_str());
    
    if (_v.abs_mean() < _env.meanchangethresh)
      break;
  }
  estimate_beta();
}

inline void
SNPSamplingAPhi::estimate_beta()
{
  // lambda for location _loc
  const double ** const ld = _lambda.const_data();
  double *betad = _beta.data();
  for (uint32_t k = 0; k < _k; ++k) {
    double s = .0;
    for (uint32_t t = 0; t < _t; ++t)
      s += ld[k][t];
    betad[k] = ld[k][0] / s;
  }
}

inline uint32_t
SNPSamplingA::duration() const
{
  time_t t = time(0);
  return t - _start_time;
}

inline void
SNPSamplingA::estimate_theta(uint32_t n, Array &theta) const
{
  const double ** const gd = _gamma.data();
  double s = .0;
  for (uint32_t k = 0; k < _k; ++k)
    s += gd[n][k];
  assert(s);
  for (uint32_t k = 0; k < _k; ++k)
    theta[k] = gd[n][k] / s;
}

inline void
SNPSamplingA::estimate_all_theta()
{
  const double ** const gd = _gamma.const_data();
  double **theta = _Etheta.data();
  for (uint32_t n = 0; n < _n; ++n) {
    double s = .0;
    for (uint32_t k = 0; k < _k; ++k)
      s += gd[n][k];
    assert(s);
    for (uint32_t k = 0; k < _k; ++k)
      theta[n][k] = gd[n][k] / s;
  }
  PopLib::set_dir_exp(_gamma, _Elogtheta);
}

inline double
SNPSamplingA::snp_likelihood(uint32_t loc, vector<uint32_t> &indivs, bool first)
{
  const double ** const thetad = _Etheta.const_data();
  const yval_t ** const snpd = _snp.y().const_data();

  if (first) {
    _pcomp.reset(loc);
    _pcomp.estimate_beta();
  } else {
    _pcomp.reset(loc);
    _pcomp.update_phis_until_conv();
  }
  const Array &beta = _pcomp.beta();

  D1Array<yval_t> a(_n);
  _snp.y().slice(1, loc, a);

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
      q += beta[k] * thetad[n][k];
    
    sum = v * pow(q, x) *  pow(1 - q, 2 - x);
    if (sum < 1e-30)
      sum = 1e-30;
    lsum += log(sum);
  }



  tst("logsum=%.5f\t%.5f\n", lsum / indivs.size(), exp(lsum / indivs.size()));
  return lsum;
}


inline void
SNPSamplingA::snp_likelihood(uint32_t loc, uint32_t n, Array &p)
{
  assert (p.n() == 3);
  assert (loc < _l);
  assert (n < _n);

  const double ** const thetad = _Etheta.const_data();
  const double ** const betad = _beta.const_data();

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
SNPSamplingA::kv_ok(uint32_t indiv, uint32_t loc) const
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
SNPSamplingA::logcoeff(yval_t x) {
  uint32_t c = 2;
  return log(gsl_sf_fact(c)) - log(gsl_sf_fact(x) * gsl_sf_fact(c - x));
}


#endif

