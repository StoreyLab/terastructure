#ifndef FAKEPHASE_HH
#define FAKEPHASE_HH

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

#define BATCH 1

class FakePhase;
class PhiCompute {
public:
  PhiCompute(const Env &env, gsl_rng **r, 
	     const uint32_t &iter,
	     uint32_t n, uint32_t k, uint32_t loc,
	     uint32_t t, const Matrix &Elogtheta, 
	     Matrix &Elogfdad, Matrix &Elogfmom,
	     const SNP &snp, 
	     const Matrix &eta, 
	     const FakePhase &pop)
    : _env(env), _r(r), _iter(iter),
      _n(n), _k(k), _loc(loc), _t(t),
      _Elogtheta(Elogtheta), 
      _Elogfdad(Elogfdad),
      _Elogfmom(Elogfmom),
      _v(_k,_t), _phidad(_n,_k), _phimom(_n,_k),
      _phinext(_k), 
      _snp(snp), 
      _lambda(_k,_t),
      _lambdaold(_k,_t), _Elogbeta(_k,_t), 
      _beta(_k), _eta(eta), _pop(pop)
  { }
  ~PhiCompute() { }

  int init_lambda();
  void reset(uint32_t loc);
  
  const Matrix& phimom() const { return _phimom; }
  const Matrix& phidad() const { return _phidad; }
  const Matrix& Elogbeta() const { return _Elogbeta;}
  const Matrix& lambda() const { return _lambda;}
  const Array& beta() const { return _beta;}

  uint32_t iter() const     { return _iter; }

  void update_phis_until_conv();
  void update_phimom(uint32_t n);
  void update_phidad(uint32_t n);
  void update_lambda();
  void compute_Elogfdad();
  void compute_Elogfmom();
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
  Matrix &_Elogfdad;
  Matrix &_Elogfmom;

  Matrix _v;

  Matrix _phidad;
  Matrix _phimom;
  Array _phinext;
  const SNP &_snp;
  
  Matrix _lambda;
  Matrix _lambdaold;
  Matrix _Elogbeta;
  Array _beta;
  
  const Matrix &_eta;
  const FakePhase &_pop;
};

class FakePhase {
public:
  FakePhase(Env &env, SNP &snp);
  ~FakePhase();

  void infer();
  bool kv_ok(uint32_t indiv, uint32_t loc) const;

#ifdef BATCH
  void batch_infer();
#else
  void batch_infer() { printf("error: batch inference disabled. Enable and recompile.\n"); }
#endif

  void load_model(string betafile = "", string thetafile = "");
  double snp_likelihood(uint32_t loc, uint32_t n);
  void snp_likelihood(uint32_t loc, uint32_t n, Array &p);
  

private:
  void init_heldout();
  void set_heldout_sample();
  void set_training_sample();

  void save_beta();
  void save_gamma();
  void save_model();

  double heldout_likelihood(bool first = false);
  double training_likelihood(bool first = false);
  double validation_likelihood();

  void init_gamma();
  uint32_t duration() const;
  
  double approx_log_likelihood();
  double logcoeff(yval_t x);
  
  double snp_likelihood(uint32_t loc, vector<uint32_t> &indiv, bool first = false);
  void estimate_pi(uint32_t p, Array &pi_p) const;
  void shuffle_nodes();

  void estimate_theta(uint32_t n, Array &theta) const;
  void estimate_all_theta();

  Env &_env;
  SNP &_snp;
  
  SNPMap _heldout_map;
  SNPMap _validation_map;
  SNPMap _training_map;

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
  vector<uint32_t> _training_loc;
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
  Array _nodec;
  Array _nodeupdatec;

  time_t _start_time;
  struct timeval _last_iter;
  FILE *_lf;

  Matrix _Elogtheta;
  Matrix _Elogfdad;
  Matrix _Elogfmom;
  Matrix _Etheta;
  
  FILE *_hf;
  FILE *_vf;
  FILE *_tf;
  FILE *_trf;
  FILE *_hef;
  FILE *_vef;
  FILE *_tef;

  PhiCompute _pcomp;
  uArray _shuffled_nodes;

  double _max_t, _max_h, _max_v, _prev_h, _prev_w, _prev_t;
  mutable uint32_t _nh, _nt;
  bool _training_done;
  uint32_t _sampled_loc;
  uint64_t _total_locations;

#ifdef BATCH
  Matrix _gphi;
#endif
};



inline void
PhiCompute::reset(uint32_t loc)
{
  _v.zero();
  _phidad.zero();
  _phimom.zero();
  _lambdaold.zero();
  _loc = loc;
  
  init_lambda();
}

inline int
PhiCompute::init_lambda()
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
  compute_Elogfmom();
  compute_Elogfdad();
  return 0;
}

inline void
PhiCompute::update_lambda()
{
  double **lambdad = _lambda.data();
  double **phimomd = _phimom.data();
  double **phidadd = _phidad.data();
  const yval_t ** const snpd = _snp.y().data();

  for (uint32_t k = 0; k < _k; ++k) {
    lambdad[k][0] = _env.eta0;
    lambdad[k][1] = _env.eta1;
    for (uint32_t i = 0; i < _n; ++i)  {
      if (!_pop.kv_ok(i, _loc))
	continue;
      lambdad[k][0] += phidadd[i][k] * _snp.dad(i,_loc) + \
	phimomd[i][k] * _snp.mom(i,_loc);
      lambdad[k][1] += phidadd[i][k] * (1 - _snp.dad(i,_loc)) +	\
	phimomd[i][k] * (1 - _snp.mom(i,_loc));
    }
  }
  PopLib::set_dir_exp(_lambda, _Elogbeta);
  compute_Elogfdad();
  compute_Elogfmom();
}

inline void
PhiCompute::update_phimom(uint32_t n)
{
  _phinext.zero();
  const double ** const elogthetad = _Elogtheta.const_data();
  const double ** const elogfd = _Elogfmom.const_data();
  for (uint32_t k = 0; k < _k; ++k)
    _phinext[k] = elogthetad[n][k] + elogfd[n][k];
  _phinext.lognormalize();
  _phimom.set_elements(n, _phinext);
}

inline void
PhiCompute::update_phidad(uint32_t n)
{
  _phinext.zero();
  const double ** const elogthetad = _Elogtheta.const_data();
  const double ** const elogfd = _Elogfdad.const_data();
  for (uint32_t k = 0; k < _k; ++k)
    _phinext[k] = elogthetad[n][k] + elogfd[n][k];
  _phinext.lognormalize();
  _phidad.set_elements(n, _phinext);
}

inline void
PhiCompute::compute_Elogfdad()
{
  _Elogfdad.zero();
  const double ** const elogbetad = _Elogbeta.const_data();
  double **elogfd = _Elogfdad.data();
  
  for (uint32_t n = 0; n < _n; ++n) {
    if (!_pop.kv_ok(n, _loc))
      continue;
    for (uint32_t k = 0; k < _k; ++k)
      for (uint32_t t = 0; t < _t; ++t) {
	double v = elogbetad[k][t] *					\
	  ((t == 0) ? _snp.dad(n,_loc) :  (1 - _snp.dad(n,_loc)));
	elogfd[n][k] += v;
      }
  }
}

inline void
PhiCompute::compute_Elogfmom()
{
  _Elogfmom.zero();
  const double ** const elogbetad = _Elogbeta.const_data();
  double **elogfd = _Elogfmom.data();
  
  for (uint32_t n = 0; n < _n; ++n) {
    if (!_pop.kv_ok(n, _loc))
      continue;
    for (uint32_t k = 0; k < _k; ++k)
      for (uint32_t t = 0; t < _t; ++t) {
	double v = elogbetad[k][t] *					\
	  ((t == 0) ? _snp.mom(n,_loc) :  (1 - _snp.mom(n,_loc)));
	elogfd[n][k] += v;
      }
  }
}

inline void
PhiCompute::update_phis_until_conv()
{
  double u = 1./_k;
  _phimom.set_elements(u);
  _phidad.set_elements(u);

  tst("location = %d", _loc);

  compute_Elogfdad();
  compute_Elogfmom();
  for (uint32_t i = 0; i < _env.online_iterations; ++i) {
    tst("iteration %d", i);
    for (uint32_t n = 0; n < _n; n++) {
      if (!_pop.kv_ok(i, _loc))
	continue;
      update_phimom(n);
      update_phidad(n);
    }
    tst("phis = %s\n", _phi.s().c_str());
    tst("before lambda = %s\n", _lambda.s().c_str());
    _lambdaold.copy_from(_lambda);
    update_lambda();
    tst("after lambda = %s\n", _lambda.s().c_str());
    tst("after Elogbeta = %s\n", _Elogbeta.s().c_str());
  
    //_v.zero();
    sub(_lambda, _lambdaold, _v);

    tst("v = %s", _v.s().c_str());
    
    if (_v.abs_mean() < _env.meanchangethresh)
      break;
  }
  estimate_beta();
}

inline void
PhiCompute::estimate_beta()
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
FakePhase::duration() const
{
  time_t t = time(0);
  return t - _start_time;
}

inline void
FakePhase::estimate_theta(uint32_t n, Array &theta) const
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
FakePhase::estimate_all_theta()
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
FakePhase::snp_likelihood(uint32_t loc, vector<uint32_t> &indivs, bool first)
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

#if LIKELIHOOD_ANALYSIS
  tst("loc = %d\n", loc);
  tst("beta = %s\n", beta.s().c_str());
  tst("snp = %s\n", a.s().c_str()); 
  FILE *f = fopen(Env::file_str("/likelihood-analysis.txt").c_str(), "a");
  tst("%d: indivs %ld\n", loc, indivs.size());
#endif

  double lsum = .0;
  for (uint32_t i = 0; i < indivs.size(); ++i)  {
    uint32_t n = indivs[i];
    assert (!_snp.is_missing(n, loc));

    yval_t ymom = _snp.mom(n, loc);
    yval_t ydad = _snp.dad(n, loc);

    double sumdad = .0;
    for (uint32_t k = 0; k < _k; ++k) {
      double m = pow(beta[k], ydad) * 
	pow(1 - beta[k], 1 - ydad) * thetad[n][k];
      sumdad += m;
    }
    double summom = .0;
    for (uint32_t k = 0; k < _k; ++k) {
      double m = pow(beta[k], ymom) * 
	pow(1 - beta[k], 1 - ymom) * thetad[n][k];
      summom += m;
    }

#if LIKELIHOOD_ANALYSIS
    fprintf(f, "%d\t%d\t%d\t%.3f\n", _iter, loc, i, sum);
#endif

    if (summom < 1e-30)
      summom = 1e-30;
    if (sumdad < 1e-30)
      sumdad = 1e-30;
    lsum += log(summom) + log(sumdad);
  }

#if LIKELIHOOD_ANALYSIS
  fclose(f);
#endif

  tst("logsum=%.5f\t%.5f\n", lsum / indivs.size(), exp(lsum / indivs.size()));
  return lsum;
}


inline void
FakePhase::snp_likelihood(uint32_t loc, uint32_t n, Array &p)
{
  assert (p.n() == 3);
  assert (loc < _l);
  assert (n < _n);

  const double ** const thetad = _Etheta.const_data();
  const double ** const betad = _beta.const_data();

  assert (!_snp.is_missing(n, loc));
  
  for (uint32_t x = 0; x < 3; ++x) {

    double sum = .0;
    for (uint32_t k = 0; k < _k; ++k) {
      double v = gsl_sf_fact(1) / 
	(gsl_sf_fact(x) * gsl_sf_fact(1 - x));
      //printf("v = %f\n", v);
      double m = v * pow(betad[loc][k], x) *			\
	pow(1 - betad[loc][k], 1 - x) * thetad[n][k];
      sum += m;
    }

    p[x] = sum;
  }
  return;
}

inline bool
FakePhase::kv_ok(uint32_t indiv, uint32_t loc) const
{
  assert (indiv < _n && loc < _l);

  KV kv(indiv, loc);
  
  const SNPMap::const_iterator u = _heldout_map.find(kv);
  if (u != _heldout_map.end())
    return false;

  const SNPMap::const_iterator w = _validation_map.find(kv);
  if (w != _validation_map.end())
    return false;

  if (_snp.is_missing(indiv, loc))
    return false;

  return true;
}

inline double
FakePhase::logcoeff(yval_t x) {
  uint32_t c = 1;
  return log(gsl_sf_fact(c)) - log(gsl_sf_fact(x) * gsl_sf_fact(c - x));
}


#endif

