#ifndef POPINF_HH
#define POPINF_HH

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

class PopInf;
class SNPCompute {
public:
  SNPCompute(const Env &env, gsl_rng **r, 
	     const uint32_t &iter,
	     uint32_t n, uint32_t k, uint32_t loc,
	     uint32_t t, const Matrix &Elogtheta, 
	     Matrix &Elogf,
	     const SNP &snp, 
	     const Matrix &eta, 
	     const PopInf &pop)
    : _env(env), _r(r), _iter(iter),
      _n(n), _k(k), _loc(loc), _t(t),
      _Elogtheta(Elogtheta), 
      _Elogf(Elogf),
      _v(_k,_t), _phi(_n,_k), _phinext(_k), 
      _snp(snp), 
      _lambda(_k,_t),
      _lambdaold(_k,_t), _Elogbeta(_k,_t), 
      _beta(_k), _eta(eta), _pop(pop)
  { }
  ~SNPCompute() { }

  int init_lambda();
  void reset(uint32_t loc);
  
  const Matrix& phi() const { return _phi; }
  const Matrix& Elogbeta() const { return _Elogbeta;}
  const Matrix& lambda() const { return _lambda;}
  const Array& beta() const { return _beta;}

  uint32_t iter() const     { return _iter; }

  void update_phis_until_conv();
  void update_phis(uint32_t n);
  void update_lambda();
  void compute_Elogf();
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
  Matrix &_Elogf;

  Matrix _v;
  
  Matrix _phi;
  Array _phinext;
  const SNP &_snp;
  
  Matrix _lambda;
  Matrix _lambdaold;
  Matrix _Elogbeta;
  Array _beta;
  
  const Matrix &_eta;
  const PopInf &_pop;
};

class PopInf {
public:
  PopInf(Env &env, SNP &snp);
  ~PopInf();

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
  Matrix _Elogf;
  Matrix _Etheta;
  
  FILE *_hf;
  FILE *_vf;
  FILE *_tf;
  FILE *_trf;
  FILE *_hef;
  FILE *_vef;
  FILE *_tef;

  SNPCompute _pcomp;
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
SNPCompute::reset(uint32_t loc)
{
  _v.zero();
  _phi.zero();
  _lambdaold.zero();
  _loc = loc;
  
  init_lambda();
}

inline int
SNPCompute::init_lambda()
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
  compute_Elogf();
  return 0;
}

inline void
SNPCompute::update_lambda()
{
  double **lambdad = _lambda.data();
  double **phid = _phi.data();
  const yval_t ** const snpd = _snp.y().data();

  for (uint32_t k = 0; k < _k; ++k) {
    lambdad[k][0] = _env.eta0;
    lambdad[k][1] = _env.eta1;
    for (uint32_t i = 0; i < _n; ++i)  {
      if (!_pop.kv_ok(i, _loc))
	continue;
      lambdad[k][0] += phid[i][k] * snpd[i][_loc];
      lambdad[k][1] += phid[i][k] * (2 - snpd[i][_loc]);
    }
  }
  //printf("env.eta = %f\n", _env.eta0);
  //printf("lambda = %s\n", _lambda.s().c_str());
  //printf("phi = %s\n", _phi.s().c_str());
  //fflush(stdout);
  PopLib::set_dir_exp(_lambda, _Elogbeta);
  compute_Elogf();
}

inline void
SNPCompute::update_phis(uint32_t n)
{
  _phinext.zero();
  const double ** const elogthetad = _Elogtheta.const_data();
  const double ** const elogfd = _Elogf.const_data();
  for (uint32_t k = 0; k < _k; ++k)
    _phinext[k] = elogthetad[n][k] + elogfd[n][k];
  _phinext.lognormalize();
  _phi.set_elements(n, _phinext);
}

inline void
SNPCompute::compute_Elogf()
{
  _Elogf.zero();
  const double ** const elogbetad = _Elogbeta.const_data();
  double **elogfd = _Elogf.data();
  const yval_t ** const snpd = _snp.y().const_data();

  for (uint32_t n = 0; n < _n; ++n) {
    if (!_pop.kv_ok(n, _loc))
      continue;
    //printf("(%d:%d) -> %d\n", n, _loc, snpd[n][_loc]);
    for (uint32_t k = 0; k < _k; ++k)
      for (uint32_t t = 0; t < _t; ++t) {
	double v = elogbetad[k][t] * ((t == 0) ? (_env.eta0 + snpd[n][_loc] - 1) :
				      (_env.eta1 + 2 - snpd[n][_loc] - 1));
	elogfd[n][k] += v;
      }
  }
}

inline void
SNPCompute::update_phis_until_conv()
{
  double u = 1./_k;
  _phi.set_elements(u);

  tst("location = %d", _loc);

  compute_Elogf();
  for (uint32_t i = 0; i < _env.online_iterations; ++i) {
    tst("iteration %d", i);
    for (uint32_t n = 0; n < _n; n++) {
      if (!_pop.kv_ok(i, _loc))
	continue;
      update_phis(n);
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
SNPCompute::estimate_beta()
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
PopInf::duration() const
{
  time_t t = time(0);
  return t - _start_time;
}

inline void
PopInf::estimate_theta(uint32_t n, Array &theta) const
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
PopInf::estimate_all_theta()
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
PopInf::snp_likelihood(uint32_t loc, vector<uint32_t> &indivs, bool first)
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
    
    //Array b(_k);
    //_Etheta.slice(0, i, b);
    //printf("theta (%d) = %s\n", i, b.s().c_str());
    double sum = .0;
    for (uint32_t k = 0; k < _k; ++k) {
      double v = gsl_sf_fact(2) / 
	(gsl_sf_fact(snpd[n][loc]) * gsl_sf_fact(2 - snpd[n][loc]));
      //printf("v = %f\n", v);
      double m = v * pow(beta[k], snpd[n][loc]) *			\
	pow(1 - beta[k], 2 - snpd[n][loc]) * thetad[n][k];
      sum += m;
    }

#if LIKELIHOOD_ANALYSIS
    fprintf(f, "%d\t%d\t%d\t%.3f\n", _iter, loc, i, sum);
#endif

    if (sum < 1e-30)
      sum = 1e-30;
    lsum += log(sum);
  }

#if LIKELIHOOD_ANALYSIS
  fclose(f);
#endif

  tst("logsum=%.5f\t%.5f\n", lsum / indivs.size(), exp(lsum / indivs.size()));
  return lsum;
}


inline void
PopInf::snp_likelihood(uint32_t loc, uint32_t n, Array &p)
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
      double v = gsl_sf_fact(2) / 
	(gsl_sf_fact(x) * gsl_sf_fact(2 - x));
      //printf("v = %f\n", v);
      double m = v * pow(betad[loc][k], x) *			\
	pow(1 - betad[loc][k], 2 - x) * thetad[n][k];
      sum += m;
    }

    p[x] = sum;
  }
  return;
}

inline bool
PopInf::kv_ok(uint32_t indiv, uint32_t loc) const
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
PopInf::logcoeff(yval_t x) {
  uint32_t c = 2;
  return log(gsl_sf_fact(c)) - log(gsl_sf_fact(x) * gsl_sf_fact(c - x));
}


#endif

