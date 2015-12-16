#ifndef SNPSAMPLINGG_HH
#define SNPSAMPLINGG_HH

#include <list>
#include <utility>
#include <unistd.h>
#include <stdio.h>
#include <stdint.h>

#include "env.hh"
#include "matrix.hh"
#include "lib.hh"
#include "snp.hh"
#include "thread.hh"
#include "tsqueue.hh"

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sf_psi.h>
#include <gsl/gsl_sf.h>

typedef vector<uint32_t> IndivsList;
typedef std::map<uint32_t, IndivsList *> ChunkMap;
class SNPSamplingG;
class PhiRunnerG : public Thread {
public:
  PhiRunnerG(const Env &env, gsl_rng **r, 
	     const uint32_t &iter,
	     const uint32_t &x,
	     uint32_t n, uint32_t k, 
	     uint32_t loc, uint32_t t, 
	     const SNP &snp, 
	     SNPSamplingG &pop,
	     TSQueue<IndivList> &out_q,
	     TSQueue<pthread_t> &in_q,
	     CondMutex &cm)
    : _env(env), _r(r), _iter(iter), _x(x),
      _prev_iter(0),
      _prev_x(0), 
      _prev_hol_mode(false),
      _n(n), _k(k), _loc(loc), _t(t),
      _phidad(_n,_k), _phimom(_n,_k),
      _phinext(_k), _lambdat(_k,_t),
      _snp(snp), 
      _pop(pop),
      _out_q(out_q),
      _in_q(in_q),
      _cm(cm),
      _oldilist(NULL),
      _idptr(NULL)
  { }
  ~PhiRunnerG() { if (_idptr) { delete _idptr; } } 

  int do_work();
  int process(const IndivsList &v);
  int init_process(const IndivsList &v);
  void reset(uint32_t loc);  
  const Matrix& phimom()   const   { return _phimom; }
  const Matrix& phidad()   const   { return _phidad; }
  const Matrix& lambdat()  const   { return _lambdat; }
  uint32_t iter()          const   { return _iter; }

  void update_phis_all();
  void update_phimom(uint32_t n);
  void update_phidad(uint32_t n);

  void update_gamma(const IndivsList &i);
  void update_lambda_t(const IndivsList &i);
  void estimate_theta(const IndivsList &i);
  void update_gamma();
  void estimate_theta();

private:
  const Env &_env;
  gsl_rng **_r;
  const uint32_t &_iter;
  const uint32_t &_x;
  uint32_t _prev_iter;
  uint32_t _prev_x;
  bool _prev_hol_mode;

  uint32_t _n;
  uint32_t _k;
  uint32_t _loc;
  uint32_t _t;

  Matrix _phidad;
  Matrix _phimom;
  Array _phinext;
  Matrix _lambdat;

  const SNP &_snp;
  SNPSamplingG &_pop;

  TSQueue<IndivsList> &_out_q;
  TSQueue<pthread_t> &_in_q;
  CondMutex &_cm;
  IndivsList *_oldilist;
  pthread_t *_idptr;
};
typedef std::map<pthread_t, PhiRunnerG *> ThreadMapG;

class SNPSamplingG {
public:
  SNPSamplingG(Env &env, SNP &snp);
  ~SNPSamplingG();

  void infer();
  bool kv_ok(uint32_t indiv, uint32_t loc) const;
  void load_model(string betafile = "", string thetafile = "");
  void snp_likelihood(uint32_t loc, uint32_t n, Array &p);
  bool hol_mode() const { return _hol_mode; }

  const uArray& shuffled_nodes() const { return _shuffled_nodes; }

  const Matrix &Elogtheta() const   { return _Elogtheta; }
  const D3 &Elogbeta() const        { return _Elogbeta;  }
  const vector<uint32_t> &indivs() const { return _indivs;   }
  const uint32_t sampled_loc() const { return _loc; }
  
  const Matrix &gamma() const  { return _gamma; }
  const D3 &lambda() const     { return _lambda; }

  Matrix &gamma()  { return _gamma; }
  D3 &lambda()     { return _lambda; }
  Matrix &Etheta()  { return _Etheta; }
  Matrix &Elogtheta()  { return _Elogtheta; }

  void update_rho_indiv(uint32_t n);
  const double alpha(uint32_t k) const     { return _alpha[k]; }
  const double rho_indiv(uint32_t n) const { return _rho_indiv[n]; }

  YArray &y() { return *_y; }
  const YArray &y() const { return *_y; }

  YArray &prev_y() { return *_prev_y; }
  const YArray &prev_y() const { return *_prev_y; }

private:
  void init_heldout_sets();
  void set_test_sample();
  void set_validation_sample();
  void set_validation_sample2();
  void infer_init_phase();

  void update_phis_until_conv(uint32_t loc);
  void update_lambda(uint32_t loc);
  void update_phimom(uint32_t n, uint32_t loc);
  void update_phidad(uint32_t n, uint32_t loc);
  void optimize_lambda(uint32_t loc);

  void estimate_beta(uint32_t loc);
  double logl();

  void compute_all_lambda();
  void compute_and_save_beta();
  void save_beta();
  void save_beta(const vector<uint32_t> &locs);
  void save_gamma();
  void save_model();
  void load_gamma();
  void compute_lambda();
  void estimate_all_beta();

  int start_threads();
  void split_all_indivs();
  double compute_likelihood(bool first, bool validation);

  void init_gamma();
  void init_lambda();

  void update_gamma();
  void update_lambda();

  void get_subsample(uint32_t loc);
  void get_subsample_nonuniform();
  uint32_t duration() const;

  void estimate_beta();
  
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
  uint32_t _nthreads;

  uint32_t _iter;
  uint32_t _x;
  Array _alpha;
  uint32_t _loc;

  Matrix _eta;

  vector<uint32_t> _heldout_loc;
  vector<uint32_t> _validation_loc;
  gsl_rng *_r;

  Matrix _gamma;
  D3 _lambda;
  Matrix _lambdat;

  double _tau0;
  double _kappa;
  double _nodetau0;
  double _nodekappa;

  Array _rho_indiv;
  uArray _c_indiv;
  
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

  uArray _shuffled_nodes;
  vector<uint32_t> _indivs;

  double _max_t, _max_h, _max_v, _prev_h, _prev_w, _prev_t;
  mutable uint32_t _nh, _nt;
  uint32_t _sampled_loc;
  uint64_t _total_locations;

  TSQueue<IndivsList> _out_q;
  TSQueue<pthread_t> _in_q;
  CondMutex _cm;
  ThreadMapG _thread_map;
  ChunkMap _chunk_map;
  BoolMap64 _cthreads;
  bool _hol_mode;

  Matrix _phimom;
  Matrix _phidad;
  Array _phinext;
  Matrix _lambdaold;
  Matrix _v;

  YArray *_y;
  YArray *_prev_y;

  YArrayMap _heldout_loc_y;
};

inline void
PhiRunnerG::reset(uint32_t loc)
{
  _lambdat.zero();
  _loc = loc;
  _prev_iter = _iter;
  _prev_hol_mode = _pop.hol_mode();
  _prev_x = 0;
}

inline void
PhiRunnerG::update_phimom(uint32_t n)
{
  //_phinext.zero();
  const double ** const elogthetad = _pop.Elogtheta().const_data();
  const double *** const elogbetad = _pop.Elogbeta().const_data(); 
  for (uint32_t k = 0; k < _k; ++k)
    _phinext[k] = elogthetad[n][k] + elogbetad[_loc][k][0];
  _phinext.lognormalize();
  _phimom.set_elements(n, _phinext);
  debug("n = %d, phimom = %s", n, _phinext.s().c_str());
}

inline void
PhiRunnerG::update_phidad(uint32_t n)
{
  //_phinext.zero();
  const double ** const elogthetad = _pop.Elogtheta().const_data();
  const double *** const elogbetad = _pop.Elogbeta().const_data();
  for (uint32_t k = 0; k < _k; ++k)
    _phinext[k] = elogthetad[n][k] + elogbetad[_loc][k][1];
  _phinext.lognormalize();
  _phidad.set_elements(n, _phinext);
  debug("n = %d, phidad = %s", n, _phinext.s().c_str());
}

inline void
PhiRunnerG::update_phis_all()
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
SNPSamplingG::duration() const
{
  time_t t = time(0);
  return t - _start_time;
}

inline double
SNPSamplingG::snp_likelihood(uint32_t loc, vector<uint32_t> &indivs, bool first)
{
  get_subsample(loc);
  const yval_t * const snpd = _y->const_data();

  if (first)
    estimate_beta(loc);
  else {
    _loc = loc;
    optimize_lambda(loc);
    _iter++;
  }

  const double ** const thetad = _Etheta.const_data();
  const double ** const betad = _Ebeta.const_data();
  double lsum = .0;
  for (uint32_t i = 0; i < indivs.size(); ++i)  {
    uint32_t n = indivs[i];
    assert (!_snp.is_missing(n, loc));

    double sum = .0;

    yval_t x = snpd[n];
    //yval_t x = snpd[n][loc];
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

inline bool
SNPSamplingG::kv_ok(uint32_t indiv, uint32_t loc) const
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
SNPSamplingG::logcoeff(yval_t x) {
  uint32_t c = 2;
  return log(gsl_sf_fact(c)) - log(gsl_sf_fact(x) * gsl_sf_fact(c - x));
}

inline int
PhiRunnerG::process(const IndivsList &v)
{
  double u = 1./_k;
  for (uint32_t i = 0; i < v.size(); ++i) {
    uint32_t n = v[i];
    if (!_pop.kv_ok(n, _loc))
      continue;
    
    _phimom.set_elements(n, u);
    _phidad.set_elements(n, u);
    update_phimom(n);
    update_phidad(n);
  }
  update_lambda_t(v);
}

inline void
PhiRunnerG::update_gamma()
{
  update_gamma(*_oldilist);
}

inline void
PhiRunnerG::estimate_theta()
{
  estimate_theta(*_oldilist);
}

#endif
