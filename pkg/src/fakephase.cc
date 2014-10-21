#include "fakephase.hh"
#include "log.hh"
#include <sys/time.h>

FakePhase::FakePhase(Env &env, SNP &snp)
  :_env(env), _snp(snp),
   _n(env.n), _k(env.k), _l(_env.l),
   _t(env.t), _iter(0), _alpha(_k),
   _eta(_k,_t),
   _gamma(_n,_k), 
   _theta(_n,_k), _beta(_l, _k),
   _tau0(env.tau0 + 1), _kappa(env.kappa),
   _nodetau0(env.nodetau0 + 1), _nodekappa(env.nodekappa),
   _rhot(.0), _noderhot(_n), _nodec(_n),
   _nodeupdatec(_n),
   _start_time(time(0)),
   _Elogtheta(_n,_k),
   _Elogfdad(_n,_k), _Elogfmom(_n,_k),
   _Etheta(_n,_k),
   _pcomp(env, &_r, _iter, _n, _k, 0, _t,
	  _Elogtheta, _Elogfdad, _Elogfmom,
	  _snp, _eta, *this),
   _shuffled_nodes(_n),
   _max_t(-2147483647),
   _max_h(-2147483647),
   _prev_h(-2147483647),
   _prev_w(-2147483647),
   _prev_t(-2147483647),
   _nh(0), _nt(0),
   _training_done(false),
   _sampled_loc(0),
   _total_locations(0)
#ifdef BATCH
  ,_gphi(_n,_k)
#endif
{
  printf("+ popinf initialization begin\n");
  fflush(stdout);

  _total_locations = _n * _l;

  info("+ running inference on %lu nodes\n", _n);
  Env::plog("individuals n", _n);
  Env::plog("locations l", _l);
  Env::plog("populations k", _k);

  _alpha.set_elements(env.alpha);
  info("alpha set to %s\n", _alpha.s().c_str());

  double **d = _eta.data();
  for (uint32_t i = 0; i < _eta.m(); ++i) {
    d[i][0] = 1.0 / _env.k;
    d[i][1] = 1.0 / _env.k;
  }

  // random number generation
  gsl_rng_env_setup();
  const gsl_rng_type *T = gsl_rng_default;
  _r = gsl_rng_alloc(T);
  if (env.seed)
    gsl_rng_set(_r, _env.seed);

  shuffle_nodes();

  unlink(Env::file_str("/likelihood-analysis.txt").c_str());

  _hef = fopen(Env::file_str("/heldout-locs.txt").c_str(), "w");
  if (!_hef)  {
    lerr("cannot open heldout pairs file:%s\n",  strerror(errno));
    exit(-1);
  }

  _vef = fopen(Env::file_str("/validation-locs.txt").c_str(), "w");
  if (!_vef)  {
    lerr("cannot open validation edges file:%s\n",  strerror(errno));
    exit(-1);
  }

  _tef = fopen(Env::file_str("/training-locs.txt").c_str(), "w");
  if (!_tef)  {
    lerr("cannot open training edges file:%s\n",  strerror(errno));
    exit(-1);
  }

  init_heldout();
  info("+ done heldout\n");
  
  info("+ initializing gamma\n");
  init_gamma();
  info("+ done initializing gamma\n");

  _pcomp.reset(0);

  // initialize expectations
  printf("+ Elogtheta and Elogbeta\n");
  PopLib::set_dir_exp(_gamma, _Elogtheta);

  info("+ done Elogtheta and Elogbeta\n");
  info("Elogtheta = %s", _Elogtheta.s().c_str());

  _hf = fopen(Env::file_str("/heldout.txt").c_str(), "w");
  if (!_hf)  {
    lerr("cannot open heldout file:%s\n",  strerror(errno));
    exit(-1);
  }

  _vf = fopen(Env::file_str("/validation.txt").c_str(), "w");
  if (!_vf)  {
    lerr("cannot open validation file:%s\n",  strerror(errno));
    exit(-1);
  }

  _trf = fopen(Env::file_str("/training.txt").c_str(), "w");
  if (!_trf)  {
    lerr("cannot open training file:%s\n",  strerror(errno));
    exit(-1);
  }

  _lf = fopen(Env::file_str("/logl.txt").c_str(), "w");
  if (!_lf)  {
    lerr("cannot open logl file:%s\n",  strerror(errno));
    exit(-1);
  }

  printf("+ computing initial heldout likelihood\n");
  if (!_env.loadcmp)
    estimate_all_theta();
  heldout_likelihood(true);
  save_gamma();
  printf("\n+ computing initial training likelihood\n");

  //training_likelihood(true);
  printf("+ done..\n");

  if (_env.compute_logl) {
    approx_log_likelihood();
    save_model();
  }

  gettimeofday(&_last_iter, NULL);
  printf("+ popinf initialization end\n");
  fflush(stdout);
}

FakePhase::~FakePhase()
{
  fclose(_lf);
  fclose(_trf);
}

void
FakePhase::init_heldout()
{
  set_heldout_sample();
  set_training_sample();

  Env::plog("heldout ratio", _env.heldout_ratio);
}

void
FakePhase::set_heldout_sample()
{
  uint32_t per_loc_h = _n * _env.heldout_ratio * 100 / 5;
  uint32_t nlocs = _l * _env.heldout_ratio;
  map<uint32_t, bool> lm;
  do {
    uint32_t loc = gsl_rng_uniform_int(_r, _l);
    map<uint32_t, bool>::const_iterator z = lm.find(loc);
    if (z != lm.end()) 
      continue;
    else
      lm[loc] = true;
    
    uint32_t c = 0;
    while (c < per_loc_h) {
      uint32_t indiv = gsl_rng_uniform_int(_r, _n);
      if (kv_ok(indiv, loc)) {
	KV kv(indiv, loc);
	_heldout_map[kv] = true;
	c++;
      }
    }
  } while (lm.size() < nlocs);
  Env::plog("heldout snps per location", per_loc_h);
  Env::plog("heldout locations", nlocs);
  Env::plog("total heldout snps", per_loc_h * nlocs);
  Env::plog("total heldout snps (check)", _heldout_map.size());
}

void
FakePhase::set_training_sample()
{
  uint32_t per_loc_h = _n * _env.heldout_ratio;
  uint32_t nlocs = _l * _env.heldout_ratio;
  map<uint32_t, bool> lm;
  do {
    uint32_t loc = gsl_rng_uniform_int(_r, _l);
    map<uint32_t, bool>::const_iterator z = lm.find(loc);
    if (z != lm.end()) 
      continue;
    else
      lm[loc] = true;
    
    uint32_t c = 0;
    while (c < per_loc_h) {
      uint32_t indiv = gsl_rng_uniform_int(_r, _n);
      if (kv_ok(indiv, loc)) {
	KV kv(indiv, loc);
	SNPMap::const_iterator u = _training_map.find(kv);
	if (u == _training_map.end()) {
	  _training_map[kv] = true;
	  c++;
	}
      }
    }
  } while (lm.size() < nlocs);
  Env::plog("training snps per location", per_loc_h);
  Env::plog("training locations", nlocs);
  Env::plog("total snps for training likelihood", per_loc_h * nlocs);
  Env::plog("total snps for training likelihood (check)", _training_map.size());
}

void
FakePhase::shuffle_nodes()
{
  for (uint32_t i = 0; i < _n; ++i)
    _shuffled_nodes[i] = i;
  gsl_ran_shuffle(_r, (void *)_shuffled_nodes.data(), _n, sizeof(uint32_t));
}

void
FakePhase::init_gamma()
{
  double **d = _gamma.data();
  for (uint32_t i = 0; i < _n; ++i) {
    for (uint32_t j = 0; j < _k; ++j)  {
      double v = (_k < 100) ? 1.0 : (double)100.0 / _k;
      d[i][j] = gsl_ran_gamma(_r, 100 * v, 0.01);
    }
  }
}

#ifdef BATCH
void
FakePhase::batch_infer()
{
  double **gphid = _gphi.data();
  while (1) {
    _gphi.zero();
    PopLib::set_dir_exp(_gamma, _Elogtheta);

    // E-step
    for (uint32_t l = 0; l < _l; ++l) {

      printf("\rbatch: %d locations done", l);
      fflush(stdout);

      _pcomp.reset(l);
      _pcomp.update_phis_until_conv();
      
      const Matrix &phidad = _pcomp.phidad();
      const Matrix &phimom = _pcomp.phimom();
      const double **phidadd = phidad.data();
      const double **phimomd = phimom.data();
      
      for (uint32_t n = 0; n < _n; ++n) {
	if (!kv_ok(n, l))
	  continue;
	for (uint32_t k = 0; k < _k; ++k)
	  gphid[n][k] += phidadd[n][k] + phimomd[n][k];
      }
    }

    // M-step
    double **gd = _gamma.data();
    _gamma.zero();
    for (uint32_t n = 0; n < _n; ++n)
      for (uint32_t k = 0; k < _k; ++k)
	gd[n][k] += _alpha[k] + gphid[n][k];
    
    _iter++;

    lerr("iteration = %d took %d secs (family:%d)\n", 
	 _iter, duration(), _family);
    
    if (_iter % _env.reportfreq == 0) {
      printf("iteration = %d took %d secs (family:%d)\n", 
	     _iter, duration(), _family);
      fflush(stdout);
      lerr("estimating theta @ %d secs", duration());
      estimate_all_theta();
      lerr("computing heldout likelihood @ %d secs", duration());
      heldout_likelihood();
      lerr("saving theta @ %d secs", duration());
      save_model();
      lerr("done @ %d secs", duration());
      if (_env.compute_logl) {
	approx_log_likelihood();
	save_model();
      }
    }
  }
}
#endif

void
FakePhase::infer()
{
  printf("Running FakePhase::infer()\n");
  while (1) {
    _sampled_loc = gsl_rng_uniform_int(_r, _l);

    PopLib::set_dir_exp(_gamma, _Elogtheta);
    
    _pcomp.reset(_sampled_loc);
    _pcomp.update_phis_until_conv();
    
    const Matrix &phidad = _pcomp.phidad();
    const Matrix &phimom = _pcomp.phimom();
    const double **phidadd = phidad.data();
    const double **phimomd = phimom.data();
    
    double scale = _env.l;
    double **gd = _gamma.data();

    for (uint32_t i = 0; i < _n; ++i) {
      if (!kv_ok(i, _sampled_loc))
	continue;
      
      _noderhot[i] = pow(_nodetau0 + _nodec[i], -1 * _nodekappa);
      
      for (uint32_t k = 0; k < _k; ++k) {
	gd[i][k] = gd[i][k] + _noderhot[i] * (_alpha[k] + (scale * (phidadd[i][k] + phimomd[i][k])) - gd[i][k]);
	assert (gd[i][k] >= .0);
      }
      _nodec[i]++;
    }

    tst("GAMMA=%s\n",_gamma.s().c_str());
    fflush(stdout);
    
    _rhot = pow(_tau0 + _iter, -1 * _kappa);
    _iter++;

    if (_iter % 10 == 0)
      lerr("iteration = %d took %d secs (family:%d)\n", 
	   _iter, duration(), _family);

    if (_iter % _env.reportfreq == 0) {
      printf("iteration = %d took %d secs (family:%d)\n", 
	     _iter, duration(), _family);
      fflush(stdout);
      lerr("estimating theta @ %d secs", duration());
      estimate_all_theta();
      lerr("computing heldout likelihood @ %d secs", duration());
      heldout_likelihood();
      lerr("saving theta @ %d secs", duration());
      save_model();
      lerr("done @ %d secs", duration());
      if (_env.compute_logl) {
	approx_log_likelihood();
	save_model();
      }
    }

    if (_env.terminate) {
      save_model();
      exit(0);
    }
  }
}

// assumes E[log theta] has been updated
double
FakePhase::approx_log_likelihood()
{
  const double ** const etad = _eta.const_data();
  const double * const alphad = _alpha.const_data();
  const double ** const elogthetad  = _Elogtheta.const_data();
  const double ** const gd = _gamma.const_data();
  const yval_t ** const xd = _snp.y().const_data();

  double v = .0, s = .0;
  for (uint32_t n = 0; n < _n; ++n) {
    v = .0;
    for (uint32_t k = 0; k < _k; ++k)
      v += gsl_sf_lngamma(alphad[k]);
    s += gsl_sf_lngamma(_alpha.sum()) - v;

    v = .0;
    for (uint32_t k = 0; k < _k; ++k) {
      v += (alphad[k] - 1) * elogthetad[n][k];
    }
    s += v;

    v = .0;
    for (uint32_t k = 0; k < _k; ++k)
      v += gsl_sf_lngamma(gd[n][k]);
    s -= gsl_sf_lngamma(_gamma.sum(n)) - v;

    v = .0;
    for (uint32_t k = 0; k < _k; ++k)
      v += (gd[n][k] - 1) * elogthetad[n][k];
    s -= v;
  }

  double p1 = s;

  double u = .0;
  uint32_t uc = 0;
  for (uint32_t l = 0; l < _l; ++l) {
    _pcomp.reset(l);
    _pcomp.update_phis_until_conv();
    
    const Matrix &phidad = _pcomp.phidad();
    const Matrix &phimom = _pcomp.phimom();
    const double **phidadd = phidad.data();
    const double **phimomd = phimom.data();

    const Matrix &Elogbeta = _pcomp.Elogbeta();
    const Matrix &lambda = _pcomp.lambda();
    const double ** const elogbetad = Elogbeta.const_data();
    const double ** const ld = lambda.const_data();
    
    for (uint32_t n = 0; n < _n; ++n) {
      if (!kv_ok(n, l))
	continue;
      uc++;

      for (uint32_t k = 0; k < _k; ++k) {
	s += phidadd[n][k] * elogthetad[n][k] - phidadd[n][k] * log(phidadd[n][k]);
	s += phimomd[n][k] * elogthetad[n][k] - phimomd[n][k] * log(phimomd[n][k]);
	s += phidadd[n][k] * (_snp.dad(n,l) * elogbetad[k][0] + 
			      (1 - _snp.dad(n,l) * elogbetad[k][1]));
	s += phimomd[n][k] * (_snp.mom(n,l) * elogbetad[k][0] + 
			      (1 - _snp.mom(n,l) * elogbetad[k][1]));
	v = .0; 
	for (uint32_t t = 0; t < _t; ++t)
	  v += gsl_sf_lngamma(etad[k][t]);
	u += gsl_sf_lngamma(_eta.sum(k)) - v;

	v = .0;
	for (uint32_t t = 0; t < _t; ++t)
	  v += (etad[k][t] - 1) * elogbetad[k][t];
	u += v;	

	v = .0;
	for (uint32_t t = 0; t < _t; ++t)
	  v += gsl_sf_lngamma(ld[k][t]);
	u -= gsl_sf_lngamma(lambda.sum(k)) - v;

	v = .0;
	for (uint32_t t = 0; t < _t; ++t)
	  v += (ld[k][t] - 1) * elogbetad[k][t];
	u -= v;
      }
    }
  }
  s += u / uc;

  double p2 = s - p1;

  info("approx. log likelihood = %f\n", s);
  fprintf(_lf, "%d\t%d\t%.5f\t%.5f\t%.5f\n", _iter, duration(), s, p1, p2);
  fflush(_lf);
  return s;
}

double
FakePhase::heldout_likelihood(bool first)
{
  uint32_t k = 0;
  double s = .0;
  SNPByLoc m;
  for (SNPMap::const_iterator i = _heldout_map.begin();
       i != _heldout_map.end(); ++i) {
    const KV &kv = i->first;

    uint32_t indiv = kv.first;
    uint32_t loc = kv.second;

    vector<uint32_t> &v = m[loc];
    v.push_back(indiv);
  }

  vector<uint32_t> indivs;
  uint32_t sz = 0;
  for (SNPByLoc::const_iterator i = m.begin(); i != m.end();
       ++i) {
    uint32_t loc = i->first;
    indivs = i->second;
    printf("\rdone:%.2f%%", ((double)sz / m.size())*100);
    double u = snp_likelihood(loc, indivs, first);
    s += u;
    k += indivs.size();
    sz++;
  }
  fprintf(_hf, "%d\t%d\t%.9f\t%d\t%f\n", _iter, duration(), (s / k), k, exp(s/k));
  fflush(_hf);
  
  double a = (s / k);
  bool stop = false;
  int why = -1;
  if (_iter > 2000) {
    if (a > _prev_h && 
	_prev_h != 0 && fabs((a - _prev_h) / _prev_h) < 0.00001) {
      stop = true;
      why = 0;
    } else if (a < _prev_h)
      _nh++;
    else if (a > _prev_h)
      _nh = 0;

    if (a > _max_h)
      _max_h = a;
    
    if (_nh > 1) {
      why = 1;
      stop = true;
    }
  }
  _prev_h = a;

  if (stop) {
    double v = 0; //validation_likelihood();
    double t = 0; //t = training_likelihood();

     FILE *f = fopen(Env::file_str("/max.txt").c_str(), "w");
    fprintf(f, "%d\t%d\t%.5f\t%.5f\t%.5f\t%.5f\t%d\n",
	    _iter, duration(),
	    a, t, v, _max_h,
	    why);
    fclose(f);

    if (_env.use_validation_stop) {
      save_model();
      save_beta();
      exit(0);
    }
  }
  return (s / k) / _n;
}

double
FakePhase::training_likelihood(bool first)
{
  uint32_t k = 0;
  double s = .0;
  SNPByLoc m;
  for (SNPMap::const_iterator i = _training_map.begin();
       i != _training_map.end(); ++i) {
    const KV &kv = i->first;

    uint32_t indiv = kv.first;
    uint32_t loc = kv.second;
    
    vector<uint32_t> &v = m[loc];
    v.push_back(indiv);
  }

  vector<uint32_t> indivs;
  uint32_t sz = 0;
  for (SNPByLoc::const_iterator i = m.begin(); i != m.end();
       ++i) {
    uint32_t loc = i->first;
    indivs = i->second;
    printf("\rdone:%.2f%%", ((double)sz / m.size())*100);
    double u = snp_likelihood(loc, indivs, first);
    s += u;
    k += indivs.size();
    sz++;
  }
  fprintf(_trf, "%d\t%d\t%.9f\t%d\t%f\n", _iter, duration(), (s / k), k, exp(s/k));
  fflush(_trf);
  return s / k ;
}

void
FakePhase::save_gamma()
{
  FILE *f = fopen(Env::file_str("/gamma.txt").c_str(), "w");
  FILE *g = fopen(Env::file_str("/theta.txt").c_str(), "w");
  if (!f || !g)  {
    lerr("cannot open gamma/theta file:%s\n",  strerror(errno));
    exit(-1);
  }
  double **gd = _gamma.data();
  double **td = _Etheta.data();
  for (uint32_t n = 0; n < _n; ++n) {
    string s = _snp.label(n);
    if (s == "")
      s = "unknown";
    fprintf(f, "%d\t%s\t", n, s.c_str());
    fprintf(g, "%d\t%s\t", n, s.c_str());
    double max = .0;
    uint32_t max_k = 0;
    for (uint32_t k = 0; k < _k; ++k) {
      fprintf(f, "%.8f\t", gd[n][k]);
      fprintf(g, "%.8f\t", td[n][k]);
      if (gd[n][k] > max) {
	max = gd[n][k];
	max_k = k;
      }
    }
    fprintf(f,"%d\n", max_k);
    fprintf(g,"%d\n", max_k);
  }
  fclose(f);
  fclose(g);
}

void
FakePhase::save_beta()
{
  FILE *f = fopen(Env::file_str("/beta.txt").c_str(), "w");
  FILE *g = fopen(Env::file_str("/lambda.txt").c_str(), "w");
  if (!f || !g)  {
    lerr("cannot open beta or lambda file:%s\n",  strerror(errno));
    exit(-1);
  }
  for (uint32_t l = 0; l < _l; ++l) {
    _pcomp.reset(l);
    _pcomp.update_phis_until_conv();
    fprintf(f, "%d\t", l);
    fprintf(g, "%d\t", l);
    const Array &beta = _pcomp.beta();
    const double ** const lambdad = _pcomp.lambda().const_data();
    for (uint32_t k = 0; k < _k; ++k) {
      fprintf(f, "%.8f\t", beta[k]);
      for (uint32_t t = 0; t < _t; ++t)
	fprintf(g, "%.8f\t", lambdad[k][t]);
    }
    fprintf(f, "\n");
    fprintf(g, "\n");
  }
  fclose(f);
  fclose(g);
}

void
FakePhase::save_model()
{
  save_gamma();
}

void
FakePhase::load_model(string betafile, string thetafile)
{
  double **thetad = _Etheta.data();
  double **betad = _beta.data();
  if (betafile == "")
    betafile = Env::file_str("/beta.txt");
  if (thetafile == "")
    thetafile = Env::file_str("/theta.txt");

  FILE *betaf = fopen(betafile.c_str(), "r");
  if (!betaf)  {
    lerr("cannot open beta file:%s\n",  strerror(errno));
    exit(-1);
  }
  
  int sz = 128 * _k;
  uint32_t l = 0;
  char *line = (char *)malloc(sz);
  while (!feof(betaf)) {
    if (fgets(line, sz, betaf) == NULL) 
      break;
    
    uint32_t k = 0;
    char *p = line;
    //printf("line = %s", line);
    //fflush(stdout);
    do {
      char *q = NULL;
      double d = strtod(p, &q);
      if (p == q) {
	if (k < _k - 1) {
	  fprintf(stderr, "error parsing beta file\n");
	  assert(0);
	}
	break;
      }
      p = q;
      if (k >= 1)
	betad[l][k-1] = d;
      k++;
    } while (p != NULL);
    l++;
    memset(line, 0, sz);
  }
  assert (l = _l);
  fclose(betaf);

  FILE *thetaf = fopen(thetafile.c_str(), "r");
  if (!thetaf)  {
    lerr("cannot open theta file:%s\n",  strerror(errno));
    exit(-1);
  }

  uint32_t n = 0;
  while (!feof(thetaf)) {
    if (fgets(line, sz, thetaf) == NULL) 
      break;

    uint32_t k = 0;
    char *p = line;
    //printf("line = %s", line);
    //fflush(stdout);
    do {
      char *q = NULL;
      if (k == 1) {
	p += 8;
	k++;
	continue;
      }
      double d = strtod(p, &q);
      if (p == q) {
	if (k < _k - 1) {
	  fprintf(stderr, "error parsing theta file\n");
	  assert(0);
	}
	break;
      }
      p = q;
      if (k >= 2) 
	thetad[n][k-2] = d;
      k++;
    } while (p != NULL);
    n++;
    memset(line, 0, sz);
  }
  assert (n = _n);
  fclose(thetaf);
}
