#include "snpsamplinga.hh"
#include "log.hh"
#include <sys/time.h>

SNPSamplingA::SNPSamplingA(Env &env, SNP &snp)
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
   _Etheta(_n,_k),
   _pcomp(env, &_r, _iter, _n, _k, 0, _t,
	  _Elogtheta, 
	  _snp, _eta, *this),
   _shuffled_nodes(_n),
   _max_t(-2147483647),
   _max_h(-2147483647),
   _prev_h(-2147483647),
   _prev_w(-2147483647),
   _prev_t(-2147483647),
   _nh(0), _nt(0),
   _sampled_loc(0),
   _total_locations(0)
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
    d[i][0] = 1.0;
    d[i][1] = 1.0;
  }

  // random number generation
  gsl_rng_env_setup();
  const gsl_rng_type *T = gsl_rng_default;
  _r = gsl_rng_alloc(T);
  if (env.seed)
    gsl_rng_set(_r, _env.seed);

  // XXX: shuffle periodically?
  shuffle_nodes();
  
  unlink(Env::file_str("/likelihood-analysis.txt").c_str());

  _vf = fopen(Env::file_str("/validation.txt").c_str(), "w");
  if (!_vf)  {
    printf("cannot open heldout file:%s\n",  strerror(errno));
    exit(-1);
  }

  _tf = fopen(Env::file_str("/test.txt").c_str(), "w");
  if (!_tf)  {
    printf("cannot open heldout file:%s\n",  strerror(errno));
    exit(-1);
  }

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

  init_heldout_sets();
  
  info("+ initializing gamma\n");
  init_gamma();
  info("+ done initializing gamma\n");

  _pcomp.reset(0);

  // initialize expectations
  printf("+ Elogtheta and Elogbeta\n");
  PopLib::set_dir_exp(_gamma, _Elogtheta);

  info("+ done Elogtheta and Elogbeta\n");
  info("Elogtheta = %s", _Elogtheta.s().c_str());


  _lf = fopen(Env::file_str("/logl.txt").c_str(), "w");
  if (!_lf)  {
    lerr("cannot open logl file:%s\n",  strerror(errno));
    exit(-1);
  }

  printf("+ computing initial heldout likelihood\n");
  if (!_env.loadcmp)
    estimate_all_theta();
  compute_likelihood(true, true);
  compute_likelihood(true, false);
  save_gamma();
  printf("\n+ computing initial training likelihood\n");

  //training_likelihood(true);
  printf("+ done..\n");

  if (_env.compute_logl) {
    save_model();
  }

  gettimeofday(&_last_iter, NULL);
  printf("+ popinf initialization end\n");
  fflush(stdout);
}

SNPSamplingA::~SNPSamplingA()
{
  fclose(_vf);
  fclose(_tf);
  fclose(_lf);
  fclose(_tef);
  fclose(_vef);
}

void
SNPSamplingA::init_heldout_sets()
{
  set_test_sample();
  set_validation_sample();

  Env::plog("test ratio", _env.test_ratio);
  Env::plog("validation ratio", _env.validation_ratio);
}

void
SNPSamplingA::set_test_sample()
{
  uint32_t per_loc_h = _n * _env.test_ratio * 100 / 5;
  uint32_t nlocs = _l * _env.test_ratio;
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
	_test_map[kv] = true;
	c++;
      }
    }
  } while (lm.size() < nlocs);
  Env::plog("test snps per location", per_loc_h);
  Env::plog("test locations", nlocs);
  Env::plog("total test snps", per_loc_h * nlocs);
  Env::plog("total test snps (check)", _test_map.size());
}

void
SNPSamplingA::set_validation_sample()
{
  uint32_t per_loc_h = _n * _env.validation_ratio * 100 / 5;
  uint32_t nlocs = _l * _env.validation_ratio;
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
	_validation_map[kv] = true;
	c++;
      }
    }
  } while (lm.size() < nlocs);
  Env::plog("validation snps per location", per_loc_h);
  Env::plog("validation locations", nlocs);
  Env::plog("total validation snps", per_loc_h * nlocs);
  Env::plog("total validation snps (check)", _validation_map.size());
}

void
SNPSamplingA::shuffle_nodes()
{
  for (uint32_t i = 0; i < _n; ++i)
    _shuffled_nodes[i] = i;
  gsl_ran_shuffle(_r, (void *)_shuffled_nodes.data(), _n, sizeof(uint32_t));
}

void
SNPSamplingA::init_gamma()
{
  double **d = _gamma.data();
  for (uint32_t i = 0; i < _n; ++i) {
    for (uint32_t j = 0; j < _k; ++j)  {
      double v = (_k < 100) ? 1.0 : (double)100.0 / _k;
      d[i][j] = gsl_ran_gamma(_r, 100 * v, 0.01);
    }
  }
  PopLib::set_dir_exp(_gamma, _Elogtheta);
}

void
SNPSamplingA::infer()
{
  printf("Running SNPSamplingA::infer()\n");
  while (1) {
    _sampled_loc = gsl_rng_uniform_int(_r, _l);
    const yval_t ** const snpd = _snp.y().const_data();

    _pcomp.reset(_sampled_loc);
    _pcomp.update_phis_until_conv();
    
    const Matrix &phidad = _pcomp.phidad();
    const Matrix &phimom = _pcomp.phimom();
    const double **phidadd = phidad.data();
    const double **phimomd = phimom.data();
    
    double scale = _env.l;
    double **gd = _gamma.data();

    const vector<uint32_t> &indivs = _pcomp.indiv();
    for (uint32_t i = 0; i < indivs.size(); ++i) {
      uint32_t n = indivs[i];
      if (!kv_ok(n, _sampled_loc))
	continue;

      _noderhot[n] = pow(_nodetau0 + _nodec[n], -1 * _nodekappa);
      
      yval_t y = snpd[n][_sampled_loc];
      for (uint32_t k = 0; k < _k; ++k) {
	gd[n][k] = gd[n][k] + _noderhot[n] *				\
	  (_alpha[k] + (scale * (y * phimomd[n][k] + (2 - y) * phidadd[n][k])) - gd[n][k]);
	assert (gd[n][k] >= .0);
      }
      _nodec[n]++;
    }
    estimate_theta(indivs);

    _iter++;

    if (_iter % 100 == 0)
      lerr("iteration = %d took %d secs (family:%d)\n", 
	   _iter, duration(), _family);

    if (_iter % _env.reportfreq == 0) {
      printf("iteration = %d took %d secs (family:%d)\n", 
	     _iter, duration(), _family);
      fflush(stdout);
      lerr("computing heldout likelihood @ %d secs", duration());
      compute_likelihood(false, true);
      compute_likelihood(false, false);
      lerr("saving theta @ %d secs", duration());
      save_model();
      lerr("done @ %d secs", duration());
    }

    if (_env.terminate) {
      save_model();
      exit(0);
    }
  }
}

void
SNPSamplingA::estimate_theta(const vector<uint32_t> &indivs)
{
  const double ** const gd = _gamma.const_data();
  double **theta = _Etheta.data();
  double **elogtheta = _Elogtheta.data();
  
  for (uint32_t i = 0; i < indivs.size(); ++i)  {
    uint32_t n = indivs[i];
    double s = .0;
    if (!kv_ok(n, _sampled_loc))
      continue;
    for (uint32_t k = 0; k < _k; ++k)
      s += gd[n][k];
    assert(s);
    double psi_sum = gsl_sf_psi(s);
    for (uint32_t k = 0; k < _k; ++k) {
      theta[n][k] = gd[n][k] / s;
      elogtheta[n][k] = gsl_sf_psi(gd[n][k]) - psi_sum;
    }
  }
}

double
SNPSamplingA::compute_likelihood(bool first, bool validation)
{
  uint32_t k = 0;
  double s = .0;

  SNPMap *mp = NULL;
  FILE *ff = NULL;
  if (validation) {
    mp = &_validation_map;
    ff = _vf;
  } else {
    mp = &_test_map;
    ff = _tf;
  }

  SNPByLoc m;
  for (SNPMap::const_iterator i = mp->begin(); i != mp->end(); ++i) {
    const KV &kv = i->first;

    uint32_t indiv = kv.first;
    uint32_t loc = kv.second;

    vector<uint32_t> &v = m[loc];
    v.push_back(indiv);
  }

  vector<uint32_t> indivs;
  uint32_t sz = 0;
  for (SNPByLoc::const_iterator i = m.begin(); i != m.end(); ++i) {
    uint32_t loc = i->first;
    indivs = i->second;
    printf("\rdone:%.2f%%", ((double)sz / m.size())*100);
    double u = snp_likelihood(loc, indivs, first);
    s += u;
    k += indivs.size();
    sz++;
  }
  fprintf(ff, "%d\t%d\t%.9f\t%d\t%f\n", _iter, duration(), (s / k), k, exp(s/k));
  fflush(ff);
  
  double a = (s / k);

  if (!validation)
    return 0;
  
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
    
    if (_nh > 3) {
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
      if (_env.save_beta)
	save_beta();
      exit(0);
    }
  }
  return (s / k) / _n;
}

void
SNPSamplingA::save_gamma()
{
  FILE *f = fopen(add_iter_suffix("/gamma").c_str(), "w");
  FILE *g = fopen(add_iter_suffix("/theta").c_str(), "w");
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

string
SNPSamplingA::add_iter_suffix(const char *c)
{
  ostringstream sa;
  if (_env.file_suffix)
    sa << c << "_" << _iter << ".txt";
  else
    sa << c << ".txt";
  return Env::file_str(sa.str());
}

void
SNPSamplingA::save_beta()
{
  FILE *f = fopen(add_iter_suffix("/beta").c_str(), "w");
  FILE *g = fopen(add_iter_suffix("/lambda").c_str(), "w");
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
SNPSamplingA::save_model()
{
  save_gamma();
}

void
SNPSamplingA::load_model(string betafile, string thetafile)
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
