#include "snpsamplingb.hh"
#include "log.hh"
#include <sys/time.h>

SNPSamplingB::SNPSamplingB(Env &env, SNP &snp)
  :_env(env), _snp(snp),
   _n(env.n), _k(env.k), _l(_env.l),
   _t(env.t), _iter(0), _alpha(_k), _loc(0),
   _eta(_k,_t),
   _gamma(_n,_k), 
   _lambda(_l,_k,_t),
   _lambdat(_k,_t),
   _gamma_ag(NULL), _lambda_ag(NULL),
   _tau0(env.tau0 + 1), _kappa(env.kappa),
   _nodetau0(env.nodetau0 + 1), _nodekappa(env.nodekappa),
   _rho_indiv(_n), _rho_loc(_l),
   _c_indiv(_n), _c_loc(_l),
   _nodeupdatec(_n),
   _start_time(time(0)),
   _Elogtheta(_n,_k),
   _Elogbeta(_l,_k,_t),
   _Etheta(_n,_k),
   _Ebeta(_l,_k),
   _pcomp(env, &_r, _iter, _n, _k, 0, _t, _snp, *this),
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
  if (_env.adagrad) {
    _gamma_ag = new Matrix(_n,_k);
    _lambda_ag = new D3(_l,_k,_t);
  }

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
  init_lambda();
  info("+ done initializing gamma\n");

  _pcomp.reset(0);

  _lf = fopen(Env::file_str("/logl.txt").c_str(), "w");
  if (!_lf)  {
    lerr("cannot open logl file:%s\n",  strerror(errno));
    exit(-1);
  }
  estimate_all_theta();

  printf("+ computing initial heldout likelihood\n");
  compute_likelihood(true, true);
  //compute_likelihood(true, false);
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

SNPSamplingB::~SNPSamplingB()
{
  fclose(_vf);
  fclose(_tf);
  fclose(_lf);
  fclose(_tef);
  fclose(_vef);
}

void
SNPSamplingB::init_heldout_sets()
{
  //set_test_sample();
  set_validation_sample();

  Env::plog("test ratio", _env.test_ratio);
  Env::plog("validation ratio", _env.validation_ratio);
}

void
SNPSamplingB::set_test_sample()
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
SNPSamplingB::set_validation_sample2()
{
  for (uint32_t l = 0; l < _l; ++l) {
    // for each location keep aside h individuals
    uint32_t h = _env.heldout_indiv_ratio * _n;
    if (h < 1)
      h = 1;
    else if (h > 10)
      h = 10;
    
    uint32_t c = 0;
    do {
      uint32_t indiv = gsl_rng_uniform_int(_r, _n);
      if (kv_ok(indiv, l)) {
	KV kv(indiv, l);
	_validation_map[kv] = true;
	c++;
      }
    } while (c < h);
  }
  Env::plog("(VAL2) total validation snps", _validation_map.size());
}

void
SNPSamplingB::set_validation_sample()
{
  uint32_t per_loc_h = 0;
  if (_n <= 1000)
    per_loc_h = _n * _env.validation_ratio * 100 / 5;
  else
    per_loc_h = _n * _env.validation_ratio * 10 / 5;
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
  Env::plog("(VAL1) total validation snps (check)", _validation_map.size());
}

void
SNPSamplingB::shuffle_nodes()
{
  for (uint32_t i = 0; i < _n; ++i)
    _shuffled_nodes[i] = i;
  gsl_ran_shuffle(_r, (void *)_shuffled_nodes.data(), _n, sizeof(uint32_t));
}

void
SNPSamplingB::init_gamma()
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
SNPSamplingB::init_lambda()
{
  double ***ld = _lambda.data();
  const double **etad = _eta.const_data();
  for (uint32_t l = 0; l < _l; ++l)
    for (uint32_t k = 0; k < _k; ++k)
      for (uint32_t t = 0; t < _t; ++t) {
	double v = (_k <= 100) ? 1.0 : (double)100.0 / _k;
	ld[l][k][t] = etad[k][t] + gsl_ran_gamma(_r, 100 * v, 0.01);
      }
  PopLib::set_dir_exp(_lambda, _Elogbeta);
}

void
SNPSamplingB::get_subsample()
{
  // get subsample of individuals
  double v = (double)(gsl_rng_uniform_int(_r, _n)) / _env.indiv_sample_size;
  uint32_t q = ((int)v) * _env.indiv_sample_size;
  _indivs.clear();
  while (_indivs.size() < _env.indiv_sample_size) {
    uint32_t n = _shuffled_nodes[q];
    if (!kv_ok(n, _loc)) {
      q = (q + 1) % _n;
      continue;
    }
    _indivs.push_back(n);
    q = (q + 1) % _n;
    continue;
  }
}

void
SNPSamplingB::update_rho_indiv(uint32_t n)
{
  _rho_indiv[n] = pow(_nodetau0 + _c_indiv[n], -1 * _nodekappa);
  _c_indiv[n]++;
}

void
SNPSamplingB::update_rho_loc(uint32_t l)
{
  _rho_loc[l] = pow(_tau0 + _c_loc[l], -1 * _kappa);
  _c_loc[l]++;
}

void
SNPSamplingB::update_gamma()
{
  const Matrix &phidad = _pcomp.phidad();
  const Matrix &phimom = _pcomp.phimom();
  const double **phidadd = phidad.data();
  const double **phimomd = phimom.data();
  const yval_t ** const snpd = _snp.y().const_data();

  double gamma_scale = _env.l;
  double **gd = _gamma.data();

  for (uint32_t i = 0; i < _indivs.size(); ++i) {
    uint32_t n = _indivs[i];
    if (!kv_ok(n, _loc))
      continue;
    
    update_rho_indiv(n);
    
    yval_t y = snpd[n][_loc];
    for (uint32_t k = 0; k < _k; ++k) {
      double gk = _alpha[k] + (gamma_scale * 
			       (y * phimomd[n][k] + (2 - y) * phidadd[n][k])) - gd[n][k];
      if (_env.adagrad)  {
	double **gd_ag = _gamma_ag->data();
	assert(gd_ag);
	gd_ag[n][k] += gk * gk;
	gd[n][k] += gk / sqrt(gd_ag[n][k]);
      } else
	gd[n][k] += _rho_indiv[n] * gk;
      assert (gd[n][k] >= .0);
      if (n == 30) {
	debug("gamma: n:%d, k:%d -> %f\n", n, k, gd[n][k]);
      }
    }
  }
}

void
SNPSamplingB::update_lambda()
{
  const Matrix &phidad = _pcomp.phidad();
  const Matrix &phimom = _pcomp.phimom();
  const double **phidadd = phidad.data();
  const double **phimomd = phimom.data();
  const yval_t ** const snpd = _snp.y().const_data();

  double lambda_scale = (double)_n / _indivs.size();
  double **ld = _lambda.data()[_loc];
  double **ldt = _lambdat.data();
  
  _lambdat.zero();
  for (uint32_t k = 0; k < _k; ++k) {
    for (uint32_t i = 0; i < _indivs.size(); ++i)  {
      uint32_t n = _indivs[i];
      if (!kv_ok(n, _loc))
	continue;
      ldt[k][0] += phimomd[n][k] * snpd[n][_loc];
      ldt[k][1] += phidadd[n][k] * (2 - snpd[n][_loc]);
    }
  }

  debug("lambdat = %s", _lambdat.s().c_str());
    
  update_rho_loc(_loc);
  for (uint32_t k = 0; k < _k; ++k) {
    double lk0 = _env.eta0 + lambda_scale * ldt[k][0] - ld[k][0];
    double lk1 = _env.eta1 + lambda_scale * ldt[k][1] - ld[k][1];
    if (_env.adagrad)  {
      double ***ld_ag = _lambda_ag->data();
      assert(ld_ag);
      ld_ag[_loc][k][0] += lk0 * lk0;
      ld_ag[_loc][k][1] += lk1 * lk1;
      ld[k][0] += lk0 / sqrt(ld_ag[_loc][k][0]);
      ld[k][1] += lk1 / sqrt(ld_ag[_loc][k][1]);
    } else {
      ld[k][0] += _rho_loc[_loc] * lk0;
      ld[k][1] += _rho_loc[_loc] * lk1;
      debug("lambda: k = %d: (%f,%f)", k, ld[k][0], ld[k][1]);
    }
  }
}

void
SNPSamplingB::estimate_theta()
{
  const double ** const gd = _gamma.const_data();
  double **theta = _Etheta.data();
  double **elogtheta = _Elogtheta.data();
  
  for (uint32_t i = 0; i < _indivs.size(); ++i)  {
    uint32_t n = _indivs[i];
    double s = .0;
    if (!kv_ok(n, _loc))
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

void
SNPSamplingB::estimate_beta()
{
  const double ***ld = _lambda.const_data();
  double **betad = _Ebeta.data();
  double ***elogbeta = _Elogbeta.data();

  for (uint32_t k = 0; k < _k; ++k) {
    double s = .0;
    for (uint32_t t = 0; t < _t; ++t)
      s += ld[_loc][k][t];
    betad[_loc][k] = ld[_loc][k][0] / s;
    
    double psi_sum = gsl_sf_psi(s);
    elogbeta[_loc][k][0] = gsl_sf_psi(ld[_loc][k][0]) - psi_sum;
    elogbeta[_loc][k][1] = gsl_sf_psi(ld[_loc][k][1]) - psi_sum;
  }
}

void
SNPSamplingB::infer()
{
  printf("Running SNPSamplingB::infer()\n");
  while (1) {
    _loc = gsl_rng_uniform_int(_r, _l);
    debug("sampled loc = %d\n", _loc);
    const yval_t ** const snpd = _snp.y().const_data();
    get_subsample();
    
    _pcomp.reset(_loc);
    _pcomp.update_phis();

    update_gamma();
    update_lambda();

    estimate_theta();
    estimate_beta();

    debug("Etheta = %s", _Etheta.s().c_str());
    debug("Elogtheta = %s", _Elogtheta.s().c_str());
    debug("Ebeta = %s", _Ebeta.s().c_str());
    debug("Elogbeta = %s", _Elogbeta.s().c_str());

    _iter++;

    if (_iter % _env.reportfreq == 0) {
      printf("iteration = %d took %d secs\n", 
	     _iter, duration());
      fflush(stdout);
      lerr("computing heldout likelihood @ %d secs", duration());
      compute_likelihood(false, true);
      //compute_likelihood(false, false);
      lerr("saving theta @ %d secs", duration());
      save_model();
      lerr("done @ %d secs", duration());

      if (_env.compute_logl) 
	logl();
    }

    //if (_iter % 1000 == 0)
    //gsl_ran_shuffle(_r, (void *)_shuffled_nodes.data(), _n, sizeof(uint32_t));
    
    if (_env.terminate) {
      save_model();
      exit(0);
    }
  }
}

double
SNPSamplingB::compute_likelihood(bool first, bool validation)
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
	_prev_h != 0 && fabs((a - _prev_h) / _prev_h) < 0.0000001) {
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
SNPSamplingB::save_gamma()
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
SNPSamplingB::add_iter_suffix(const char *c)
{
  ostringstream sa;
  if (_env.file_suffix)
    sa << c << "_" << _iter << ".txt";
  else
    sa << c << ".txt";
  return Env::file_str(sa.str());
}

void
SNPSamplingB::save_model()
{
  save_gamma();
}

void
SNPSamplingB::load_model(string betafile, string thetafile)
{
  double **thetad = _Etheta.data();
  double **betad = _Ebeta.data();
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

void
SNPSamplingB::estimate_all_theta()
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


// assumes E[log theta] has been updated
double
SNPSamplingB::logl()
{
  const double ** const etad = _eta.const_data();
  const double * const alphad = _alpha.const_data();
  const double ** const elogthetad  = _Elogtheta.const_data();
  const double ** const gd = _gamma.const_data();

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
  const double ***elogbeta = _Elogbeta.const_data();
  const double **elogtheta = _Elogtheta.const_data();
  const double **ebeta = _Ebeta.const_data();
  const double ***ld = _lambda.const_data();
  const yval_t ** const snpd = _snp.y().const_data();

  for (uint32_t l = 0; l < _l; ++l) {
    _pcomp.reset(l);
    _pcomp.update_phis_all();
    
    const Matrix &phimom = _pcomp.phimom();
    const Matrix &phidad = _pcomp.phidad();
    const double **phimomd = phimom.const_data();
    const double **phidadd = phidad.const_data();
    for (uint32_t n = 0; n < _n; ++n) {
      if (!kv_ok(n, l))
	continue;
      
      yval_t y = snpd[n][l];

      for (uint32_t k = 0; k < _k; ++k) {
	double x0 = elogtheta[n][k] + elogbeta[l][k][0];
	double x1 = elogtheta[n][k] + elogbeta[l][k][1];

	s += y * (phimomd[n][k] * x0 - phimomd[n][k] * log(phimomd[n][k]));
	s += (2 - y) * (phidadd[n][k] * x1 - phidadd[n][k] * log(phidadd[n][k]));
      }
      s += logcoeff(y);
    }
  }
  double p2 = s - p1;
  
  for (uint32_t l = 0; l < _l; ++l) {
    for (uint32_t k = 0; k < _k; ++k) {
      v = .0; 
      for (uint32_t t = 0; t < _t; ++t)
	v += gsl_sf_lngamma(etad[k][t]);
      s += gsl_sf_lngamma(_eta.sum(k)) - v;
      
      v = .0;
      for (uint32_t t = 0; t < _t; ++t)
	v += (etad[k][t] - 1) * elogbeta[l][k][t];
      s += v;	
      
      v = .0;
      for (uint32_t t = 0; t < _t; ++t)
	v += gsl_sf_lngamma(ld[l][k][t]);
      s -= gsl_sf_lngamma(ld[l][k][0] + ld[l][k][1]) - v;
      
      v = .0;
      for (uint32_t t = 0; t < _t; ++t)
	v += (ld[l][k][t] - 1) * elogbeta[l][k][t];
      s -= v;
    }
  }

  double p3 = s - (p1 + p2);

  info("approx. log likelihood = %f\n", s);
  fprintf(_lf, "%d\t%d\t%.5f\t%.5f\t%.5f\t%.5f\n", 
	  _iter, duration(), s, p1, p2, p3);
  fflush(_lf);
  return s;
}

void
SNPSamplingB::save_beta()
{
  const double **ebeta = _Ebeta.const_data();
  FILE *f = fopen(add_iter_suffix("/beta").c_str(), "w");
  if (!f)  {
    lerr("cannot open beta or lambda file:%s\n",  strerror(errno));
    exit(-1);
  }
  for (uint32_t l = 0; l < _l; ++l) {
    fprintf(f, "%d\t", l);
    for (uint32_t k = 0; k < _k; ++k) {
      fprintf(f, "%.8f\t", ebeta[l][k]);
    }
    fprintf(f, "\n");
  }
  fclose(f);
}
