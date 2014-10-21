#include "snpsamplingc.hh"
#include "log.hh"
#include <sys/time.h>

SNPSamplingC::SNPSamplingC(Env &env, SNP &snp)
  :_env(env), _snp(snp),
   _n(env.n), _k(env.k), _l(_env.l),
   _t(env.t), _nthreads(_env.nthreads),
   _iter(0), _alpha(_k), _loc(0),
   _eta(_k,_t),
   _gamma(_n,_k), 
   _lambda(_l,_k,_t),
   _lambdat(_k,_t),
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

  if (_nthreads > 0) {
    Thread::static_initialize();
    PhiRunner::static_initialize();
    start_threads();
  }
}

SNPSamplingC::~SNPSamplingC()
{
  fclose(_vf);
  fclose(_tf);
  fclose(_lf);
  fclose(_tef);
  fclose(_vef);
}

void
SNPSamplingC::init_heldout_sets()
{
  //set_test_sample();
  set_validation_sample();

  Env::plog("test ratio", _env.test_ratio);
  Env::plog("validation ratio", _env.validation_ratio);
}

void
SNPSamplingC::set_test_sample()
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
SNPSamplingC::set_validation_sample2()
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
SNPSamplingC::set_validation_sample()
{
  uint32_t per_loc_h = 0;
  if (_n <= 1000 || _env.simulation)
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
SNPSamplingC::shuffle_nodes()
{
  for (uint32_t i = 0; i < _n; ++i)
    _shuffled_nodes[i] = i;
  gsl_ran_shuffle(_r, (void *)_shuffled_nodes.data(), _n, sizeof(uint32_t));
}

void
SNPSamplingC::init_gamma()
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
SNPSamplingC::init_lambda()
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

int
SNPSamplingC::start_threads()
{
  for (uint32_t i = 0; i < _nthreads; ++i) {
    PhiRunner *t = new PhiRunner(_env, &_r, 
				 _iter, _n, _k, 
				 0, _t, _snp, *this,
				 _out_q, _in_q, _cm);
    if (t->create() < 0)
      return -1;
    _thread_map[t->id()] = t;
  }
  return 0;
}

void
SNPSamplingC::get_subsample()
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
SNPSamplingC::update_rho_indiv(uint32_t n)
{
  _rho_indiv[n] = pow(_nodetau0 + _c_indiv[n], -1 * _nodekappa);
  _c_indiv[n]++;
}

void
SNPSamplingC::update_rho_loc(uint32_t l)
{
  _rho_loc[l] = pow(_tau0 + _c_loc[l], -1 * _kappa);
  _c_loc[l]++;
}

void
SNPSamplingC::update_lambda()
{
  const yval_t ** const snpd = _snp.y().const_data();
  double lambda_scale = (double)_n / _indivs.size();
  double **ld = _lambda.data()[_loc];
  double **ldt = _lambdat.data();
    
  update_rho_loc(_loc);
  for (uint32_t k = 0; k < _k; ++k) {
    double lk0 = _env.eta0 + lambda_scale * ldt[k][0] - ld[k][0];
    double lk1 = _env.eta1 + lambda_scale * ldt[k][1] - ld[k][1];
    ld[k][0] += _rho_loc[_loc] * lk0;
    ld[k][1] += _rho_loc[_loc] * lk1;
    debug("lambda: k = %d: (%f,%f)", k, ld[k][0], ld[k][1]);
  }
}

void
SNPSamplingC::estimate_beta()
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
SNPSamplingC::infer()
{
  printf("Running SNPSamplingC::infer(), nthreads = %d\n", _nthreads);
  uint64_t threads_used = 0;
  while (1) {
    _loc = gsl_rng_uniform_int(_r, _l);
    const yval_t ** const snpd = _snp.y().const_data();
    get_subsample();

    _cm.lock();
    _cm.broadcast();
    _cm.unlock();

    // split indivs into _nthread chunks
    uint32_t chunk_size = (int)(((double)_indivs.size()) / _nthreads);
    debug("chunk size = %d\n", chunk_size);
    uint32_t t = 0, c = 0;
    for (uint32_t i = 0; i < _indivs.size(); ++i) {
      uint32_t n = _indivs[i];
      ChunkMap::iterator it = _chunk_map.find(t);
      if (it == _chunk_map.end()) {
	IndivsList *il = new IndivsList;
	_chunk_map[t] = il;
      }
      IndivsList *il = _chunk_map[t];
      il->push_back(n);
      c++;
      if (c >= chunk_size && t < (uint32_t)_nthreads - 1) {
	c = 0;
	t++;
      }
    }

    uint32_t x = 0;
    for (ChunkMap::iterator it = _chunk_map.begin(); 
	 it != _chunk_map.end(); ++it) {
      IndivsList *il = it->second;
      debug("pushing chunk %d", x);
      fflush(stdout);
      _out_q.push(il);
      x++;
    }

    _lambdat.zero();
    uint32_t nt = 0;
    do {
      debug("main: threads %d done, size = %d\n", 
	     nt, _in_q.size());
      pthread_t *p = _in_q.pop();
      assert(p);
      PhiRunner *t = _thread_map[*p];
      const Matrix &lambdat = t->lambdat();
      const double **ldt_t = lambdat.const_data();
      _cthreads[t->id()]=true;
      double **ldt = _lambdat.data();
      for (uint32_t k = 0; k < _k; ++k)
	for (uint32_t r = 0; r < _t; ++r)
	  ldt[k][r] += ldt_t[k][r];
      fflush(stdout);
      delete p;
      nt++;
    } while (nt != _nthreads || !_in_q.empty());
    
    assert (nt == _nthreads);

    debug("lambdat = %s", _lambdat.s().c_str());
    
    threads_used += _cthreads.size();

    update_lambda();
    _chunk_map.clear();
    _cthreads.clear();
    estimate_beta();

    debug("Etheta = %s", _Etheta.s().c_str());
    debug("Elogtheta = %s", _Elogtheta.s().c_str());
    debug("Ebeta = %s", _Ebeta.s().c_str());
    debug("Elogbeta = %s", _Elogbeta.s().c_str());

    _iter++;

    if (_iter % _env.reportfreq == 0) {
      printf("iteration = %d took %d secs\n", 
	     _iter, duration());
      lerr("iteration = %d took %d secs, mean threads used %.2f\n", 
	   _iter, duration(), (double)threads_used / _iter);
      fflush(stdout);
      lerr("computing heldout likelihood @ %d secs", duration());
      compute_likelihood(false, true);
      //compute_likelihood(false, false);
      lerr("saving theta @ %d secs", duration());
      save_model();
      lerr("done @ %d secs", duration());
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
SNPSamplingC::compute_likelihood(bool first, bool validation)
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
	_prev_h != 0 && fabs((a - _prev_h) / _prev_h) < 0.00000001) {
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
SNPSamplingC::save_gamma()
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
SNPSamplingC::add_iter_suffix(const char *c)
{
  ostringstream sa;
  if (_env.file_suffix)
    sa << c << "_" << _iter << ".txt";
  else
    sa << c << ".txt";
  return Env::file_str(sa.str());
}

void
SNPSamplingC::save_model()
{
  save_gamma();
}

void
SNPSamplingC::load_model(string betafile, string thetafile)
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
SNPSamplingC::estimate_all_theta()
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


int
PhiRunner::do_work()
{
  bool first = true;
  do {
    IndivsList *ilist = _out_q.pop();
    if (first || _prev_iter != _iter) {
      first = false;
      _prev_iter = _iter;
      reset(_pop.sampled_loc());
      debug("thread = %ld, location = %ld", id(), _loc);
    }
    process(*ilist);
    pthread_t *p = new pthread_t(pthread_self());
    _in_q.push(p);
    //printf("thread %ld done, in size = %d\n", id(), _in_q.size());
    
    _cm.lock();
    while (_iter == _prev_iter)
      _cm.wait();
    _cm.unlock();
    debug("thread all done");
    fflush(stdout);
    
  } while (1);
}

int
PhiRunner::process(const IndivsList &v)
{
  double u = 1./_k;
  for (uint32_t i = 0; i < v.size(); ++i) {
    uint32_t n = v[i];
    _phimom.set_elements(n, u);
    _phidad.set_elements(n, u);
    update_phimom(n);
    update_phidad(n);
  }
  debug("iter = %d, thread = %ld, phimom = %s", _iter, id(), _phimom.s().c_str());
  debug("iter = %d, thread = %ld, phidad = %s", _iter, id(), _phidad.s().c_str());
  update_gamma(v);
  update_lambda_t(v);
  estimate_theta(v);
}

void
PhiRunner::update_gamma(const IndivsList &indivs)
{
  const double **phidadd = _phidad.const_data();
  const double **phimomd = _phimom.const_data();
  const yval_t ** const snpd = _snp.y().const_data();

  double gamma_scale = _env.l;
  double **gd = _pop.gamma().data();

  // no locking needed
  // each thread owns it's own set of indivs
  for (uint32_t i = 0; i < indivs.size(); ++i) {
    uint32_t n = indivs[i];
    if (!_pop.kv_ok(n, _loc))
      continue;
    
    _pop.update_rho_indiv(n);
    yval_t y = snpd[n][_loc];
    for (uint32_t k = 0; k < _k; ++k) {
      double gk = _pop.alpha(k) + (gamma_scale *			\
				   (y * phimomd[n][k] + (2 - y) * phidadd[n][k])) - gd[n][k];
      gd[n][k] += _pop.rho_indiv(n) * gk;
      if (n == 30) {
	debug("gamma: thread: %d, n:%d, k:%d -> %f\n", id(), n, k, gd[n][k]);
      }
    }
  }
}

void
PhiRunner::estimate_theta(const IndivsList &indivs)
{
  const double ** const gd = _pop.gamma().const_data();
  double **theta = _pop.Etheta().data();
  double **elogtheta = _pop.Elogtheta().data();
  
  for (uint32_t i = 0; i < indivs.size(); ++i)  {
    uint32_t n = indivs[i];
    double s = .0;
    if (!_pop.kv_ok(n, _loc))
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
PhiRunner::update_lambda_t(const IndivsList &indivs)
{
  const double **phidadd = _phidad.const_data();
  const double **phimomd = _phimom.const_data();
  const yval_t ** const snpd = _snp.y().const_data();

  double **ldt = _lambdat.data();
  for (uint32_t k = 0; k < _k; ++k) {
    for (uint32_t i = 0; i < indivs.size(); ++i)  {
      uint32_t n = indivs[i];
      if (!_pop.kv_ok(n, _loc))
	continue;
      ldt[k][0] += phimomd[n][k] * snpd[n][_loc];
      ldt[k][1] += phidadd[n][k] * (2 - snpd[n][_loc]);
    }
  }
}

void
SNPSamplingC::save_beta()
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
