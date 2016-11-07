#include "env.hh"
#include "marginf.hh"
#include "snpsamplinga.hh"
//#include "snpsamplingb.hh"
//#include "snpsamplingc.hh"
//#include "snpsamplingd.hh"
#include "snpsamplinge.hh"
//#include "snpsamplingf.hh"
//#include "snpsamplingg.hh"
#include "log.hh"
#include <stdlib.h>

#include <string>
#include <iostream>
#include <sstream>
#include <signal.h>

string Env::prefix = "";
Logger::Level Env::level = Logger::DEBUG;
FILE *Env::_plogf = NULL;
void usage();
void test();

Env *env_global = NULL;

volatile sig_atomic_t sig_handler_active = 0;

void
term_handler(int sig)
{
  if (env_global) {
    printf("Got termination signal. Saving model state and quitting.\n");
    fflush(stdout);
    env_global->terminate = 1;
  } else {
    signal(sig, SIG_DFL);
    raise(sig);
  }
}

int 
main(int argc, char **argv)
{
  signal(SIGTERM, term_handler);
  
  bool force_overwrite_dir = false;
  string datfname = "network.dat";
  string label = "";
  uint32_t n = 0, k = 0, l = 0;
  int i = 0;

  bool batch = false;
  bool online = true;
  bool logl = false;
  string eta_type = "default"; // "default", "sparse", "regular" or "dense"
  uint32_t rfreq  = 10000;
  bool rfreq_set = false;
  string idfile = "";
  bool loadcmp = false;
  bool marginf = false;
  bool snpsamplinga = false;
  bool snpsamplingb = false;
  bool snpsamplingc = false;
  bool snpsamplingd = false;
  bool snpsamplinge = false;
  bool snpsamplingf = false;
  bool snpsamplingg = false;
  double seed = 0;

  bool file_suffix = false;
  bool save_beta = false;
  bool adagrad = false;
  bool simulation1 = false;
  bool simulation2 = false;
  bool simulation3 = false;
  bool use_test_set = false;
  bool compute_beta = false;
  string locations_file = "";
  uint32_t nthreads = 6;
  double stop_threshold = 1e-5; //1e-5

  if (argc == 1) {
    usage();
    exit(-1);
  }
  
  while (i <= argc - 1) {
    if (strcmp(argv[i], "-help") == 0) {
      usage();
      exit(0);
    } else if (strcmp(argv[i], "-force") == 0) {
      fprintf(stdout, "+ overwrite option set\n");
      force_overwrite_dir = true;
    } else if (strcmp(argv[i], "-online") == 0) {
      fprintf(stdout, "+ online option set\n");
      online = true;
      batch = false;
    } else if (strcmp(argv[i], "-file") == 0) {
      if (i + 1 > argc - 1) {
	fprintf(stderr, "+ insufficient arguments!\n");
	exit(-1);
      }
      datfname = string(argv[++i]);
      fprintf(stdout, "+ using file %s\n", datfname.c_str());
    } else if (strcmp(argv[i], "-bed") == 0) {
      if (i + 1 > argc - 1) {
        fprintf(stderr, "+ insufficient arguments!\n");
        exit(-1);
      }
      datfname = string(argv[++i]);
      fprintf(stdout, "+ using file %s\n", datfname.c_str());
    } else if (strcmp(argv[i], "-batch") == 0) {
      batch = true;
      online = false;
      fprintf(stdout, "+ batch option set\n");
    } else if (strcmp(argv[i], "-n") == 0) {
      n = atoi(argv[++i]);
      fprintf(stdout, "+ n = %d\n", n);
    } else if (strcmp(argv[i], "-k") == 0) {
      k = atoi(argv[++i]);
      fprintf(stdout, "+ K = %d\n", k);
    } else if (strcmp(argv[i], "-l") == 0) {
      l = atoi(argv[++i]);
      fprintf(stdout, "+ L = %d\n", l);
    } else if (strcmp(argv[i], "-label") == 0) {
      label = string(argv[++i]);
    } else if (strcmp(argv[i], "-eta-type") == 0) {
      eta_type = string(argv[++i]);
      fprintf(stdout, "+ eta-type = %s\n", eta_type.c_str());
    } else if (strcmp(argv[i], "-rfreq") == 0) {
      rfreq = atoi(argv[++i]);
      fprintf(stdout, "+ rfreq = %d\n", rfreq);
      rfreq_set = true;
    } else if  (strcmp(argv[i], "-logl") == 0) {
      logl = true;
      fprintf(stdout, "+ logl option set\n");
    } else if (strcmp(argv[i], "-idfile") == 0) {
      idfile = string(argv[++i]);
      fprintf(stdout, "+ idfile = %s\n", idfile.c_str());
    } else if (strcmp(argv[i], "-loadcmp") == 0) {
      loadcmp = true;
      fprintf(stdout, "+ loadcmp option set\n");
    } /*else if (strcmp(argv[i], "-A") == 0) {
      marginf = true;
      fprintf(stdout, "+ algorithm A option set\n");
    } else if (strcmp(argv[i], "-snpsamplinga") == 0) {
      snpsamplinga = true;
      fprintf(stdout, "+ snp sampling A option set\n");
    } else if (strcmp(argv[i], "-B") == 0) {
      snpsamplingb = true;
      fprintf(stdout, "+ algorithm B option set\n");
    } else if (strcmp(argv[i], "-C") == 0) {
      snpsamplingc = true;
      fprintf(stdout, "+ algorithm C option set\n");
    } else if (strcmp(argv[i], "-D") == 0) {
      snpsamplingd = true;
      fprintf(stdout, "+ algorithm D option set\n");
    }*/ else if (strcmp(argv[i], "-E") == 0) {
      snpsamplinge = true;
      fprintf(stdout, "+ algorithm E option set\n");
    } else if (strcmp(argv[i], "-stochastic")==0) {
      snpsamplinge = true;
      fprintf(stdout, "+ stochastic option set\n");
    } /*else if (strcmp(argv[i], "-F") == 0) {
      snpsamplingf = true;
      simulation3 = true;
      fprintf(stdout, "+ algorithm F option set\n");
    } else if (strcmp(argv[i], "-G") == 0) {
      snpsamplingg = true;
      simulation3 = true;
      fprintf(stdout, "+ algorithm G option set\n");
    }*/ else if (strcmp(argv[i], "-seed") == 0) {
      seed = atof(argv[++i]);
      fprintf(stdout, "+ random seed set to %.5f\n", seed);
    } else if (strcmp(argv[i], "-file-suffix") == 0) {
      file_suffix = true;
    } else if (strcmp(argv[i], "-save-beta") == 0) {
      save_beta = true;
    } else if (strcmp(argv[i], "-sim1") ==0){
      simulation1 = true;
    } else if (strcmp(argv[i], "-sim2") ==0){
      simulation2 = true;
    } else if (strcmp(argv[i], "-adagrad") ==0){
      adagrad = true;
    } else if (strcmp(argv[i], "-nthreads") ==0){
      nthreads = atoi(argv[++i]);
    } else if (strcmp(argv[i], "-use-test-set") == 0){
      use_test_set = true;
    } else if (strcmp(argv[i], "-locations-file") == 0) {
      locations_file = string(argv[++i]);
    } else if (strcmp(argv[i], "-compute-beta") == 0) {
      compute_beta = true;
    } else if (strcmp(argv[i], "-stop-threshold") == 0) {
      stop_threshold = atof(argv[++i]);
    } else if (i > 0) {
      fprintf(stdout,  "error: unknown option %s\n", argv[i]);
      assert(0);
    } 
    ++i;
  };

  if (!rfreq_set)
    rfreq = 100000;

  assert (!(batch && online));
  
  Env env(n, k, l, batch, 
	  force_overwrite_dir, datfname, label, eta_type,
	  rfreq, logl, loadcmp, seed, file_suffix, 
	  save_beta, adagrad, nthreads, simulation1 || simulation2 || simulation3, 
	  use_test_set, compute_beta, locations_file, stop_threshold);
  env_global = &env;
  
  SNP snp(env);

  if (simulation1 && !simulation3) {
    if (snp.sim1() < 0) {
      fprintf(stderr, "error in simulation; quitting\n");
      return -1;
    }
    env.n = snp.n();
  } else if (simulation2) {
    if(env.k != 6){
      fprintf(stderr, "expecting k=6 for -sim2; quitting\n");
    }
    if (snp.sim2() < 0) {
      fprintf(stderr, "error in simulation; quitting\n");
    }
  } else if (simulation3) {
    assert(env.k == 6);
    if (snp.sim3() < 0) {
      fprintf(stderr, "error in simulation; quitting\n");
      exit(-1);
    }
    /*
    if (snp.sim1() < 0) {
      fprintf(stderr, "error in simulation; quitting\n");
      return -1;
    }
    env.n = snp.n();
    lerr("env.n = %d, snp.n = %d", env.n, snp.n());
    */
  } else {
    if (snp.read(datfname.c_str()) < 0) {
      fprintf(stderr, "error reading %s; quitting\n", 
	      datfname.c_str());
      return -1;
    }
    if (idfile != "" && snp.read_idfile(idfile.c_str()) < 0)
      fprintf(stderr, "error reading %s; quitting\n", 
	      idfile.c_str());
    env.n = snp.n();
  }

  if (!loadcmp) {  
    if (snpsamplinga) {
      SNPSamplingA snpsamplingA(env, snp);
      snpsamplingA.infer();
    } /*else if (snpsamplingb) {
      SNPSamplingB snpsamplingB(env, snp);
      snpsamplingB.infer();
    } else if (snpsamplingc) {
      SNPSamplingC snpsamplingC(env, snp);
      snpsamplingC.infer();
    } else if (snpsamplingd) {
      SNPSamplingD snpsamplingD(env, snp);
      snpsamplingD.infer();
    }*/ else if (snpsamplinge) {
      SNPSamplingE snpsamplingE(env, snp);
      snpsamplingE.infer();
    }/* else if (snpsamplingf) {
      SNPSamplingF snpsamplingF(env, snp);
      snpsamplingF.infer();
    } else if (snpsamplingg) {
      SNPSamplingG snpsamplingg(env, snp);
      snpsamplingg.infer();
    } else {
      MargInf marg(env, snp);
      marg.infer();
    }*/
  } else {
    MargInf popinf1(env, snp);
    MargInf popinf2(env, snp);
    popinf1.load_model("beta_ps.txt", "theta_ps.txt");
    popinf2.load_model("beta.txt", "theta.txt");
    
    Matrix skl(env.n,env.l);
    Matrix js_skl(env.n,env.l);
    Array a(env.n);
    Array b(env.l);
    uArray ac(env.n);
    uArray bc(env.l);

    double **skld = skl.data();
    double **js_skld = js_skl.data();
    double s = .0, js = .0;

    uint32_t t = 0;
    for (uint32_t n = 0; n < env.n; n++) 
      for (uint32_t l = 0; l < env.l; l++) {
	if (snp.is_missing(n, l)) {
	  skld[n][l] = .0;
	  continue;
	}

	Array p1(3), p2(3);
	popinf1.snp_likelihood(l, n, p1);
	popinf2.snp_likelihood(l, n, p2);
	
	skld[n][l] = snp.symmetrized_kl(p1, p2);
	js_skld[n][l] = snp.js_divergence(p1,p2);
	t++;
	s += skld[n][l];
	js += js_skld[n][l];

	a[n] += skld[n][l];
	b[l] += skld[n][l];

	ac[n]++;
	bc[l]++;

	if (n % 10 == 0) {
	  printf("\r%d", n);
	  fflush(stdout);
	}
      }
    printf("t = %d, s = %.4f, m = %.4f, thrown = %d\n", t, s, s/t, snp.thrown());
    printf("t = %d, js = %.4f, m = %.4f", t, js, js/t);
    fflush(stdout);

    FILE *f = fopen("skln.txt", "w");
    FILE *g = fopen("skll.txt", "w");
    
    for (uint32_t n = 0; n < env.n; ++n) 
      fprintf(f, "%d\t%d\t%f\n", n, ac[n], ac[n] > 0 ? a[n] / ac[n] : .0);

    for (uint32_t l = 0; l < env.l; ++l) 
      fprintf(g, "%d\t%d\t%f\n", l, bc[l], bc[l] > 0 ? b[l] / bc[l] : .0);

    fclose(g);
    fclose(f);
  }
}

void
usage()
{
  fprintf(stdout, "Population inference software for SNP data.\n"
	  "popgen [OPTIONS]\n"
	  "\t-help\t\tusage\n"
	  "\t-file <name>\t location by individuals ASCII matrix of SNP values (0,1,2)\n"
	  "\t-n <N>\t\t number of individuals\n"
	  "\t-l <L>\t\t number of locations\n"
	  "\t-k <K>\t\t number of populations\n"
	  "\t-batch\t\t run batch variational inference\n"
	  "\t-stochastic\t run stochastic variational inference\n"
	  "\t-label\t\t descriptive tag for the output directory\n"
	  "\t-force\t\t overwrite existing output directory\n"
	  "\t-rfreq <val>\t checks for convergence and logs output every <val> iterations\n"
	  "\t-idmap\t\t file containing individual name/meta-data, one per line\n"
	  );
  fflush(stdout);
}
