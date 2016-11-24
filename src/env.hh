#ifndef ENV_HH
#define ENV_HH

#include <unistd.h>
#include <stdint.h>
#include <errno.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <map>
#include <list>
#include <vector>
#include "matrix.hh"
#include "log.hh"

typedef uint8_t yval_t;

typedef D2Array<yval_t> AdjMatrix;
typedef D2Array<double> Matrix;
typedef D3Array<double> D3;
typedef D2Array<KV> MatrixKV;
typedef std::pair<uint32_t, uint32_t> LocIndiv;
typedef D1Array<yval_t> YArray;
typedef std::map<uint32_t, YArray *> YArrayMap;

typedef std::map<KV, bool> SNPMap;
typedef std::map<LocIndiv, bool> LocIndivMap;
typedef std::map<uint32_t, uint32_t> IDMap;
typedef std::map<uint32_t, uint32_t> FreqMap;
typedef std::map<string, uint32_t> FreqStrMap;
typedef std::map<string, uint32_t> StrMap;
typedef std::map<uint32_t, string> StrMapInv;
typedef D1Array<std::vector<uint32_t> *> SparseMatrix;
typedef std::map<uint32_t, bool> NodeMap;
typedef std::map<uint32_t, bool> BoolMap;
typedef std::map<uint64_t, bool> BoolMap64;
typedef std::map<uint32_t, uint32_t> NodeValMap;
typedef std::map<uint32_t, vector<uint32_t> > MapVec;
typedef MapVec SparseMatrix2;
typedef std::map<uint32_t, bool> SampleMap;
typedef map<uint32_t, vector<uint32_t> > SNPByLoc;
typedef vector<uint32_t> IndivList;

//typedef std::map<Edge, int> CountMap;
//typedef std::map<Edge, double> ValueMap;
typedef std::map<uint32_t, string> StrMapInv;

class Env {
public:
  Env(uint32_t N, uint32_t K, uint32_t L, 
      bool batch, 
      bool force_overwrite_dir, string dfname, 
      string label,
      string etype,
      uint32_t rfreq, bool logl, bool loadcmp,
      double seed, bool file_suffix,
      bool save_beta, bool adagrad, uint32_t nthreads,
      bool simulation, bool use_test_set,
      bool compute_beta, string locations_file,
      double stop_threshold);
  ~Env() { fclose(_plogf); }
  
  static string prefix;
  static Logger::Level level;

  uint32_t n;
  uint32_t k;
  uint32_t l;
  uint32_t t;
  uint32_t blocks;
  uint32_t indiv_sample_size;
  uint32_t nthreads;

  bool batch_mode;
  double meanchangethresh;
  double alpha;

  double validation_ratio;
  double heldout_indiv_ratio;
  double test_ratio;

  double eta0_dense;
  double eta1_dense;
  double eta0_regular;
  double eta1_regular;
  double eta0_uniform;
  double eta1_uniform;
  double eta0_sparse;
  double eta1_sparse;
  double eta0;
  double eta1;
  
  int reportfreq;
  double epsilon;
  double logepsilon;

  double tau0;
  double nodetau0;
  double nodekappa;
  double kappa;
  uint32_t online_iterations;
  bool terminate;
  string datfname;

  string label;
  string eta_type;

  bool use_validation_stop;
  bool use_training_stop;
  bool use_test_set;
  bool compute_logl;
  bool loadcmp;
  double seed;
  bool file_suffix;
  bool save_beta;
  bool adagrad;
  bool simulation;
  bool compute_beta;
  string locations_file;
  double stop_threshold;
   
  static string file_str(string fname);

private:
  static FILE *_plogf;
};


inline string
Env::file_str(string fname)
{
  string s = prefix + fname;
  return s;
}

inline
Env::Env(uint32_t N, uint32_t K, uint32_t L, 
	 bool batch, 
	 bool force_overwrite_dir, string dfname, 
	 string lbl,
	 string etype,
	 uint32_t rfreq, bool logl, bool lcmp, 
	 double seedv, bool file_suffixv, 
	 bool save_betav, bool adagradv, 
	 uint32_t nthreadsv, bool simulationv,
	 bool use_test_setv, bool compute_betav,
	 string locations_filev,
	 double stop_thresholdv)
  : n(N),
    k(K),
    l(L),
    t(2),
    blocks(100),
    indiv_sample_size(N/blocks),
    nthreads(nthreadsv),
    batch_mode(batch),
    meanchangethresh(0.001),
    alpha((double)1.0/k),
    heldout_indiv_ratio(0.001),
    validation_ratio(0.005),
    test_ratio(0.005),
    eta0_dense(4700.59),
    eta1_dense(0.77),
    eta0_regular(3.87),
    eta1_regular(1.84),
    eta0_uniform(1.00),
    eta1_uniform(1.00),
    eta0_sparse(0.97),
    eta1_sparse(6.33),
    eta0(eta0_uniform),
    eta1(eta1_uniform),
    reportfreq(rfreq),
    epsilon(1e-30),
    logepsilon(log(epsilon)),
    
    tau0(1), //default 1
    kappa(0.5),
    nodetau0(1), //default 1
    nodekappa(0.5),

    online_iterations(10), //default 10
    terminate(false),

    datfname(dfname),
    label(lbl),
    eta_type(etype),
    use_validation_stop(true),
    use_training_stop(false),
    compute_logl(logl),
    loadcmp(lcmp),
    seed(seedv),
    file_suffix(file_suffixv),
    save_beta(save_betav),
    adagrad(adagradv),
    simulation(simulationv),
    use_test_set(use_test_setv),
    compute_beta(compute_betav),
    locations_file(locations_filev),
    stop_threshold(stop_thresholdv)
{
  ostringstream sa;
  sa << "n" << n << "-";
  sa << "k" << k << "-";
  sa << "l" << l;
  if (label != "")
    sa << "-" << label;
  else if (datfname.length() > 3) {
    string q = datfname.substr(0,2);
    if (q == "..")
      q = "xx";
    sa << "-" << q;
  }
  
  if (seed != 0)
    sa << "-" << "seed" << seed;
  prefix = sa.str();
  level = Logger::TEST;

  fprintf(stdout, "+ Creating directory %s\n", prefix.c_str());
  assert (Logger::initialize(prefix, "infer.log", 
			     force_overwrite_dir, level) >= 0);
  fflush(stdout);
  
  if (N > 10000) {
    blocks = 100;
    indiv_sample_size = N / blocks;
  } else {
    blocks = 10;
    indiv_sample_size = N / blocks;
  }

  string ndatfname = file_str("/network.dat");
  unlink(ndatfname.c_str());
  assert (symlink(datfname.c_str(), ndatfname.c_str()) >= 0);
  fprintf(stderr, "+ done initializing env\n");
}

/* 
   src: http://www.delorie.com/gnu/docs/glibc/libc_428.html
   Subtract the `struct timeval' values X and Y,
   storing the result in RESULT.
   Return 1 if the difference is negative, otherwise 0.  
*/
inline int
timeval_subtract(struct timeval *result, struct timeval *x, struct timeval *y)
{
  /* Perform the carry for the later subtraction by updating y. */
  if (x->tv_usec < y->tv_usec) {
    int nsec = (y->tv_usec - x->tv_usec) / 1000000 + 1;
    y->tv_usec -= 1000000 * nsec;
    y->tv_sec += nsec;
  }
  if (x->tv_usec - y->tv_usec > 1000000) {
    int nsec = (x->tv_usec - y->tv_usec) / 1000000;
    y->tv_usec += 1000000 * nsec;
    y->tv_sec -= nsec;
  }

  /* Compute the time remaining to wait.
     tv_usec is certainly positive. */
  result->tv_sec = x->tv_sec - y->tv_sec;
  result->tv_usec = x->tv_usec - y->tv_usec;

  /* Return 1 if result is negative. */
  return x->tv_sec < y->tv_sec;
}

inline void
timeval_add (struct timeval *result, const struct timeval *x)
{
  result->tv_sec  += x->tv_sec;
  result->tv_usec += x->tv_usec;
  
  if (result->tv_usec >= 1000000) {
    result->tv_sec++;
    result->tv_usec -= 1000000;
  }
}

#endif
