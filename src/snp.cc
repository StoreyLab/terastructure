#include "snp.hh"
#include "log.hh"

static int scurr = 0;

int
SNP::read(string s)
{
  //check extension
  string ext;
  ext = s.substr(s.length()-4, 4);
  if(ext == ".bed"){
    printf("+ bed format detected\n");
    int ret = SNP::read_bed(s);
    return ret;
  } else if(ext == ".012") {
    printf("+ .012 detected");
  } else {
    lerr("unrecognized file extension");
    return -1;
  }

  uint32_t missing = 0;
  fprintf(stdout, "+ reading (%d,%d) snps from %s\n", 
	  _env.n, _env.l, s.c_str());
  fflush(stdout);

  uint64_t a0=0,a1=0,a2=0;

  _y = new AdjMatrix(_env.n, _env.l);

  yval_t **yd = _y->data();

  //FILE *maff = fopen(Env::file_str("/maf.tsv").c_str(), "w");
  FILE *f = fopen(s.c_str(), "r");
  if (!f) {
    lerr("cannot open file %s:%s", s.c_str(), strerror(errno));
    return -1;
  }
  char tmpbuf[2048*10];
  assert (tmpbuf > 0);
  string s1, s2;
  uint32_t id1, id2;
  
  uint32_t loc = 0;
  while (!feof(f)) {
    if (fscanf(f, "%s\n", tmpbuf) < 0) {
      printf("Error: unexpected lines in file\n");
      exit(-1);
    }
    double m = 0;
    uint32_t c = 0;
    for (uint32_t i = 0; i < _env.n; ++i) {
      if (tmpbuf[i] == '-') {
				missing++;
				yd[i][loc] = 3; //mark as missing
      } else {
				yd[i][loc] = tmpbuf[i] - '0';
				if (yd[i][loc] == 0)
				  a0++;
				else if (yd[i][loc] == 1)
				  a1++;
				else if (yd[i][loc] == 2)
				  a2++;
				m += yd[i][loc];
				c++;
      }
      debug("%c %d\n", tmpbuf[i], yd[i][loc]);
    }
    assert(c);
    /*m /= (2 * c);
    _maf[loc] = 0.5 - fabs(0.5 - m);
    fprintf(maff, "%d\t%.5f\t%.5f\n", loc, m, _maf[loc]);*/

    loc++;
    if (loc >= _env.l)
      break;
    if (loc % 10000 == 0) {
      printf("\r%d locations read", loc);
      fflush(stdout);
    }
  }
  fflush(stdout);
  fclose(f);
  //fclose(maff);

  return 0;
}

int
SNP::read_bed(string s)
{
  uint32_t missing = 0;
	uint32_t n = 0, l = 0;
  string prefix = s.substr(0, s.length()-4);

  
  //read in number of SNPs from .bim
  string bim = prefix + ".bim";
  FILE *bim_f = fopen(bim.c_str(), "r");
  if(!bim_f) {
		lerr("cannot open file %s:%s", bim.c_str(), strerror(errno));
		return -1;
  }
	char tmpbuf[2048*10];
	assert (tmpbuf > 0);
	while ( fgets(tmpbuf, 20480, bim_f) != NULL ) {
		l++;
	}
	fclose(bim_f);

  printf("+ bim file tells us %d SNPs\n", l);
  if(_env.l != l) {
    lerr("-l input doesn't match SNPs in bim file\n");
    return -1;
  }

	//read in number of individuals from .fam
  string fam = prefix + ".fam";
  FILE *fam_f = fopen(fam.c_str(), "r");
  if(!fam_f) {
    lerr("cannot open file %s:%s", fam.c_str(), strerror(errno));
    return -1;
  }
  while ( fgets(tmpbuf, 20480, fam_f) != NULL ) {
    n++;
  }
  fclose(fam_f);

  printf("+ fam file tells us %d individuals\n", n);
  if(_env.n != n) {
    lerr("-n input doesn't match individuals in fam file\n");
    return -1;
  }

  uint64_t a0=0,a1=0,a2=0;
  _y = new AdjMatrix(_env.n, _env.l);
  yval_t **yd = _y->data();

  //compute blocksize
  int numbytes = n/4;
  if(_env.n % 4 != 0) 
    numbytes++;

  //begin bed reading
  string bed = prefix + ".bed";
  FILE *bed_f = fopen(bed.c_str(), "r");
  if(!bed_f) {
    lerr("cannot open file %s:%s", bed.c_str(), strerror(errno));
    return -1;
  }
  
  char input;
  uint32_t counter = 0;

  //check first three bytes for what we're expecting
  input = fgetc(bed_f);
  if((int) input != 108) { // 108
    lerr("%s magic number incorrect\n", bed.c_str());
    return -1;
  }

  input = fgetc(bed_f);
  if((int) input != 27) { // 27
    lerr("%s magic number incorrect\n", bed.c_str());
    return -1;
  }

  input = fgetc(bed_f);
  if((int) input == 1) {
    ;
  } else if((int) input == 0) {
    lerr("individual major mode not supported yet!\n");
    return -1;
  } else {
    lerr("mode problem in %s\n", bed.c_str());
    return -1;
  }

  //now read in the SNPs!
  //FILE *maff = fopen(Env::file_str("/maf.tsv").c_str(), "w");
  char buffer[numbytes];
  uint8_t currbyte;
  uint32_t shiftcount = 0; //number of times i've shifted the bits
  uint32_t byteind = 0; //index on buffer
  uint32_t loc = 0;
  double m = 0; //maf
  uint32_t c = 0; //count 

  while(fread(buffer, 1, numbytes, bed_f) == numbytes) { //assuming bed is well formed...
    shiftcount = 0;
    byteind = 0;
    c = 0;
    m = 0.0;
    currbyte = (uint8_t) buffer[0];
    //loop over SNPs
    for(uint32_t i = 0; i < _env.n; i++) {
      if(currbyte % 4 == 1) { //missing val
        missing++;
        yd[i][loc] = 3;
      } else{
          if(currbyte % 4 == 3) { //0 or 2?
          yd[i][loc] = 2;
          a0++;
        } else if(currbyte % 4 == 2) { //1
          yd[i][loc] = 1;
          a1++;
        } else if(currbyte % 4 == 0) { //2 or 0?
          yd[i][loc] = 0;
          a2++;
        }
        m += yd[i][loc];
        c++;
      }
      
      currbyte >>= 2;
      shiftcount++;
      if(shiftcount == 4) {
        shiftcount = 0;
        byteind++;
        currbyte = (uint8_t) buffer[byteind];
      }
    }
    assert(c);
    /*m /= (2 * c);
    _maf[loc] = 0.5 - fabs(0.5 - m);
    fprintf(maff, "%d\t%.5f\t%.5f\n", loc, m, _maf[loc]);*/

    loc++;
    if (loc >= _env.l)
      break;
    if (loc % 20000 == 0){
      printf("\r%d locations read", loc);
      fflush(stdout);
    }
  }

  fflush(stdout);

  fclose(bed_f);
  //fclose(maff);

}

int
SNP::sim1()
{
  fprintf(stdout, "+ simulating (%d,%d) snps\n", _env.n, _env.l);
  fflush(stdout);

  _y = new AdjMatrix(_env.n, _env.l);
    
  // need to read in the Fst and allele freqs from hgdp
  vector<double> fst;
  vector<double> af;
  double fstin, afin;
  uint32_t max_sim = 430775;
  
  FILE *f = fopen("hgdp_BN.txt", "r");
  if (!f){
    printf("error reading hgdp_BN.txt \n");
    return -1;
  }
  for(uint32_t i = 0; i < max_sim; i++) { //hard code number of rows...
    if(fscanf(f, "%lf %lf\n", &fstin, &afin) < 0) {
      printf("Error: BN simulation input\n");
      exit(-1);
    }
    fst.push_back(fstin);
    af.push_back(afin);
  }
  fclose(f);
  
  //simulation parameters currently hardcoded
  uint32_t blocksize = _env.n/50; //250; //block size of outer dirichlet
  double dir_alpha = 0.2;
  double dir_gamma = 50.0;
  
  Matrix S(_env.n, _env.k); 
  double **Sd = S.data();
  
  //init outer dirichlet parameters 
  //is it bad to just leave the pointer chilling in this function? should I have a init_alpha() function? deconstructor will handle it at the end of sim()?
  vector<double> Alpha(_env.k);
  for(uint32_t i = 0; i < _env.k; i++) Alpha[i] = dir_alpha;
  
  //init inner dirichlet parameters
  vector<double> Gamma(_env.k);
  vector<double> Param(_env.k);
  vector<double> Tmp(_env.k); //i'm going to reuse this guy..
    
  //draw S matrix
  for(uint32_t j = 0; j < _env.n; j++) {
    if(j % blocksize == 0) {
      gsl_ran_dirichlet(_r, _env.k, Alpha.data(), Gamma.data());
      for(uint32_t k = 0; k < _env.k; k++) {
        Param[k] = Gamma[k] * dir_gamma;
      }
    }
    gsl_ran_dirichlet(_r, _env.k, Param.data(), Tmp.data());
    
    //for(uint32_t k = 0; k < _env.k; k++) printf("%f ", Tmp[k]); printf("\n");
        
    //renormalize s.t. min is 1e-6
    double offset = 1e-6 / (1 - (_env.k*1e-6));
    double newsum = 1 + (_env.k*offset);
    for(uint32_t k = 0; k < _env.k; k++) {
      Sd[j][k] = (Tmp[k] + offset) / newsum;
    }
  }
  
  //temp for debugging purposes: output S matrix
  f = fopen(_env.file_str("/S_out.txt").c_str(), "w");
  for(uint32_t j = 0; j < _env.n; j++) {
    for(uint32_t k = 0; k < _env.k ; k++) {
      fprintf(f, "%.6f ", Sd[j][k]);
    }
    fprintf(f, "\n");
  }
  fclose(f);
  
  vector<double> Allele_freqs(_env.k);
  uint32_t index;
  double bp0, bp1, marg_af;
  
  //simulate y!
  yval_t **yd = _y->data();

  for(uint32_t i = 0; i < _env.l; i++) {
    //simulate the ancestral allele frequencies
    index = gsl_rng_uniform_int(_r, max_sim);
    if(fst[index] < 1e-6) {
      for(uint32_t k = 0; k < _env.k; k++) {
        Allele_freqs[k] = af[index];
      }
    }
    else {
      bp0 = af[index] * (1-fst[index])/fst[index];
      bp1 = (1-af[index]) * (1-fst[index])/fst[index];
      for(uint32_t k = 0; k < _env.k; k++) {
        Allele_freqs[k] = gsl_ran_beta(_r, bp0, bp1);
      }
    }
    //compute marginal af and draw the binomial
    for(uint32_t j = 0; j < _env.n; j++) {
      marg_af = 0.0;
      for(uint32_t k = 0; k < _env.k; k++) {
        marg_af += Allele_freqs[k] * Sd[j][k];
      }
      yd[j][i] = (yval_t) gsl_ran_binomial(_r, marg_af, 2);
      
      if(i % 10000 == 0) {
        printf("\r%d locations simulated", i);
        fflush(stdout);
      }
    }
  }
  
  return 0;
}

//sim based off reading the fitted betas
int
SNP::sim2(){
  fprintf(stdout, "+ simulating (%d,%d) snps\n", _env.n, _env.l);
  fflush(stdout);

  _y = new AdjMatrix(_env.n, _env.l);

  //read in fitted betas
  uint32_t max_sim = 1854622;
  Matrix G(max_sim, 6); //fixed size
  double **Gd = G.data();

  FILE *f = fopen("TGP_1718_k6_beta.txt", "r");

  uint32_t tmp;
  double asdf;
  for(uint32_t i = 0; i < max_sim; i++) {
    //just assuming we're not gonna change that file.
    fscanf(f, "%d\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t\n", &tmp, &Gd[i][0], &Gd[i][1], &Gd[i][2], &Gd[i][3], &Gd[i][4], &Gd[i][5]);
  }

  fclose(f);

  //simulation parameters currently hardcoded
  uint32_t blocksize = _env.n/50; //250; //block size of outer dirichlet
  double dir_alpha = 0.2;
  double dir_gamma = 50.0;
  
  Matrix S(_env.n, _env.k); 
  double **Sd = S.data();

  //init outer dirichlet parameters 
  //is it bad to just leave the pointer chilling in this function? should I have a init_alpha() function? deconstructor will handle it at the end of sim()?
  vector<double> Alpha(_env.k);
  for(uint32_t i = 0; i < _env.k; i++) Alpha[i] = dir_alpha;
  
  //init inner dirichlet parameters
  vector<double> Gamma(_env.k);
  vector<double> Param(_env.k);
  vector<double> Tmp(_env.k); //i'm going to reuse this guy..

  //draw S matrix
  for(uint32_t j = 0; j < _env.n; j++) {
    if(j % blocksize == 0) {
      gsl_ran_dirichlet(_r, _env.k, Alpha.data(), Gamma.data());
      for(uint32_t k = 0; k < _env.k; k++) {
        Param[k] = Gamma[k] * dir_gamma;
      }
    }
    gsl_ran_dirichlet(_r, _env.k, Param.data(), Tmp.data());
    
    //for(uint32_t k = 0; k < _env.k; k++) printf("%f ", Tmp[k]); printf("\n");
        
    //renormalize s.t. min is 1e-6
    double offset = 1e-6 / (1 - (_env.k*1e-6));
    double newsum = 1 + (_env.k*offset);
    for(uint32_t k = 0; k < _env.k; k++) {
      Sd[j][k] = (Tmp[k] + offset) / newsum;
    }
  }
  
  //temp for debugging purposes: output S matrix
  f = fopen(_env.file_str("/S_out.txt").c_str(), "w");
  for(uint32_t j = 0; j < _env.n; j++) {
    for(uint32_t k = 0; k < _env.k ; k++) {
      fprintf(f, "%.6f ", Sd[j][k]);
    }
    fprintf(f, "\n");
  }
  fclose(f);

  //output betas
  f = fopen(_env.file_str("/G_out.txt").c_str(), "w");
  FILE *x = fopen(_env.file_str("/X_out.txt").c_str(), "w");

  //simulate
  double marg_af;
  uint32_t index;
  yval_t **yd = _y->data();

  for(uint32_t i = 0; i < _env.l; i++) {
    //pick the proper TGP fitted betas
    index = gsl_rng_uniform_int(_r, max_sim);

    //compute marginal af and draw the binomial
    for(uint32_t j = 0; j < _env.n; j++) {
      marg_af = 0.0;
      for(uint32_t k = 0; k < _env.k; k++) {
        marg_af += Gd[index][k] * Sd[j][k];
      }
      yd[j][i] = (yval_t) gsl_ran_binomial(_r, marg_af, 2);
      
    }

    if(i % 10000 == 0) {
      printf("\r%d locations simulated", i);
      fflush(stdout);
    }

    //output if we've outputted less than 10K
    if(i < 10000){
      for(uint32_t k = 0; k < _env.k; k++) {
        fprintf(f, "%8f ", Gd[index][k]);
      }
      fprintf(f, "\n");

      for(uint32_t j = 0; j < _env.n; j++) {
        fprintf(x, "%d", yd[j][i]);
      }
      fprintf(x, "\n");
    }
  }

  fclose(f);
  fclose(x);

  return 0;
}


void
BigSim::sim_set_y(uint32_t loc, uint32_t block, YArray &y)
{
  double **Gd = _G.data();
  double **Sd = _S.data();
  IDMap::const_iterator itr = _loc_to_idx.find(loc);
  if (itr == _loc_to_idx.end()) {
    uint32_t x = gsl_rng_uniform_int(_snp._r, _env.l);
    _loc_to_idx[loc] = x;
    itr = _loc_to_idx.find(loc);
  }

  uint32_t idx = itr->second;
  uint32_t sz = _env.n / _env.blocks;
  uint32_t pos = block * sz;
  debug("simulating individuals %d to %d for location %d", pos, pos + sz, loc);
  for (uint32_t i = pos; i < pos + sz; i++) {
    double marg_af = 0.0;
    for(uint32_t k = 0; k < _env.k; k++)
      marg_af += Gd[idx][k] * Sd[i][k];
    y[i] = (yval_t) gsl_ran_binomial(_snp._r, marg_af, 2);
  }
  debug("done simulating");
}

void
BigSim::sim_set_y(uint32_t loc, YArray &y)
{
  double **Gd = _G.data();
  double **Sd = _S.data();
  IDMap::const_iterator itr = _loc_to_idx.find(loc);
  if (itr == _loc_to_idx.end()) {
    uint32_t x = gsl_rng_uniform_int(_snp._r, _env.l);
    _loc_to_idx[loc] = x;
    itr = _loc_to_idx.find(loc);
  }
  uint32_t idx = itr->second;

  gsl_rng_set(_snp._r, loc+32767); // XXX
  
  debug("simulating individuals %d to %d for location %d", pos, pos + sz, loc);
  for (uint32_t i = 0; i < _env.n; i++) {
    double marg_af = 0.0;
    for(uint32_t k = 0; k < _env.k; k++)
      marg_af += Gd[idx][k] * Sd[i][k];
    debug("loc:%d, n:%d, marg_af:%.5f", loc, i, marg_af);
    y[i] = (yval_t) gsl_ran_binomial(_snp._r, marg_af, 2);
  }
  debug("done simulating");
}

//sim based off reading the fitted betas
int
BigSim::sim()
{
  fprintf(stdout, "+ BigSim: simulating (%d,%d) snps\n", _env.n, _env.l);
  fflush(stdout);

  //read in fitted betas
  double **Gd = _G.data();

  FILE *f = fopen("TGP_1718_k6_beta.txt", "r");
  if (!f) {
    fprintf(stderr, "cannot find beta file\n");
    fflush(stdout);
    exit(-1);
  }
  uint32_t tmp;
  double asdf;
  for(uint32_t i = 0; i < L; i++) {
    //for (uint32_t k = 0; k < 6; k++)
    //Gd[i][k] = gsl_rng_uniform(_snp._r);
    //just assuming we're not gonna change that file.
    fscanf(f, "%d\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t\n", 
	   &tmp, &Gd[i][0], &Gd[i][1], &Gd[i][2], &Gd[i][3], &Gd[i][4], &Gd[i][5]);
  }
  fclose(f);
  lerr("done reading file");

  //simulation parameters currently hardcoded
  uint32_t blocksize = _env.n/50; //250; //block size of outer dirichlet
  double dir_alpha = 0.2;
  double dir_gamma = 50.0;
  
  double **Sd = _S.data();

  //init outer dirichlet parameters 
  //is it bad to just leave the pointer chilling in this function? should I have a init_alpha() function? deconstructor will handle it at the end of sim()?
  vector<double> Alpha(_env.k);
  for(uint32_t i = 0; i < _env.k; i++) Alpha[i] = dir_alpha;
  
  //init inner dirichlet parameters
  vector<double> Gamma(_env.k);
  vector<double> Param(_env.k);
  vector<double> Tmp(_env.k); //i'm going to reuse this guy..

  //draw S matrix
  for(uint32_t j = 0; j < _env.n; j++) {
    if(j % blocksize == 0) {
      gsl_ran_dirichlet(_snp._r, _env.k, Alpha.data(), Gamma.data());
      for(uint32_t k = 0; k < _env.k; k++) {
        Param[k] = Gamma[k] * dir_gamma;
      }
    }
    gsl_ran_dirichlet(_snp._r, _env.k, Param.data(), Tmp.data());
    
    //for(uint32_t k = 0; k < _env.k; k++) printf("%f ", Tmp[k]); printf("\n");
        
    //renormalize s.t. min is 1e-6
    double offset = 1e-6 / (1 - (_env.k*1e-6));
    double newsum = 1 + (_env.k*offset);
    for(uint32_t k = 0; k < _env.k; k++) {
      Sd[j][k] = (Tmp[k] + offset) / newsum;
    }
  }
  
  //temp for debugging purposes: output S matrix
  f = fopen(_env.file_str("/S_out.txt").c_str(), "w");
  for(uint32_t j = 0; j < _env.n; j++) {
    for(uint32_t k = 0; k < _env.k ; k++) {
      fprintf(f, "%.6f ", Sd[j][k]);
    }
    fprintf(f, "\n");
  }
  fclose(f);

  //output betas
  f = fopen(_env.file_str("/G_out.txt").c_str(), "w");
  FILE *x = fopen(_env.file_str("/X_out.txt").c_str(), "w");
}

int
BigSim::sim1()
{
  fprintf(stdout, "+ BigSim sim1: simulating (%d,%d) snps\n", _env.n, _env.l);
  fflush(stdout);

  //read in fitted betas
  double **Gd = _G.data();

  //need to read in the Fst and allele freqs from hgdp
  vector<double> fst;
  vector<double> af;
  double fstin, afin;
  
  FILE *f = fopen("hgdp_BN.txt", "r");
  if(!f){
    printf("error reading hgdp_BN.txt \n");
    return -1;
  }
  for(uint32_t i = 0; i < BigSim::HGDP_size ; i++) { //hard code number of rows...
    if(fscanf(f, "%lf %lf\n", &fstin, &afin) < 0) {
      printf("Error: BN simulation input\n");
      exit(-1);
    }
    fst.push_back(fstin);
    af.push_back(afin);
  }
  fclose(f);
  lerr("done reading file");

  //populate G
  uint32_t index;
  double bp0, bp1;

  for(uint32_t i = 0; i < _env.l; i++) {
    index = gsl_rng_uniform_int(_snp._r, BigSim::HGDP_size);
    if(fst[index] < 1e-6) {
      for(uint32_t k = 0; k < _env.k; k++) {
        Gd[i][k] = af[index];
      }
    }
    else {
      bp0 = af[index] * (1-fst[index])/fst[index];
      bp1 = (1-af[index]) * (1-fst[index])/fst[index];
      for(uint32_t k = 0; k < _env.k; k++) {
        Gd[i][k] = gsl_ran_beta(_snp._r, bp0, bp1);
      }
    }
  }

  //simulation parameters currently hardcoded
  uint32_t blocksize = _env.n/50; //250; //block size of outer dirichlet
  double dir_alpha = 0.2;
  double dir_gamma = 50.0;
  
  double **Sd = _S.data();

  //init outer dirichlet parameters 
  vector<double> Alpha(_env.k);
  for(uint32_t i = 0; i < _env.k; i++) Alpha[i] = dir_alpha;
  
  //init inner dirichlet parameters
  vector<double> Gamma(_env.k);
  vector<double> Param(_env.k);
  vector<double> Tmp(_env.k); //i'm going to reuse this guy..

  //draw S matrix
  for(uint32_t j = 0; j < _env.n; j++) {
    if(j % blocksize == 0) {
      gsl_ran_dirichlet(_snp._r, _env.k, Alpha.data(), Gamma.data());
      for(uint32_t k = 0; k < _env.k; k++) {
        Param[k] = Gamma[k] * dir_gamma;
      }
    }
    gsl_ran_dirichlet(_snp._r, _env.k, Param.data(), Tmp.data());
    
    //for(uint32_t k = 0; k < _env.k; k++) printf("%f ", Tmp[k]); printf("\n");
        
    //renormalize s.t. min is 1e-6
    double offset = 1e-6 / (1 - (_env.k*1e-6));
    double newsum = 1 + (_env.k*offset);
    for(uint32_t k = 0; k < _env.k; k++) {
      Sd[j][k] = (Tmp[k] + offset) / newsum;
    }
  }
  
  //temp for debugging purposes: output S matrix
  f = fopen(_env.file_str("/S_out.txt").c_str(), "w");
  for(uint32_t j = 0; j < _env.n; j++) {
    for(uint32_t k = 0; k < _env.k ; k++) {
      fprintf(f, "%.6f ", Sd[j][k]);
    }
    fprintf(f, "\n");
  }
  fclose(f);

  //output betas
  f = fopen(_env.file_str("/G_out.txt").c_str(), "w");
  for(uint32_t i = 0; i < _env.l; i++) {
    for(uint32_t k = 0; k < _env.k; k++) {
      fprintf(f, "%.6f ", Gd[i][k]);
    }
    fprintf(f, "\n");
  }
  fclose(f);
}


int
SNP::read_idfile(string s)
{
  FILE *f = fopen(s.c_str(), "r");
  if (!f) {
    lerr("cannot open file %s:%s", s.c_str(), strerror(errno));
    return -1;
  }
  uint32_t id = 0;
  string idstr;
  char tmpbuf[128];
  while (!feof(f)) {
    if (fscanf(f, "%s\n", tmpbuf) < 0) {
      fprintf(stderr, "error: unexpected line in file\n");
      exit(-1);
    }
    _labels[id] = string(tmpbuf);
    id++;
  }
  fclose(f);
  return 0;
}

