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

