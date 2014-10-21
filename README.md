*This package implements a scalable, multi-threaded implementation of the TeraStructure algorithm for fitting a Bayesian model of genetic variationin human populations on tera-sample-sized data sets ($10^{12}$ observed genotypes, e.g., 1M individuals at 1M SNPs).*

Installation
------------

Required libraries: gsl, gslblas, pthread

On Linux/Unix run

 cd pkg
 ./configure
 make; make install

On Mac OS, the location of the required gsl, gslblas and pthread
libraries may need to be specified:

 ./configure LDFLAGS="-L/opt/local/lib" CPPFLAGS="-I/opt/local/include"
 make; make install

The binary 'gaprec' will be installed in /usr/local/bin unless a
different prefix is provided to configure. (See pkg/INSTALL.)

Reference
---------

**Fitting probabilistic models of genetic variation on millions of humans**

P. Gopalan, W. Hao, D.M. Blei, J.D. Storey
*In submission.*

Abstract
--------

The goal of population genetics is to quantitatively understand variation of genetic polymorphisms among individuals. Researchers have developed sophisticated statistical methods to capture the complex population structure that underlies observed genotypes in humans. The number of humans that have been densely genotyped across the genome has grown significantly in recent years. In aggregate about 1M individuals have been densely genotyped to date, and if we could analyze this data then we would have a nearly complete picture of human genetic variation. Existing state-of-the-art methods, however, cannot scale to data of this size. To this end, we have developed TeraStructure.

TeraStructure is a new algorithm to fit Bayesian models of genetic variation in human populations on tera-sample-sized data sets (10^12 observed genotypes, e.g., 1M individuals at 1M SNPs). It is a principled approach to approximate Bayesian inference that iterates between subsampling locations of the genome and updating an estimate of the latent population structure. On real and simulated data sets of up to 10K individuals, TeraStructure is twice as fast as existing methods and recovers the latent population structure with equal accuracy. On genomic data simulated at the tera-sample-size scales, TeraStructure continues to be accurate and is the only method that can complete its analysis.


POPGEN: Population genetics inference software
----------------------------------------------

**popgen** [OPTIONS]

        -help            usage
        -file <name>     location by individuals matrix of SNP values (0,1,2)
        -n <N>           number of individuals
        -l <L>           number of locations
        -k <K>           number of populations
        -batch           run batch variational inference
        -stochastic      run stochastic variational inference
        -label           descriptive tag for the output directory

	-marginf         use the marginal model and nonconjugate inference
	-broken          use the original (broken) model
	(by default it runs inference for the fake phase model)

	OPTIONAL
	-file-suffix	 save files with the corresponding iteration as suffix
        -force           overwrite existing output directory
        -rfreq <val>     checks for convergence and logs output every <val> iterations
        -idfile <name>	 file containing individual name/meta-data, one per line
	-seed <val>	 value is a real number (read as "double")
	      		 sets the seed for the GSL library

Example
-------

Please see pkg/scripts/pop.pl for an example of how nonconjugate inference is run under the marginal model.

./src/popgen -file hgdp_940.012 -n 940 -k 6 -l 619883 -label snp  -rfreq 1000 -idfile 1kG_1500.indv -marginf

* Output written to directory n940-k6-l619883-snp/
* theta.txt, beta.txt: approximate posterior mixed-memberships and 

Note that theta.txt is saved periodically even before convergence and can be used to monitor results. Convergence is fast even on large datasets requiring only several hours. However, beta.txt is saved at the end, and it can take a long time (several days).


Data sets
---------

Real and simulated data sets are on Dropbox at:

https://www.dropbox.com/home/research/genomic-data

If you cannot access it, please send a note to pgopalan@cs.princeton.edu.

Model fits
----------

Model fits for the HGDP data, 1000 Genomes data and Balding and
Nichols simulations using the UNPHASED conjugate model are on Dropbox
at:

https://www.dropbox.com/home/research/genomic-data

* n1500-l1.5M-k1to10-fits.tgz
* n940-l6K-k5to15-fits.tgz
* simulations/admix_except_3_and_001.tgz
