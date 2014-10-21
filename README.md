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

Documentation
-------------

See tex/other-docs

* FAKEPHASE-latentvar-model-nonconjugate-inference.pdf
* UNPHASED-marginal-model-nonconjugate-inference.pdf
* UNPHASED-latentvar-model-conjugate-inference.pdf

latentvar => we use population indicators Z

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
