*This package implements a scalable, multi-threaded implementation of the TeraStructure algorithm for fitting a Bayesian model of genetic variation in human populations on tera-sample-sized data sets (10^12 observed genotypes, e.g., 1M individuals at 1M SNPs).*

Manual
------

See our wiki for instructions and other details: https://github.com/StoreyLab/terastructure/wiki

Citation
--------

**Fitting probabilistic models of genetic variation on millions of humans**  
**P. Gopalan, W. Hao, D.M. Blei, J.D. Storey**    
link: http://www.nature.com/ng/journal/vaop/ncurrent/full/ng.3710.html

Abstract
--------

The goal of population genetics is to quantitatively understand variation of genetic polymorphisms among individuals. Researchers have developed sophisticated statistical methods to capture the complex population structure that underlies observed genotypes in humans. The number of humans that have been densely genotyped across the genome has grown significantly in recent years. In aggregate about 1M individuals have been densely genotyped to date, and if we could analyze this data then we would have a nearly complete picture of human genetic variation. Existing state-of-the-art methods, however, cannot scale to data of this size. To this end, we have developed TeraStructure.

TeraStructure is a new algorithm to fit Bayesian models of genetic variation in human populations on tera-sample-sized data sets (10^12 observed genotypes, e.g., 1M individuals at 1M SNPs). It is a principled approach to approximate Bayesian inference that iterates between subsampling locations of the genome and updating an estimate of the latent population structure. On real and simulated data sets of up to 10K individuals, TeraStructure is twice as fast as existing methods and recovers the latent population structure with equal accuracy. On genomic data simulated at the tera-sample-size scales, TeraStructure continues to be accurate and is the only method that can complete its analysis.
