../src/terastructure -file test.bed -n 200 -l 10000 -k 3 -stochastic -nthreads 1 -rfreq 1000 -seed 1234 -label test
cd n200-k3-l10000-test-seed1234/
../../src/terastructure -file ../test.bed -n 200 -l 10000 -k 3 -stochastic -nthreads 1 -compute-beta
