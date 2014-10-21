#!/usr/bin/perl -w

for my $admix ("001", "01", "1") {
    for (my $i = 4; $i <= 6; $i += 1) {
	my $cmd = "cd ${admix}; ../../pop -file BN_sim_dirichlet_admix_${admix}.012 -n 1000 -l 100000 -label SNP -rfreq 1000 -k $i -force &";
	system($cmd);
    }
}
