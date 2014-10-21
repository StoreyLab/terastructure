#!/usr/bin/perl

use strict;
use warnings;

for (my $i = 6; $i <= 10; $i += 2) {
    my $cmd = "../src/popgen -file ./data/hgdp_940.012 -n 940 -k $i -l 619883 -label snp -force -rfreq 1000 -idfile ./data/1kG_1500.indv -force -marginf &";
    system($cmd);
    $cmd = "../src/popgen -file ./data/1kG_1500_geno_99.012 -n 1500 -k $i -l 1345735 -label snp -force -rfreq 1000 -idfile ./data/1kG_1500.indv -force -marginf &";
    system($cmd);
}

#
