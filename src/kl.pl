#!/usr/bin/perl

use strict;
use warnings;

my $cmd = "ls .";
my @files = split ' ', `$cmd`;
my $last_spawned_id = 0;
my $MAGIC = "a1b2c3d4e5";

my $c = 0;
my $id = 0;
my $n = 0;
foreach my $f (@files) {
    if ($f =~ /theta_(\d+)/) {
	if ($n < 10) {
	    if ($c % 5 == 0) {
		system("echo $1 >> kl_$n.txt");
		$cmd = "XYZ=$MAGIC R -q --slave -f /scratch/pgopalan/popgen/src/simulation_KL.R --args theta_$1.txt >> kl_$n.txt &";
		print($cmd."\n");
		if (system($cmd) < 0) {
		    exit(-1);
		}
		$last_spawned_id = $1;
		$n++;
	    }
	    $c++;
	    $id = $1;
	} else {
	    my $m = 0;
	    do { 
		# wait for all to complete
		$cmd = "ps eax | grep popgen | grep $MAGIC | grep -v grep | wc -l";
		$m = `$cmd` + 0;
		sleep(10);
		print ("$m processes running\n");
	    } while($m > 0);
	    print("all done!\n");
	    $n = 0;
	}
    }
}

if ($last_spawned_id != $id) {
    $cmd = "R -q --slave -f /scratch/pgopalan/popgen/src/simulation_KL.R --args theta_$id.txt >> kl_last.txt &";
    #$cmd = "R -f /scratch/pgopalan/popgen/src/simulation_KL.R theta_$id.txt >> kl.txt";	
    print($cmd."\n");
    if (system($cmd) < 0) {
	exit(-1);
    }
}
