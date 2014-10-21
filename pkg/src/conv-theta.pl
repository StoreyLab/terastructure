#!/usr/bin/perl

use strict;
use warnings;

#
# x.pl <theta-file-name> <K>

die "arg: <theta> file and K needed\n" if (!$ARGV[0] || !$ARGV[1]);
my $K = $ARGV[1];
my $u = 1.0/$K;
my $s = 0;
open F, "<$ARGV[0]";
while(<F>) {
    if ($_ =~ /(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\d+)/) {
	my $a = $3  - $u;
	my $b = $4  - $u;
	my $c = $5  - $u;
	my $d = $6  - $u;
	my $e = $7  - $u;
	my $f = $8  - $u;
	$s = $a + $b + $c + $d + $e + $f;
	die "unexpected values in <theta>" if !($s > .0);
	printf "%d\t%s\t%.8f\t%.8f\t%.8f\t%.8f\t%.8f\t%.8f\t%d\n",
	$1, $2, $a/$s, $b/$s, $c/$s, $d/$s, $e/$s, $f/$s, $9;
    }
}

close F;
