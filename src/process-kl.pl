#!/usr/bin/perl

open F, "<KL.txt";
while (<F>) {
    my $itr = $_;
    my $val = <F>;
    if ($val =~ /\[\d+\]\s+(\S+)/) {
	$val = $1;
	$val =~ s/^\s+|\s+$//g ;
	chomp $itr;
	print "$itr $val\n";
    }
}
close F;
