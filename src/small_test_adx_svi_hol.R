snp_ll = function(x, betas, thetas){
    p = sum(betas*thetas)
    #print(paste(x, p))
    lik = p^x * (1-p)^(2-x)
    if(x==1){
        lik = 2 * lik
    }
    if(lik < 1e-30) return(1e-30)
    return(log(lik))
}


n = 1718
m = 100000
d = 6

argv = commandArgs()

svi.path = argv[length(argv)-1]
adx.path = argv[length(argv)]



sviQ = read.table(paste(svi.path, "/theta.txt", sep=""))
sviQ = t(as.matrix(sviQ[,3:(3+d-1)]))

sviP = scan(paste(svi.path, "/beta.txt", sep=""), nlines=m)
sviP = matrix(sviP, d+1,)
sviP = sviP[2:(d+1),]
sviP = t(sviP)

adxQ = read.table(paste(adx.path, ".Q", sep=""))
adxQ = t(as.matrix(adxQ))

adxP = scan(paste(adx.path, ".P", sep=""), nlines=m)
adxP = matrix(adxP, d,)
adxP = t(adxP)

lengths = as.vector(read.table("TGP_small_test.len", nrows=m)[,1])
lengths = cumsum(lengths)

text.in = scan("TGP_small_test.loc", what=character(), nlines=m)

svi_hll = 0
adx_hll = 0
currsnp = 0
prevind = 0
totalcount = 0


for(loc in text.in){
    len = nchar(loc)
    snp = as.integer(substr(loc, len, len))
    ind = as.integer(substr(loc, 1, len-2)) #ugh, stupid R, don't forget +1
    if( lengths[currsnp+1] <= totalcount ) currsnp = currsnp + 1 #we pushed forward!
    
    #ugh
    
    svi_hll = svi_hll + snp_ll(snp, sviP[currsnp+1,], sviQ[,ind+1])
    adx_hll = adx_hll + snp_ll(snp, 1-adxP[currsnp+1,], adxQ[,ind+1])
    
    totalcount = totalcount + 1
    if(currsnp %% 5000 == 0) print(c(svi_hll,  adx_hll)/totalcount)
}

print(paste("SVI: ", svi_hll/totalcount))
print(paste("ADX: ", adx_hll/totalcount))
