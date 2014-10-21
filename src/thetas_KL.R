#run in fit folder from popgen. assuming theta.txt and S_out.txt exist
#compute the minimum symmetric KL between theta.txt and S_out.txt for
#all possible permutations of the labels

KL = function(theta0, theta1){
    theta0[theta0<1e-30] = 1e-30
    theta1[theta1<1e-30] = 1e-30
    sum(theta0*log(theta0/theta1))
}


S = read.table("S_out.txt")
S = as.matrix(S)

d = ncol(S)
n = nrow(S)

FILELIST=system("ls -t theta_*",intern=TRUE)

THETA = read.table(FILELIST[1]) #newest file
THETA = as.matrix(THETA[,3:(3+d-1)])

if(n > 5000){
    IND = sample(1:n, 5000)
} else{
    IND = 1:n
}

S_small = S[IND,]


tmp = rep(list(1:d), d)
A = as.matrix(do.call(expand.grid, tmp))
FULL_IND = A[apply(A,1,function(x){length(unique(x))})==d,]

min_SVIKL = 1e10
for(i in 1:nrow(FULL_IND)){
    marginf = THETA[IND, FULL_IND[i,]]

    SVIKL = rep(0, length(IND))
    for(j in 1:length(IND)){
        SVIKL[j] = KL(S_small[j,], marginf[j,])
    }

    MEDIAN = median(SVIKL)

    if(MEDIAN < min_SVIKL){
        min_SVIKL = MEDIAN
        SVIIND = FULL_IND[i,]
    }
}


SVIKL = rep(0, n)

median_KLs = rep(0, length(FILELIST))

for(i in 1:length(FILELIST)){
    THETA = read.table(FILELIST[i])
    THETA = as.matrix(THETA[,3:(3+d-1)])[,SVIIND]

    median_KLs[i] = median(sapply(1:n, function(j) {KL(S[j,], THETA[j,])}))
}

print(as.matrix(rev(median_KLs)))

#print all the thetas for the last theta_#####.txt
THETA = read.table(FILELIST[1])
THETA = as.matrix(THETA[,3:(3+d-1)])[,SVIIND]

write.table(sapply(1:n, function(j) {KL(S[j,], THETA[j,])}), file="KL_by_indv.txt", row.names=F, col.names=F)

