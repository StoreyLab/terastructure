# bottom line of the file SampleInformation.txt has been removed
f = open("SampleInformation.txt", "r")

SampleInformationHeader = f.readline().strip().split()
INDEX1 = SampleInformationHeader.index("IncludedInDataset952[No1stOr2ndDegreeRelatives]?")
INDEX2 = SampleInformationHeader.index("PopulationName")
indiv = {}

#pull out the list that's in the H952 set
for a in f:
    b = a.strip().split()
    if b[INDEX1] == '1':
        #a stupid trick
        tmp = '000000000000'+b[0]
        tmp = tmp[-5:]
        indiv['HGDP'+tmp] = b[INDEX2]


#in H952 but not in website data; manually figured this out...
del indiv['HGDP00987']
del indiv['HGDP00453']
del indiv['HGDP00452']
del indiv['HGDP01219']
del indiv['HGDP00247']
del indiv['HGDP00248']
del indiv['HGDP01149']
del indiv['HGDP00660']
del indiv['HGDP01344']
del indiv['HGDP01233']
del indiv['HGDP00754']
del indiv['HGDP00944']


g = open("HGDP_FinalReport_Forward.txt","r")
h = open("HGDP_940.tped", "w")
j = open("HGDP_Map.txt", "r")

hgdpindiv = g.readline() #compute indices to keep
hgdpindiv = hgdpindiv.strip()
hgdpindiv = hgdpindiv.split()

valid = [hgdpindiv.index(a) for a in indiv.keys()]

counter = 0
for a in g:
    snpinfo = j.readline().split()
    if snpinfo[1] not in map(str, range(1, 22+1)):
        continue
    a = a.strip()
    a = a.split()
    a = a[1:]
    geno = [a[b] for b in valid]
    outtped = [snpinfo[1], snpinfo[0], "1", snpinfo[2]]
    missing_count = 0.0
    allele1_count = 0.0
    total_allele  = 0.0
    homozygous1 = ''
    for b in xrange(0,len(geno)):
        if geno[b] == '--':
            outtped.append("0 0")
            missing_count = missing_count + 1
        else:
            outtped.append(geno[b][0] + ' ' + geno[b][1])
            if geno[b][1] != geno[b][0]: #heterozygous
                allele1_count = allele1_count + 1.0
            elif homozygous1 == '': #geno[b][0] == geno[b][1] is True
                homozygous1 = geno[b]
                allele1_count = allele1_count + 2.0
            elif homozygous1 == geno[b]:
                allele1_count = allele1_count + 2.0
            total_allele = total_allele + 2.0
    if missing_count <= 47: # 5% missing per SNP max!
        if allele1_count/total_allele >= 0.01 or allele1_count/total_allele <= 0.99: # 1% ma
            h.write(" ".join(outtped)+"\n")
        else:
            print("MAF FAILl; THIS SHOULDN'T HAPPEN FOR HGDP")
#    if counter%1000 ==0:
#        print(str(allele1_count/total_allele) + " " + str(total_allele) + " "+ " ".join(outtped))
    if counter % 50000 == 0:
        print(counter)
    counter = counter+1

#tfam
k = open("HGDP_940.tfam", "w")

for a in valid:
    outtfam = [str(a), hgdpindiv[a], '0', '0', '1', '0']
    k.write(" ".join(outtfam) + "\n")

k.close()
f.close()
g.close()
h.close()
j.close()
