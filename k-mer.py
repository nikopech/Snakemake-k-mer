import matplotlib.pyplot as plt
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.mlab as mlab
import matplotlib.pylab as plt 
import pandas as pd
import math 
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000
from fit import fit_func as fit_func
import operator
import count_kmers
from OrderedCounter import OrderedCounter as OrderedCounter
import scipy.stats
import godel_f
from nltk.probability import FreqDist, MLEProbDist

# open input file 
#fastq_filehandle = open(sys.argv[2], "r")
fastq_filehandle = open('files/Saccharomyces_cerevisiae.R64-1-1.dna.toplevel.fa', "r")
# open output file
#output_file = open(sys.argv[3], "w")
# initialize k from the first argument, convert to an integer
k =6#sys.argv[1]
# Start with an empty dictionary
counts = {}

# Loop over each line in the file
for row in fastq_filehandle:
	# Keep the rows with data
	if "REF" not in row:
		# Each row
		row = row.strip()
		#Use of count_kmers routine
		counts =count_kmers.count_kmers(row,k,counts)
	
			
# Sort dictionary by value
sorted_counts = dict(sorted(counts.items(), key=operator.itemgetter(1)))

#Array of the dictionary values
data=np.array(list(sorted_counts.values()))

# Count of the class values
counter=OrderedCounter(data)   

# Count of the frequency of each class
counter_class=OrderedCounter(counter.values())

# Array of the classes
classes={}
classes=[count for n,count in counter.items() for i in range(count)]


# Calculate entropy of each element
# calculate probability for each byte as number of occurrences / array length
probabilities = [n_x/sum(sorted_counts.values()) for x,n_x in sorted_counts.items()]

#print(sum(probabilities))

# calculate per-character entropy fractions
e_x = [-p_x*math.log(p_x,2) for p_x in probabilities]



# Second way for calculate entropy with nltk library
'''
freq_dist = FreqDist(sorted_counts)
prob_dist = MLEProbDist(freq_dist)
px = [prob_dist.prob(x) for x,n_x in sorted_counts.items()]
e_x = [-p_x*math.log(p_x,2) for p_x in px]
'''


# Calculate the prime numbers for Godel Numbers
prime_numbers=[]
prime_numbers=godel_f.sieve(k)
# Calculate Godel Numbers
godel_numbers={}
godel_numbers=godel_f.godel(sorted_counts,prime_numbers,godel_numbers)


# Variables (X,Y) for the machine learning algorithms
Y = np.array(ar).T
X = np.vstack(([np.array(e_x)],[np.array(list( godel_numbers.values()))])).T

# Shape of the variables
print(X.shape)
print(Y.shape)

# Fitting
result_fit_algorithms = fit_func(X,Y)
# Fitting of godel numbers in norm curve
result_godel_fit=godel_f.norm_fit_godel_numbers(godel_numbers,k)
# Mean calculate of results
result=(result_fit_algorithms+result_godel_fit)/2

# Plot 
count_kmers.plot_counts(data,k)
count_kmers.plot_spectrum(counter,k)
count_kmers.plot_class(counter,k)

# Output result
#output_file.write("%f" %result)
#output_file.close
