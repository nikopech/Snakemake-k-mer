import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.pylab as plt 
import pandas as pd
import math 
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000
import fit 
import operator
import count_kmers
from OrderedCounter import OrderedCounter as OrderedCounter
import scipy.stats
import godel_f
from nltk.probability import FreqDist, MLEProbDist
import plot

# open input file 
fastq_filehandle = open(sys.argv[2], "r")
#  output file
output_file = sys.argv[3]
# initialize k from the first argument, convert to an integer
k =int(sys.argv[1])

# Start with an empty dictionary
counts = {}
i=0
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
unsorted_values=list(counts.values())

#Array of the dictionary values
unsorted_data=np.array(unsorted_values)
plot.plot_counts(unsorted_data,k,1)

sorted_data=np.array(list(sorted_counts.values()))
plot.plot_counts(sorted_data,k,2)

del counts
del unsorted_data
del unsorted_values

# Count of the class values
counter=OrderedCounter(sorted_data)   
plot.plot_spectrum(counter,k)
del sorted_data

# Count of the frequency of each class
counter_class=OrderedCounter(counter.values())
plot.plot_class(counter_class,k)

# Array of the classes
classes={}
classes=[count for n,count in counter.items() for i in range(count)]

del counter
del counter_class

#Calculate entropy with nltk library

freq_dist = FreqDist(sorted_counts)
prob_dist = MLEProbDist(freq_dist)
px = [prob_dist.prob(x) for x,n_x in sorted_counts.items()]
e_x = [-p_x*math.log(p_x,2) for p_x in px]

del freq_dist
del prob_dist
del px

# Calculate the prime numbers for Godel Numbers
prime_numbers=[]
prime_numbers=godel_f.sieve(k)

# Calculate Godel Numbers
godel_numbers={}
godel_numbers=godel_f.godel(sorted_counts,prime_numbers,godel_numbers)

# Variables (X,Y) for the machine learning algorithms
Y = np.array(classes).T
X = np.vstack(([np.array(e_x)],[np.array(list( godel_numbers.values()))])).T

del prime_numbers
del e_x

# Fitting
result_fit_algorithms=[]
result_fit_algorithms = fit.fit_func(X,Y,k)
# Fitting of godel numbers in norm curve
result_godel_fit=godel_f.norm_fit_godel_numbers(godel_numbers,k)

del godel_numbers

#PCA fitting result
result_pca_fit = fit.pca(X,Y,k)


#List of results
result_fit_algorithms.append(result_godel_fit)
result_fit_algorithms.append(result_pca_fit)
print(result_fit_algorithms)
# Output result
with open(output_file,'w') as file:
	for item in result_fit_algorithms:
		file.write("%s\n" % item)
	
fastq_filehandle.close
del X
del Y