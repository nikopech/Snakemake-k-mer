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
import os
import dataPreProcessing

# open input file 
fastq_filehandle = open(sys.argv[2], "r")
fastq_filehandle.close

#  output file
output_file = sys.argv[3]
# initialize k from the first argument, convert to an integer
k =int(sys.argv[1])

# Creating the .txt file from fasta file
#files = [f for f in os.listdir('files/') if os.path.isfile(os.path.join('files/', f))]
files = [f for f in os.listdir('files/') if f.endswith('.fasta')]
file = files[0]
filename = file[0:-6]
filename_txt = filename + '.txt'																																																																																																																																

if not os.path.exists('files/'+filename_txt):
	data = dataPreProcessing.read_fasta_file('files/' + file)
	dataPreProcessing.save_results_to_file(data, filename_txt, 'files')
	del data

txt_filehandle = open('files/' + filename_txt, "r")

# Start with an empty dictionary
counts = {}

for row in txt_filehandle:
    	
		#Use of count_kmers routine
		counts =count_kmers.count_kmers(row,k,counts)

#Close input file
txt_filehandle.close

# Sort dictionary by value
sorted_counts = dict(sorted(counts.items(), key=operator.itemgetter(1)))
del counts
#Array of the dictionary values
sorted_data=np.array(list(sorted_counts.values()))
#Array of the dictionary keys
sorted_keys=np.array(list(sorted_counts.keys()))

# Count of the class values
counter=OrderedCounter(sorted_data)   

# Plot data
plot.plot_counts(sorted_data,k)
plot.plot_spectrum(counter,k)

# Count of the frequency of each class
counter_class=OrderedCounter(counter.keys())

# Array of the classes
classes={}
classes=[count for n,count in counter.items() for i in range(count)]

# Variables (X,Y) for the machine learning algorithms
Y = np.array(classes).T

del classes
del counter

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
del prime_numbers

# Variables (X,Y) for the machine learning algorithms
X = np.vstack(([np.array(e_x)],[np.array(list( godel_numbers.values()))])).T


# Create info file
info = open("k={0}/info.txt".format(k),"a")
df_data = pd.DataFrame({'K-mer':sorted_keys,'Value':sorted_data,'Godel_number':np.array(list(godel_numbers.values())),'Entropy':e_x}) 
info.write(df_data.to_string())
info.close()

# Fitting of godel numbers in norm curve
result_godel_fit=godel_f.norm_fit_godel_numbers(godel_numbers,k)

# Free memory
del godel_numbers
del e_x
del counter_class
del sorted_data
del sorted_keys

# Fitting
result_fit_algorithms=[]
result_machine = []
result_fit_algorithms.append(k)
result_machine = fit.fit_func(X,Y,k)
result_fit_algorithms.extend(result_machine)
#PCA fitting result
result_pca_fit = fit.pca(X,Y,k)

#List of results
result_fit_algorithms.append(result_godel_fit)
result_fit_algorithms.append(result_pca_fit)

del X
del Y

df=pd.DataFrame(result_fit_algorithms,index=['k-Lengyh','LinearDiscriminantAnalysis','KNN', 'DecisionTreeClassifier',
	'GaussianNB','MultinomialNB','NormfitGodelNumbering','PCA'],columns=['Results'])
	
# Output result
file= open(output_file,'a')
file.write(df.to_string())
file.close

time_file= open('time.txt','a')
time_file.write( str(k)+'\n')
time_file.close

# For print the Sequence matrix
#matrix.create_matrix(sys.argv[2],k)

del result_fit_algorithms,result_godel_fit,result_pca_fit
