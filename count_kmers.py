import numpy as np
import matplotlib.pylab as plt 
import pandas as pd

def count_kmers(read, k,counts):
    """Count kmer occurrences in a given read.

    Parameters
    ----------
    read : string
        A single DNA sequence.
    k : int
        The value of k for which to count kmers.

    Returns
    -------
    counts : dictionary, {'string': int}
        A dictionary of counts keyed by their individual kmers (strings
        of length k).

    Examples
    --------
    >>> count_kmers("GATGAT", 3)
    {'ATG': 1, 'GAT': 2, 'TGA': 1}
    """
   
    # Calculate how many kmers of length k there are
    num_kmers = len(read) - k + 1
    # Loop over the kmer start positions
    for i in range(num_kmers):
        # Slice the string to get the kmer
        kmer = read[i:i+k]
        # Add the kmer to the dictionary if it's not there
        if kmer not in counts:
            counts[kmer] = 0
        # Increment the count for this kmer
        counts[kmer] += 1
    # Return the final counts
    return counts
	
def plot_counts(data,k):

	plt.ylabel('counts',fontsize=12)
	plt.title( '%i plot of values' %k,fontsize=16)
	plt.xlabel( '%i place' %k,fontsize=12)
	plt.plot(data,'bo')
	# Save of result
	plt.savefig('Step1:k-mer-for-%i.png'%k)
	plt.clf()
	
def plot_spectrum(counter,k):
	plt.ylabel('counts',fontsize=12)
	plt.title( '%i-mer spectrum' %k,fontsize=16)
	plt.xlabel( '%i -mer frequency' %k,fontsize=12)
	plt.plot(list(counter.values()),list(counter.keys()),'bo')
	# Save of result
	plt.savefig('Step2:k-mer-for-%i.png'%k)
	plt.clf()
	
def plot_class(counter2,k):

		
		plt.bar(counter2.values(),counter2.keys(), align='center', alpha=0.5)
		plt.ylabel('counts',fontsize=12)
		plt.title( 'class-of-k-%i.png'%k,fontsize=16)
		plt.xlabel( 'class',fontsize=12)
		
		plt.show()
		plt.savefig('class-of-k-%i.png'%k)
		plt.clf()
		