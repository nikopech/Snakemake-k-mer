import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy import array


def count_kmers(read, k):
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
	
# open input file 
fastq_filehandle = open(sys.argv[2], "r")
# open output file
output_file = open(sys.argv[3], "w")
# initialize k from the first argument, convert to an integer
k = int(sys.argv[1])
# Start with an empty dictionary
counts = {}

# Loop over each line in the file
for row in fastq_filehandle:
	# Keep the rows with data
	if "REF" not in row:
		# Each row
		row = row.strip()
		#Use of count_kmers routine
		counts =count_kmers(row,k)

#Save data in outpout file
output_file.write("The %i-mer dictionary is:" %k)
output_file.write(repr(counts))
#List of the dictionary values
values=list(counts.values())
#Array of the dictionary values
data=np.array(values)
# Design
plt.ylabel('counts',fontsize=12)
plt.title( '%i-mer spectrum' %k,fontsize=16)
plt.xlabel( '%i -mer frequency' %k,fontsize=12)
plt.plot(data)
# Save of result
plt.savefig('k-mer-for-%i.png'%k)

# Close the files
fastq_filehandle.close()
output_file.close()