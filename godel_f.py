import numpy as np
import matplotlib.pylab as plt 
import math 
from scipy.stats import norm
from scipy import stats
from sklearn.metrics import mean_squared_error
import pandas as pd




def  godel(read,out_array,godel_numbers):
	"""Count godel numbers in a given read.

    Parameters
    ----------
    read : dictionay
        Dictionary with the DNA sequences.
    out_array : array
        The array with the prime numbers.
	godel_numbers : dictionary
		Dictionary with the godel numbers.
    Returns
    -------
    godel_numbers : dictionary, {'string': float}
        A dictionary of strings keyed by their individual godel numbers (strings
        of length k).
 
    """

	# For every string
	for elem in read.keys():
		point_in_string=0
		# If the key is new
		if elem not in godel_numbers:
			godel_numbers[elem] = 0
		# Calculate the godel number for the key
		for x in elem:
				if x == 'A':
					godel_numbers[elem]+=out_array[point_in_string]*1
					point_in_string=point_in_string+1
				if x == 'T':
					godel_numbers[elem]+=out_array[point_in_string]*4
					point_in_string=point_in_string+1
				if x == 'G':
					godel_numbers[elem]+=out_array[point_in_string]*3
					point_in_string=point_in_string+1
				if x == 'C':
					godel_numbers[elem]+=out_array[point_in_string]*2
					point_in_string=point_in_string+1
			
	return godel_numbers

def sieve(n):
	"""Count the array of prime numbers of k-mer

    Parameters
    ----------
    n : int
        K-mer number.
    
    Returns
    -------
    out_array : array, { float}
        An array with the log prime numbers.
 
    """
	#Prime numbers
	lower = 0
	upper = 100
	j=0
	prime_numbers =[]
	for num in range(lower, upper + 1):
	
		if j<n:
			# all prime numbers are greater than 1
			if num > 1:
				for i in range(2, num):
					if (num % i) == 0:
						break
				else:
					prime_numbers.insert(j,num)
					j=j+1;

	out_array = np.log(prime_numbers) 
					
	return out_array


def norm_fit_godel_numbers(godel_numbers,k):
	"""Count the error and plot the distribution of godel numbers from norm curve.

    Parameters
    ----------
	godel_numbers : dictionary
		Dictionary with the godel numbers.
    k : int
        K-mer number.
    
    Returns
    -------
    mean_squared_error :  float
        The error from norm curve.
 
    """
	x=np.array(list(godel_numbers.values()))                                                                                                                                                                                                                                                                                                                                                                                                                                          
	n, bins, patches = plt.hist(x,bins='auto', density=True, facecolor = 'grey', alpha = 0.5, label='before');       

	plt.xlabel('Godel Numbers')
	plt.ylabel('probability')
	plt.title('Distribution of Godel Numbers');
	plt.savefig('HistogramGodel-k-mer-for-%i.png'%k)
	plt.show()
	plt.clf()
	
	
	# fit Gaussian distribution
	x.sort()
	mean, std = stats.norm.fit(x, loc=0)
	pdf_norm = stats.norm.pdf(x, mean, std)
	# visual inspection
	plt.hist(x, bins='auto', density=True)
	plt.plot(x, pdf_norm, label='Fitted normal distribution')
	plt.xlabel('Godel Numbers)')
	plt.ylabel('Probability')
	plt.title('Distribution of Godel Numbers')
	plt.savefig('Histomer-for-%i.png'%k)
	plt.show()
	plt.clf()
	
	return 1/mean_squared_error(pdf_norm, x ,squared=False)
	
	'''
	def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

	rmse_val = rmse(np.array(d), np.array(p))	
	print("rms error is: " + str(rmse_val))
	return rmse_val
	'''