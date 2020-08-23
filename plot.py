import seaborn as sns
import numpy as np
import matplotlib.pylab as plt 
import pandas as pd
import os

def plot_counts(data,k):
	"""Plot counts

    Parameters
    ----------
    data : array
        The array with the k-mer values.
	k: int
		K-mer value
	"""
	plt.ylabel('counts')
	plt.title( '%i-mer values plot' %k)
	plt.xlabel( 'position depth %i-mer' %k)
	plt.plot(data,'bo')
	plt.ylim( data.min()-1, data.max()+1 )
	# Save of result
	output_dir ="k={}".format(k)
	os.makedirs(output_dir)
	plt.savefig('k=%i/values of k-mer-for-%i.png'%(k,k))
	plt.clf()
	
	
def compare_algorithms(results,names,k):
	"""Box plot plot of the ML algorithms
    Parameters
    ----------
    results : array
        The array with the results of ML algorithms.
	names: int
		The array with the names of ML algorithms.
	"""
	fig = plt.figure()
	fig.suptitle('Algorithm Comparison')
	ax = fig.add_subplot(111)
	plt.boxplot(results)
	ax.set_xticklabels(names)
	plt.show()
	plt.savefig('k=%i/Algorithm box plot for k=%i.png'%(k,k))
	plt.clf()	
	
	
def godel_histogram(x,pdf_norm,k):	
	"""Plot godel numbers histogram.

    Parameters
    ----------
    x : array
        The array with the godel numbers values.
	k: int
		K-mer value
	"""
	plt.hist(x, bins='auto', density=True)
	plt.plot(x, pdf_norm, label='Fitted normal distribution')
	plt.xlabel('Godel Numbering')
	plt.ylabel('Probability')
	plt.title('Distribution of Godel Numbers')
	plt.savefig('k=%i/Godel-Histogram-for-%i.png'%(k,k))
	plt.show()
	plt.clf()

def plot_spectrum(counter,k):
	"""Plot the k-mer spectrum

    Parameters
    ----------
    counter : array
        The array with the k-mer values.
	k: int
		K-mer value
	"""
	plt.ylabel('counts')
	plt.title( '%i-mer spectrum plot' %k)
	plt.xlabel( '%i -mer frequency' %k)
	plt.plot(list(counter.values()),list(counter.keys()),'bo')
	plt.ylim( min(counter.keys())-1, max(counter.keys())+1 )
	#plt.xticks(np.arange(0, max(list(counter.values())), step=list(max(counter.values()))/6))
	# Save of result
	plt.savefig('k=%i/Spectrum-k-mer-for-%i.png'%(k,k))
	plt.clf()
	
def plot_class(data_counter,k):
	"""Plot bar of the classes of the k-mer values.

    Parameters
    ----------
    data_counter : array
        The array with the classes values.
	k: int
		K-mer value
	"""
	plt.bar(data_counter.values(),data_counter.keys(), align='center', alpha=0.35)
	plt.ylabel('counts')
	plt.title( 'class-of-k-%i.png'%k)
	plt.xlabel( 'class')
	plt.xlim(min(data_counter.values())-0.35,max(data_counter.values())+0.35)
	plt.ylim(min(data_counter.keys())-1,max(data_counter.keys())+1)
	plt.show()
	plt.savefig('k=%i/class-of-k-%i.png'%(k,k))
	plt.clf()