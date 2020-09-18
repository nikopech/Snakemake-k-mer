import sys
import matplotlib.pyplot as plt
import math
import re
import os,glob
from pathlib import Path
import numpy as np
from numpy import cumsum, ones
import pandas as pd 
import operator
import seaborn as sns; sns.set()
import matrix

# Every file in results directory
entries = os.listdir('results/')

# Compare variables
algorith_statistics_best=[0,0,0,0,0,0,0,0]
compare=0

# For every file
for entry in entries:
	f =open( os.path.join('results/', entry),'r')
	X = pd.read_fwf(os.path.join('results/', entry))
	Lines = X['Results'].tolist()

	compare=0
	i=-1

	# Compare each value
	for line in Lines: 
		i=i+1
		# If the value is better
		if (line != 'nan'):
			if(i==0):
				k=int(line)
			else:
				if(float(line) > float(algorith_statistics_best[i])) :
					compare=compare+1
				
	#If the file is better from the previous
	if(compare>3):
		algorith_statistics_best=Lines
		best_file=entry
		best_k=k


# Print the best file 
output_file= open(sys.argv[1],'w')
output_file.write("The best fit is for file=%s\n"%best_file)
output_file.write("The best fit is for k=%i"%best_k)
output_file.close

# Print the time.txt in sort order
results =[]
# This opens a handle to your file, in 'r' read mode
with open('time.txt') as fp:
	# Read in all the lines of your file into a list of lines
	for line in fp:
		results.append(line)

float_results=[ float(x) for x in results ]

data=float_results[1::2]
y=np.array(float_results[0::2])
y = y.astype(int)

dictionary={}
dictionary=dict(zip(y, data))

sort_dictionary=dict(sorted(dictionary.items(),key=operator.itemgetter(0)))
data_x= np.cumsum(np.array(list(sort_dictionary.values())))

# Plot time of k files
plt.plot(data_x,sort_dictionary.keys(),color="blue")
plt.ylabel('number of k files')
plt.xlabel('time(sec)')

plt.title('Time plot')
plt.savefig('time.png')
plt.show()


matrix.create_matrix(sys.argv[2],best_k)
