import sys
import matplotlib.pyplot as plt
import math
import tkinter as tk
from tkinter.simpledialog import askfloat
import re
import os,glob
from pathlib import Path
import numpy as np
from numpy import cumsum, ones

# Every file in results directory
entries = os.listdir('results/')

# Compare variables
algorith_statistics_best=[0,0,0,0,0,0,0]
compare=0

# For every file
for entry in entries:
	f =open( os.path.join('results/', entry),'r')
	Lines = f.readlines()
	compare=0
	i=-1
	# Compare each value
	for line in Lines: 
		i=i+1
		# If the value is better
		if (line != 'nan'):
			if (float(line) > float(algorith_statistics_best[i])) :
				compare=compare+1
	#If the file is better from the previous
	if(compare>3):
		algorith_statistics_best=Lines
		best_file=entry

# Print the best file 
output_file = open(sys.argv[-1], "w")
output_file.write("The best fit is for %s"%best_file)
output_file.close

results =[]
# This opens a handle to your file, in 'r' read mode
with open('time.txt') as fp:
	# Read in all the lines of your file into a list of lines
	for line in fp:
		results.append(line)

float_results=[ float(x) for x in results ]

data= np.cumsum(float_results)
y=np.arange(1,data.size+1,1)
# Plot time of k files
plt.plot(data,y)
plt.ylabel('number of k files')
plt.xlabel('time(sec)')
plt.yticks(range(1,data.size+1))
plt.title('Time plot')
plt.savefig('time.png')
plt.show()