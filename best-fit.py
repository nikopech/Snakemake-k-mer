import sys
import matplotlib.pyplot as plt
import math
import tkinter as tk
from tkinter.simpledialog import askfloat
import re
import os,glob
from pathlib import Path


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
