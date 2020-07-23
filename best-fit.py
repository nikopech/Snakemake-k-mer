import sys
import matplotlib.pyplot as plt
import math
import tkinter as tk
from tkinter.simpledialog import askfloat
import re
import os,glob

min=float('inf')

for filename in range(len(sys.argv)-1):
	f = open(sys.argv[filename], 'r')
	first_line = f.readline().rstrip()
	if (first_line != 'nan'):
		if (float(first_line) < min) :
			min=first_line
			min_file=filename
	
output_file = open(sys.argv[-1], "w")
output_file.write("The best fit is for %s"%min_file[:-4] )
output_file.close
