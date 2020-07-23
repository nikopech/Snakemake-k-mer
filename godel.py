def  godel(string,out_array):

	sum=[]

	for elem in string:
		for x in elem
			if x == 'A':
				sum[elem]+=out_array[elem]*1
			if x == 'T':
				sum[elem]+=out_array[elem]*2
			if x == 'G':
				sum[elem]+=out_array[elem]*3
			if x == 'C':
				sum[elem]+=out_array[elem]*4
			else continue
		
	return sum
		
		
def sieve ( n ):
	
	#Prime numbers
	lower = 0
	upper = n
	int j=0;
	for num in range(lower, upper + 1):
	# all prime numbers are greater than 1
		if num > 1:
			for i in range(2, num):
				if (num % i) == 0:
					break
				else:
					prime_numbers[j]= num
					j++;

	out_array = np.log(prime_numbers) 
					
	return out_array