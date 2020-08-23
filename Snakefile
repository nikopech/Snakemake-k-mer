
k=[17,14,13,26,27,30,32,36,40,43,41]

data_dir = 'files'
result_dir = 'results'
data_fns = os.listdir(data_dir)
data_pfxes = list(map(lambda p: p[:p.rfind('.')],data_fns))

rule all:
	input: first_result = expand(os.path.join(result_dir,'{sample}_result_{kmer}.txt'),sample=data_pfxes,kmer=k),
	       second_result = 'result.txt'
			

rule k_mer_files:
	input:
	    data = expand(os.path.join(data_dir,'{sample}.fa'),sample=data_pfxes)
	output:
	    out_file = os.path.join(result_dir,'{sample}_result_{kmer}.txt')
	message:
	    'Call {input.data} with k={wildcards.kmer}' 
	threads:8
	priority:2
	shell:r"""
			start=$(date +%s.%N)
			python k-mer.py {wildcards.kmer} {input.data} {output.out_file} 
			dur=$(echo "$(date +%s.%N) - $start" | bc) 
			printf "%.6f," $dur >> "time.txt" 
			"""

rule best_fit:
	output:
	    final_output = 'result.txt'
	priority:1
	message:
	    'Best kmer search'
	shell: r"""
			python best-fit.py {output.final_output} 
			"""		
