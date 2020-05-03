
data_dir = 'files'
result_dir = 'results'
sample_fns = os.listdir(data_dir)
sample_pfxes = list(map(lambda p: p[:p.rfind('.')],sample_fns))
		   
rule all:
    input: expand(os.path.join(result_dir, '{sample}_result_step1.txt'),sample=sample_pfxes)	   

rule call_snakemake:
	input: 
		data=os.path.join(data_dir, '{sample}.fa')
	output: 
		filename=os.path.join(result_dir, '{sample}_result_step1.txt')
	params: k = config['k']
	message:'Call of Snakefile'
	shell:	"""
			echo "Step 1: Run of input file {input.data} with k={params.k}" >> {output.filename}
			snakemake -s Snakefile-file --config k={params.k}
			"""