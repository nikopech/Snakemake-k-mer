import datatable

def read_fasta_file(path_to_fasta, write_file = False):
    
    # initialize lists
    out = []

    sequence_header = ""
    sequence = []

    # read lines in input file
    with open(path_to_fasta, 'r') as fh:
        
        for line in fh:

            if line[0] == '>':
                                
                DT_seq = [[sequence_header], [''.join(sequence)], [len(''.join(sequence))]] 
                out.append(datatable.Frame(DT_seq, names = ['sequence_header', 'sequence', 'sequence_length']))

                sequence_header = line[1:len(line)].rstrip("\n")
                sequence = []

            else:
                sequence.append(line.rstrip())


    # add last element
    out.append(datatable.Frame(DT_seq, names = ['sequence_header', 'sequence', 'sequence_length']))

    # convert to datatable
    out = datatable.rbind(out)

    # filter out first row
    out = out[1:out.nrows, :]

    if write_file:
        out.to_csv("out.csv")

            
    return out