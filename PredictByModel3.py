from utils import *
from viterbi import *

if __name__ == '__main__':
    #use the third model to predict the gene
    model = load_model('models/validated_on_' + str(3))
    for i in range(6,7):
        print("Round " +str(i)+ " begin")
        genome_transfer = translate_observations_to_indices(read_fasta_file('genome'+str(i)+'.fa')["genome"+str(i)])
        w = compute_w_log(model, genome_transfer)
        path = backtrack_log(model, genome_transfer, w)
        with open('predict/pred-ann' + str(i) + '.fa', 'w') as f:
            prefix = ">pred-ann" + str(i) + "\n"
            f.write(prefix + path)
        print("Round " + str(i) + " done")



