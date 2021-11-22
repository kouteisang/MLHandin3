from utils import *
from viterbi import *
from TrainByCounting import  *

if __name__ == '__main__':
    for i in range(1, 6):
        genome_transfer_list = []
        ann_transfer_list = []
        for j in range(1, 6):
            if i == j:
                continue
            file_genome = "genome" + str(j)
            file_ann = 'true-ann'+str(j)
            genome = read_fasta_file(file_genome + '.fa')[file_genome] #x
            ann = read_fasta_file(file_ann + '.fa')[file_ann] #z
            genome_transfer = translate_observations_to_indices(genome) # x
            ann_transfer = translate_ann_to_indices(ann, genome) #z
            genome_transfer_list.extend(genome_transfer)
            ann_transfer_list.extend(ann_transfer)
        model = training_by_counting(43, 4, genome_transfer_list, ann_transfer_list)
        save_model(model, 'models/validated_on_' + str(i))
        print("done with training model %d turn", i)
        print("====================init_probs====================")
        print(model.init_probs)
        print("====================emission_probs====================")
        print(model.emission_probs)
        print("====================trans_probs====================")
        print(model.trans_probs)
    print("done with model training...")

    for i in range(1, 2):
        print("Round "+str(i)+" predicting...")
        genome_transfer = translate_observations_to_indices(read_fasta_file('genome'+str(i)+'.fa')["genome"+str(i)])
        model = load_model('models/validated_on_' + str(i))
        w = compute_w_log(model, genome_transfer)
        path = backtrack_log(model, genome_transfer, w)
        with open('validation/pred-ann'+str(i)+'.fa', 'w') as f:
            prefix=">pred-ann" + str(i)+"\n"
            f.write(prefix+path)
        print("Round "+str(i)+" done")



