#coding:utf-8

class Config(object):
    init_scale = 0.04
    learning_rate = 0.001
    max_grad_norm = 15
    num_layers = 3
    num_steps = 25 # number of steps to unroll the RNN for
    hidden_size = 1000 # size of hidden layer of neurons
    iteration = 40
    save_freq = 5 #The step (counted by the number of iterations) at which the model is saved to hard disk.
    keep_prob = 0.5
    batch_size = 32
    model_path = './model/Model' #the path of model that need to save or load

    #parameters for generation
    save_time = 40 #load save_time saved models
    is_sample = True #true means using sample, if not using max
    is_beams = True #whether or not using beam search
    beam_size = 2 #size of beam search
    len_of_generation = 200 #The number of characters by generated
    start_sentence = u'沒有菸抽的日子' #the seed sentence to generate text