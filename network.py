#coding=utf8
from NetworkComponet import *

#theano.config.exception_verbosity="high"
#theano.config.optimizer="fast_compile"

'''
RL for Coref
'''

#### Constants
GPU = True
if GPU:
    print >> sys.stderr,"Trying to run under a GPU. If this is not desired,then modify NetWork.py\n to set the GPU flag to False."
    try:
        theano.config.device = 'gpu'
        print >> sys.stderr,"Use gpu"
    except: pass # it's already set 
    theano.config.floatX = 'float32'
else:
    print >> sys.stderr,"Running with a CPU. If this is not desired,then modify the \n NetWork.py to set\nthe GPU flag to True."
    theano.config.floatX = 'float64'

class MLP_Layers():
    def __init__(self,n_inpt,n_hidden,activate,inpt=None):
        self.params = []
        self.inpt = inpt

        w_h_1,b_h_1 = init_weight(n_inpt,n_hidden,pre="inpt_layer_",special=True,ones=False) 
        self.params += [w_h_1,b_h_1]

        self.hidden_layer_1 = activate(T.dot(self.inpt,w_h_1) + b_h_1)

        w_h_2,b_h_2 = init_weight(n_hidden,n_hidden/2,pre="hidden_layer_1_",special=True,ones=False) 
        self.params += [w_h_2,b_h_2]

        self.hidden_layer_2_ = activate(T.dot(self.hidden_layer_1,w_h_2) + b_h_2)

        w_h_2_,b_h_2_ = init_weight(n_hidden/2,n_hidden/2,pre="hidden_layer_2_",special=True,ones=False) 
        self.params += [w_h_2_,b_h_2_]

        self.hidden_layer_2 = activate(T.dot(self.hidden_layer_2_,w_h_2_) + b_h_2_)

        self.output = self.hidden_layer_2

class MLP_Layers_Worker():
    def __init__(self,n_inpt,n_cluster,n_hidden,activate,inpt=None,inpt_cluster_info=None):
        self.params = []
        self.inpt = inpt
        self.inpt_cluster_info = inpt_cluster_info

        w_h_1,b_h_1 = init_weight(n_inpt,n_hidden,pre="inpt_layer_",special=True,ones=False) 
        self.params += [w_h_1,b_h_1]

        w_h_cluster,b_h_c = init_weight(n_cluster,n_hidden,pre="inpt_layer_",special=True,ones=False) 
        self.params += [w_h_cluster]

        self.hidden_layer_1 = activate(T.dot(self.inpt,w_h_1) + T.dot(self.inpt_cluster_info,w_h_cluster) + b_h_1)

        w_h_2,b_h_2 = init_weight(n_hidden,n_hidden/2,pre="hidden_layer_1_",special=True,ones=False) 
        self.params += [w_h_2,b_h_2]

        self.hidden_layer_2_ = activate(T.dot(self.hidden_layer_1,w_h_2) + b_h_2)

        w_h_2_,b_h_2_ = init_weight(n_hidden/2,n_hidden/2,pre="hidden_layer_2_",special=True,ones=False) 
        self.params += [w_h_2_,b_h_2_]

        self.hidden_layer_2 = activate(T.dot(self.hidden_layer_2_,w_h_2_) + b_h_2_)

        w_h_3,b_h_3 = init_weight(n_hidden/2,1,pre="output_layer_",special=True,ones=False)
        self.params += [w_h_3,b_h_3]

        self.output = T.dot(self.hidden_layer_2,w_h_3) + b_h_3

class MLP_Layers_Manager():
    def __init__(self,n_inpt,n_hidden,activate,inpt=None,pretrain_inpt=None):
        self.params = []
        self.inpt = inpt

        w_h_1,b_h_1 = init_weight(n_inpt,n_hidden,pre="inpt_layer_",special=True,ones=False) 
        self.params += [w_h_1,b_h_1]

        self.hidden_layer_1 = activate(T.dot(self.inpt,w_h_1) + b_h_1)

        w_h_2,b_h_2 = init_weight(n_hidden,n_hidden/2,pre="hidden_layer_1_",special=True,ones=False) 
        self.params += [w_h_2,b_h_2]

        self.hidden_layer_2_ = activate(T.dot(self.hidden_layer_1,w_h_2) + b_h_2)

        w_h_2_,b_h_2_ = init_weight(n_hidden/2,n_hidden/2,pre="hidden_layer_2_",special=True,ones=False) 
        self.params += [w_h_2_,b_h_2_]

        self.hidden_layer_2 = activate(T.dot(self.hidden_layer_2_,w_h_2_) + b_h_2_)

        self.output = self.hidden_layer_2

        self.pretrain_inpt = pretrain_inpt
        self.hidden_layer_1_pretrain = activate(T.dot(self.pretrain_inpt,w_h_1) + b_h_1)
        self.hidden_layer_2_pretrain_ = activate(T.dot(self.hidden_layer_1_pretrain,w_h_2) + b_h_2)
        self.hidden_layer_2_pretrain = activate(T.dot(self.hidden_layer_2_pretrain_,w_h_2_) + b_h_2_)
        w_h_3,b_h_3 = init_weight(n_hidden/2,1,pre="output_layer_",special=True,ones=False)
        self.params += [w_h_3,b_h_3]
        self.output_pretrain = (T.dot(self.hidden_layer_2_pretrain,w_h_3) + b_h_3).flatten()


class Manager():
    def __init__(self,n_inpt,n_single,n_hidden):
        self.params = []

        activate = ReLU

        # For mention_pairs
        self.x_mention_pair_inpt = T.ftensor3("input_pair_embeddings")
        self.x_mention_pair_inpt_pretrain = T.fmatrix("input_pair_embeddings")
        self.pair_layer = MLP_Layers_Manager(n_inpt,n_hidden,activate,self.x_mention_pair_inpt,self.x_mention_pair_inpt_pretrain)
        self.params += self.pair_layer.params
        self.cluster_representation = T.max(self.pair_layer.output,axis=1) ## matrix, each row is the representation of the cluster
        
        # Get mention_pair scores
        w_score_pair,b_score_pair = init_weight(n_hidden/2,1,pre="output_layer_",special=True,ones=False)
        self.params += [w_score_pair,b_score_pair]
        self.cluster_score = (T.dot(self.cluster_representation,w_score_pair) + b_score_pair).flatten()

        self.show_cluster_representation = theano.function(inputs=[self.x_mention_pair_inpt],outputs=[self.cluster_representation])
        self.show_cluster_score = theano.function(inputs=[self.x_mention_pair_inpt],outputs=[self.cluster_score])

        # For single mention
        self.x_inpt_single = T.fmatrix("input_single_embeddings")

        self.single_layer = MLP_Layers(n_single,n_hidden,activate,self.x_inpt_single)
        self.params += self.single_layer.params
        w_score_single,b_score_single = init_weight(n_hidden/2,1,pre="output_layer_",special=True,ones=False)
        self.params += [w_score_single,b_score_single]
        self.single_score = (T.dot(self.single_layer.output,w_score_single) + b_score_single).flatten()

        self.scores_all = T.concatenate((self.single_score,self.cluster_score))
        self.show_score_all = theano.function(inputs=[self.x_mention_pair_inpt,self.x_inpt_single],outputs=[self.scores_all])

        # Policy Graident
        self.policy = softmax(self.scores_all)[0]

        self.predict = theano.function(
            inputs=[self.x_inpt_single,self.x_mention_pair_inpt],
            outputs=[self.policy],
            on_unused_input='warn')

        lr = T.fscalar()
        Reward = T.fscalar("Reward")
        y = T.iscalar('classification')

        l2_norm_squared = sum([(abs(w)).sum() for w in self.params])
        #lmbda_l2 = 0.0000003
        lmbda_l2 = 0.0

        self.get_weight_sum = theano.function(inputs=[],outputs=[l2_norm_squared])

        self.cost = (-Reward) * T.log(self.policy[y] + 1e-6)\
                + lmbda_l2*l2_norm_squared

        grads = T.grad(self.cost, self.params)
        clip_grad = 5.0
        cgrads = [T.clip(g,-clip_grad, clip_grad) for g in grads]
        updates = lasagne.updates.rmsprop(cgrads, self.params, learning_rate=lr)

        self.train_step = theano.function(
            inputs=[self.x_inpt_single,self.x_mention_pair_inpt,y,Reward,lr],
            outputs=[self.cost],
            on_unused_input='warn',
            updates=updates)

        # Pretrain
        self.output_pretrain = T.concatenate((self.single_score,self.pair_layer.output_pretrain))
        self.classification_results = sigmoid(self.output_pretrain)
        self.pretrain_policy = softmax(self.output_pretrain)[0]

        pre_lr = T.fscalar()
        lable = T.ivector()

        pre_cost = (- T.sum(T.log(self.classification_results + 1e-6 )*lable)\
                    - T.sum(T.log(1-self.classification_results+ 1e-6 )*(1-lable)))/(T.sum(lable) + T.sum(1-lable))\
                    + lmbda_l2*l2_norm_squared

        pregrads = T.grad(pre_cost, self.params)
        clip_grad = 5.0
        pre_cgrads = [T.clip(g,-clip_grad, clip_grad) for g in pregrads]
        pre_updates = lasagne.updates.rmsprop(pre_cgrads, self.params, learning_rate=pre_lr)

        self.pre_train_step = theano.function(
            inputs=[self.x_inpt_single,self.x_mention_pair_inpt_pretrain,lable,pre_lr],
            outputs=[pre_cost],
            on_unused_input='warn',
            updates=pre_updates)

        self.predict_pretrain_policy = theano.function(
            inputs=[self.x_inpt_single,self.x_mention_pair_inpt_pretrain],
            outputs=[self.pretrain_policy],
            on_unused_input='warn')


class Worker():
    def __init__(self,n_inpt,n_single,n_cluster,n_hidden):
        ## input = 1738 for cn 1374 for en
        ## embedding for each mention = 855 for cn, 673 for en
        ## pair_feature = 28

        activate=ReLU
        #activate=tanh

        dropout_prob = T.fscalar("probability of dropout")

        self.params = []

        self.x_inpt = T.fmatrix("input_pair_embeddings")

        self.cluster_info_inpt = T.fvector("input_pair_embeddings")

        self.mention_pair_layer = MLP_Layers_Worker(n_inpt,n_cluster,n_hidden,activate,self.x_inpt,self.cluster_info_inpt)
        self.params += self.mention_pair_layer.params

        self.output_layer = self.mention_pair_layer.output.flatten()

        ## for single
        self.x_inpt_single = T.fmatrix("input_single_embeddings")

        self.single_layer = MLP_Layers_Worker(n_single,n_cluster,n_hidden,activate,self.x_inpt_single,self.cluster_info_inpt)
        self.params += self.single_layer.params

        self.output_layer_single = self.single_layer.output.flatten()

        self.output_layer_all = T.concatenate((self.output_layer_single,self.output_layer))

        self.policy = softmax(self.output_layer_all)[0]

        self.predict = theano.function(
            inputs=[self.x_inpt_single,self.x_inpt,self.cluster_info_inpt],
            outputs=[self.policy],
            on_unused_input='warn')

        lr = T.fscalar()
        Reward = T.fscalar("Reward")
        y = T.iscalar('classification')

        l2_norm_squared = sum([(abs(w)).sum() for w in self.params])
        #lmbda_l2 = 0.0000003
        lmbda_l2 = 0.0

        self.get_weight_sum = theano.function(inputs=[],outputs=[l2_norm_squared])

        cost = (-Reward) * T.log(self.policy[y] + 1e-6)\
                + lmbda_l2*l2_norm_squared

        grads = T.grad(cost, self.params)
        clip_grad = 5.0
        cgrads = [T.clip(g,-clip_grad, clip_grad) for g in grads]
        updates = lasagne.updates.rmsprop(cgrads, self.params, learning_rate=lr)

        self.train_step = theano.function(
            inputs=[self.x_inpt_single,self.x_inpt,self.cluster_info_inpt,y,Reward,lr],
            outputs=[cost],
            on_unused_input='warn',
            updates=updates)

        ### for pre_train
        self.classification_results = sigmoid(self.output_layer_all)

        pre_lr = T.fscalar()
        lable = T.ivector()

        pre_cost = (- T.sum(T.log(self.classification_results + 1e-6 )*lable)\
                    - T.sum(T.log(1-self.classification_results+ 1e-6 )*(1-lable)))/(T.sum(lable) + T.sum(1-lable))\
                    + lmbda_l2*l2_norm_squared

        pregrads = T.grad(pre_cost, self.params)
        clip_grad = 5.0
        pre_cgrads = [T.clip(g,-clip_grad, clip_grad) for g in pregrads]

        pre_updates = lasagne.updates.rmsprop(pre_cgrads, self.params, learning_rate=pre_lr)

        self.pre_train_step = theano.function(
            inputs=[self.x_inpt_single,self.x_inpt,self.cluster_info_inpt,lable,pre_lr],
            outputs=[pre_cost],
            on_unused_input='warn',
            updates=pre_updates)

        self.pre_predict = theano.function(
            inputs=[self.x_inpt_single,self.x_inpt,self.cluster_info_inpt],
            outputs=[self.classification_results],
            on_unused_input='warn')

    def show_para(self):
        for para in self.params:
            print >> sys.stderr, para,para.get_value() 

def test_manager():

    #x_inpt = T.fmatrix("input_pair_embeddings")
    x_inpt = T.ftensor3("input_pair_embeddings")

    mention_pair_layer = Manager(3,4,4,ReLU)
    #out = T.max(mention_pair_layer.output,axis=1)
    #out = mention_pair_layer.output

    #f = theano.function(inputs=[x_inpt],outputs=[out])

    #x = T.matrix()
    #maxx = T.max(x,axis=0)
    #f = theano.function(inputs=[x],outputs=[maxx])

    hi = [[[1,2,3],[4,10,6]],[[2,2,3],[5,5,6]]]
    x = [[7,6,5,4]]
    print mention_pair_layer.show_cluster_representation(hi)
    print mention_pair_layer.show_cluster_score(hi)
    print mention_pair_layer.show_score_all(hi,x)

    zp_x = [[2,3,4],[2,8,4]]
    x_sinlge = [[1,2,3,4]]
    lable = [0,1,0]
    pre_lr = 0.5
    print mention_pair_layer.predict_pretrain_policy(x_sinlge,zp_x)
    print mention_pair_layer.pre_train_step(x_sinlge,zp_x,lable,pre_lr)
    print mention_pair_layer.predict_pretrain_policy(x_sinlge,zp_x)

def test_worker():
    r = Worker(3,4,5,4)

    zp_x = [[2,3,4],[2,8,4]]
    x_sinlge = [[1,2,3,4]]
    #x_cluster = [[4,5,6,7,8]]
    x_cluster = [0,0,0,0,0]

    print list(r.predict(x_sinlge,zp_x,x_cluster)[0])
    #print r.predict(zp_x)[0][0]
    #print r.predict(zp_x)[0][1]
    #r.train_step(zp_x,0,0.2,5)
    #r.train_step(zp_x,0,0.2,5)
    #r.train_step(zp_x,0,0.2,5)
    #r.train_step(zp_x,1,1,5)
    #print r.predict(zp_x)[0]

    lable = [0,1,0]
    pre_lr = 0.5
    print r.pre_predict(x_sinlge,zp_x,x_cluster)
    #r.show_para()
    print r.pre_train_step(x_sinlge,zp_x,x_cluster,lable,pre_lr)
    print r.pre_predict(x_sinlge,zp_x,x_cluster)
    #r.show_para()


if __name__ == "__main__":
    test_manager()
    test_worker()
