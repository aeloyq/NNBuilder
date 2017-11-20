# -*- coding: utf-8 -*-
"""
Created on  四月 27 23:10 2017

@author: aeloyq
"""
from simple import *


class beamsearch(LayerBase):
    def __init__(self, **kwargs):
        LayerBase.__init__(self, **kwargs)

    def apply_predict_contexts(self, batchsize):
        '''
        Return two ordereddicts
        First one's keys are names of contexts and values are corresponding updates graph of contexts
        Second one's keys are names of contexts and values are corresponding shapes of contexts
        :param batchsize: scalar
        :return:
        '''
        predict_contexts, predict_contexts_shapes = OrderedDict(), OrderedDict()
        return predict_contexts, predict_contexts_shapes

    def apply_predict_initstates(self, batchsize, beamsize):
        '''
        Return two ordereddicts
        First one's keys are names of initstates and values are corresponding attr of initstates
        Second one's keys are names of initstates and values are corresponding shapes of initstates
        :param batchsize: scalar
        :return:
        '''
        predict_initstates_attr = OrderedDict()
        predict_initstates_shapes = OrderedDict()
        return predict_initstates_attr, predict_initstates_shapes

    def apply_init_predict_step(self, contexts, initstates, batchsize, beamsize):
        '''
        Use shared init_states as input return
        1.A graph of probability
          Which has size of
          (batchsize,beamsize,catglorysize)
          Or
          (batchsize,catglorysize)
        2.Updates of initstates
        :param initstates:
        :return:
        '''
        probability = T.zeros([batchsize, beamsize, 100], [None, None, None], kernel.config.floatX)
        updates = []
        for k, v in initstates.items():
            updates.append((v, v))
        return probability, updates

    def init_beamsearch(self, batchsize, beamsize):
        beamchoice = kernel.placeholder('BeamChoice', ['batch', 'search'], kernel.config.catX)
        contexts = OrderedDict()
        initstates = OrderedDict()
        predict_contexts, predict_contexts_shapes = self.apply_predict_contexts(batchsize)
        predict_initstates_attr, predict_initstates_shapes = self.apply_predict_initstates(batchsize, beamsize)
        contexts_updates = []
        for k, v in predict_contexts.items():
            contexts[k] = kernel.shared(np.ones(predict_contexts_shapes[k], kernel.config.floatX), k, v.attr)
            contexts_updates.append((contexts[k], v))
        for k, v in predict_initstates_shapes.items():
            initstates[k] = kernel.shared(np.ones(predict_initstates_shapes[k], kernel.config.floatX), k,
                                          predict_initstates_attr[k])
        fn_contexts = kernel.compile([self._model_inputs['X'], self._model_inputs['X_Mask']], updates=contexts_updates, strict=False)
        chosen_initstates = OrderedDict()
        for k, v in initstates.items():
            raw_shape = v.shape[2:]
            raw_attr = v.attr[2:]
            raw_flatten = v.reshape([batchsize * beamsize] + raw_shape, [None] + raw_attr)
            chosen_initstate = raw_flatten[beamchoice.flatten()].reshape([batchsize, beamsize] + raw_shape,
                                                                         ['batch', 'search'] + raw_attr)
            chosen_initstates[k] = chosen_initstate

        probability, initstates_updates = self.apply_init_predict_step(contexts, chosen_initstates, batchsize, beamsize)
        fn_step = kernel.compile([beamchoice], [probability], updates=initstates_updates, strict=False)

        return contexts, initstates, fn_contexts, fn_step

    def get_beamsearch_predict_function(self, catglorysize, batchsize, maxlen=50, beamsize=12):
        contexts, initstates, init_predict, feed_step = self.init_beamsearch(batchsize, beamsize)

        def gen_sample(*inputs):
            #  extract inputs
            X, X_Mask = inputs[:2]
            #  initcontexts
            #  no returns since all the changes are about shared values
            #  we just update them
            init_predict(X, X_Mask)
            # init beamsearch vars
            samples_all = []
            trans_all = []
            mod_all = []
            scores_all = []
            beamchoices = np.zeros([batchsize, beamsize], kernel.config.catX)
            for _ in range(batchsize):
                samples_all.append([])
                trans_all.append([])
                mod_all.append([])
                scores_all.append([])
            masks = np.ones([batchsize, beamsize], 'int8')
            probsums = np.zeros([batchsize, beamsize], kernel.config.floatX)
            #  first search all the same for every search channel
            prob_b_k_v = feed_step(beamchoices)[0]
            prob_b_k_v = - np.log(prob_b_k_v)
            prob_b_kv = prob_b_k_v[:, 0]
            #  beamsearch step by step
            for n_step in range(1, maxlen + 1):
                for batch in range(batchsize):
                    bmask = masks[batch]
                    btrans = trans_all[batch]
                    bmods = mod_all[batch]
                    bsample = samples_all[batch]
                    bscore = scores_all[batch]
                    prob = prob_b_kv[batch].flatten()
                    #  how many live channel to search in this minibatch
                    #  set mask for this minibatch
                    bnlive = bmask.sum()
                    bndead = beamsize - bmask.sum()
                    # shift live channel to left and dead ones to right
                    for i in range(beamsize):
                        if i < bnlive:
                            bmask[i] = 1
                        else:
                            bmask[i] = 0
                    # find top (k-dead_channel)
                    b_step_out = prob.argpartition(bnlive)[:bnlive]
                    #  append trans and mod
                    b_step_trans = b_step_out // catglorysize
                    b_step_mod = b_step_out % catglorysize
                    btrans.append(b_step_trans)
                    bmods.append(b_step_mod)
                    #  update beamchoices and probsums in this batch and pad inf score for dead channel to keep dim = beamsize
                    beamchoices[batch] = np.pad(b_step_trans, (0, bndead), 'constant',
                                                constant_values=(0, 0))
                    probsums[batch] = np.pad(prob[b_step_out], (0, bndead), 'constant',
                                             constant_values=(0, np.inf))
                    # build sample at final loop or <eos> predicted
                    for i, (t, m, o) in enumerate(zip(b_step_trans, b_step_mod, b_step_out)):
                        if m == 0 or n_step == maxlen:
                            bmask[i] = 0  # set corresbonding mask to 0
                            sample = []  # declare a new sample list
                            # trace back to step 0 find every predicted word in this channel
                            # k from n_step-1 to 0
                            ii = i  # the ith word will be token corresponding to kth step
                            for k in range(n_step - 1, -1, -1):
                                word = bmods[k][ii]
                                sample.append(word)
                                ii = btrans[k][ii]
                            bsample.append(sample[::-1])
                            bscore.append(probsums[batch][i] / n_step)
                # if all channels of all batches dead
                #  break loop
                if (np.equal(masks, 0)).all():
                    break
                # get probality and update states
                prob_b_k_v = feed_step(beamchoices)[0]
                prob_b_k_v = probsums[:, :, None] - np.log(prob_b_k_v * masks[:, :, None])
                prob_b_kv = prob_b_k_v.reshape([batchsize, beamsize * catglorysize])
            # return samples and score
            return [sample[np.array(score).argmin()] for sample, score in
                    zip(samples_all, scores_all)]

        return gen_sample