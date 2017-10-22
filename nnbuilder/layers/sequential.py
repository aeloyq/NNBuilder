# -*- coding: utf-8 -*-
"""
Created on  四月 26 21:08 2017

@author: aeloyq
"""
import numpy as np
import copy
from utils import *
from basic import *
from simple import *
from roles import *
from ops import *
from nnbuilder.kernel import *


class sequence(linear):
    '''
    abstract layer
    '''

    def __init__(self, **kwargs):
        component.__init__(self, **kwargs)
        self.seq_inscan_param = []
        self.trs_inscan_param = []

    def init_seq_params(self, in_dim, unit_dim, biased=False, inscan=False, unit_name=''):
        self.init_linear_params(in_dim, unit_dim, biased, weight_name='Wt', weight_role=weight, unit_name=unit_name)
        if inscan:
            self.seq_inscan_param.append(unit_name + 'Wt')

    def init_trs_params(self, in_dim, unit_dim, biased=False, inscan=False, unit_name=''):
        self.init_linear_params(in_dim, unit_dim, biased, weight_name='U', weight_role=trans,
                                init_functions=(param_init_functions.orthogonal, param_init_functions.zeros),
                                unit_name=unit_name)
        if inscan:
            self.trs_inscan_param.append(unit_name + 'U')


class seqlinear(sequence, itglinear):
    def __init__(self, units, **kwargs):
        sequence.__init__(self, **kwargs)
        self.seq_units_dim = units.copy()
        self.trs_units_dim = units.copy()

    def seq_layer_dot(self, name, X, role, inputs=None):
        if role is weight:
            pname = name + 'Wt'
        else:
            pname = name + 'U'
        unit_dim = self.shapes[pname][-1]
        shape = X.shape[X.attr.index(batch):]
        inscan = False
        if inputs is not None:
            if inputs['mode'] is scan:
                inscan = True
        if inscan:
            X = self.apply_ops(name=name, tvar=X, ops=dropout, shape=shape, step='apply', inputs=inputs)
        else:
            X = self.apply_ops(name=name, tvar=X, ops=dropout, shape=shape)
        w = self.apply_ops(name=name, tvar=self.params[pname], ops=weightnorm)
        o = T.dot(X, w)
        if inscan:
            o = self.apply_ops(name=name, tvar=o, ops=normalization, unit_dim=unit_dim,
                               pname=pname, step='apply', inputs=inputs)
        else:
            o = self.apply_ops(name=name, tvar=o, ops=normalization, unit_dim=unit_dim)
        return o


class rechidden(seqlinear, hidden):
    unit_name = 'Recurrent'

    def __init__(self, unit, biased, activation=T.tanh, masked=True, out='all', backward=False, **kwargs):
        self.units_dim = OrderedDict()
        self.unit_dim = unit
        self.get_units(unit)
        self.units = self.units_dim.keys()
        seqlinear.__init__(self, units=self.units_dim, biased=biased, **kwargs)
        self.biased = biased
        self.activation = activation
        self.outputs = OrderedDict()
        self.length = None
        self.batch_size = None
        self.masked = masked
        self.out = out
        self.backward = backward
        self.sequences = []
        self.initiate_states = []
        self.non_iters = []
        self.non_sequences = []
        self.outs = ['all', 'final', 'mean']

    def init_all_seq_params(self):
        for name, unit in self.seq_units_dim.items():
            self.init_seq_params(self.in_dim, unit, unit_name=name)

    def init_all_trs_params(self):
        for name, unit in self.trs_units_dim.items():
            self.init_trs_params(self.unit_dim, unit, self.biased, inscan=True, unit_name=name)

    def init_params(self):
        self.init_all_seq_params()
        self.init_all_trs_params()

    def get_units(self, unit):
        self.units_dim[rechidden.unit_name] = unit

    def get_unit_name(self, name):
        return name

    def seq_layer_dot(self, name, X, role, inputs=None):
        return seqlinear.seq_layer_dot(self, self.get_unit_name(name), X, role, inputs)

    def set_input(self, X):
        base.set_input(self, X)
        self.get_length()
        self.get_batch_size()

    def prepare(self, X):
        '''
        | State Before | Initiate State | Fixed Context |
        | 0,1,......-1 | 0,1,........-1 | 0,1,.......-1 |
        | X Xg   Masks | H_ C_      None| ?? DropoutMask|
        :param X: Input Tensor
        :return: list of [state_before, initiate_state, context, n_steps]
        '''
        sequences = self.apply_sequences(X)
        self.sequences = sequences.keys()
        initiate_states = self.apply_initiate_states(X)
        self.initiate_states = initiate_states.keys()
        non_iters = self.apply_non_iters(X)
        self.non_iters = non_iters
        non_iters.extend(self.apply_ops('batchnorm', [], normalization, step='share'))
        non_sequences = self.apply_non_sequences(X)
        self.non_sequences = non_sequences.keys()
        non_sequences.update(
            self.apply_ops('dropout', OrderedDict(), dropout, step='share'))
        n_steps = self.apply_n_steps()
        outputs_info = initiate_states.values() + [None] * len(non_iters)
        return sequences, initiate_states, non_iters, outputs_info, non_sequences, n_steps

    def apply_sequences(self, X, inputs=None):
        sequences = OrderedDict()
        x = self.seq_layer_dot(self.units[0], X, weight, inputs)
        x = self.apply_bias(self.units[0], x, self.biased)
        sequences[self.units[0]] = x
        return sequences

    def apply_initiate_states(self, X):
        initiate_states = OrderedDict()
        initiate_states['h_'] = utils.zeros_initiator(self.batch_size, self.unit_dim)
        return initiate_states

    def apply_non_iters(self, X):
        return []

    def apply_non_sequences(self, X):
        return OrderedDict()

    def apply_n_steps(self):
        return self.length

    def get_batch_size(self):
        self.batch_size = self.input.shape[self.input.attr.index(batch)]

    def get_length(self):
        self.length = self.input.shape[self.input.attr.index(time)]

    def get_out_dim(self):
        self.out_dim = self.unit_dim

    def apply_mask(self, tvar, tvar_, inputs):
        '''
        mask here refers to mask in each step
        so its shape must be batchsize(n_samples)
        assert location of dim refers to batchsize should be -2
        :param tvar: tensor variable which need to be masked
        :param mask: sizeof batchsize
        :return:
        '''
        tvar_ = inputs[tvar_]
        if not self.masked:
            return tvar
        else:
            mask = inputs['m']
            if tvar.ndim == 2:
                return mask[:, None] * tvar + (1. - mask)[:, None] * tvar_
            elif tvar.ndim == 3:
                return mask[None, :, None] * tvar + (1. - mask)[None, :, None] * tvar_
            elif tvar.ndim == 4:
                return mask[None, None, :, None] * tvar + (1. - mask)[None, None, :, None] * tvar_

    def apply_bias(self, name, tvar, biased):
        name = self.get_unit_name(name)
        if biased:
            return tvar + self.params[name + 'Bi']
        else:
            return tvar

    def process_output(self, output, out, mask):
        if out == 'all':
            output = output
        elif out == 'final':
            output = output[-1]
        elif out == 'mean':
            if mask is None:
                output = output.mean(0)
            else:
                if output.ndim == 3:
                    output = (output * mask[:, :, None]).sum(0) / T.cast(
                        mask.sum(0)[:, None], kernel.config.floatX)
                elif output.ndim == 4:
                    output = (output * mask[:, :, :, None]).sum(0) / mask.sum(0)[:, :,
                                                                     None]
                else:
                    raise AssertionError
        else:
            raise AssertionError
        return output

    def apply(self, inputs):
        h = inputs['x'] + self.seq_layer_dot(self.units[0], inputs['h_'], trans, inputs)
        h = self.apply_activation(h, self.activation)
        h = self.apply_mask(h, 'h_', inputs)
        return h

    def pre_apply_wrapper(self, args, sequences, initiate_states, non_iters, non_sequences):
        keys = sequences.keys() + initiate_states.keys() + non_sequences.keys()
        inputs = OrderedDict()
        for idx, key in enumerate(keys):
            inputs[key] = args[idx]
        return inputs

    def su_apply_wrapper(self, out):
        if not isinstance(out, (tuple, list)):
            out = [out]
        else:
            out = list(out)
        out = self.apply_ops('batchnorm', out, normalization, step='add')
        if len(out) == 1:
            out = out[0]
        return out

    def apply_wrapper(self, sequences, initiate_states, non_iters, non_sequences, mode=scan):
        def step(*args):
            inputs = self.pre_apply_wrapper(args, sequences, initiate_states, non_iters, non_sequences)
            inputs['mode'] = mode
            out = self.apply(inputs)
            return self.su_apply_wrapper(out)

        return step

    def pre_feed(self, X):
        sequences, initiate_states, non_iters, outputs_info, non_sequences, n_steps = self.prepare(X)
        if self.masked:
            sequences['m'] = self.data['X_Mask']
        return sequences, initiate_states, non_iters, outputs_info, non_sequences, n_steps

    def su_feed(self, outputs, updates, initiate_states, non_iters):
        output = outputs[0]
        output = self.process_output(output, self.out, self.data['X_Mask'])
        self.updates.update(updates)
        doutputs = OrderedDict()
        for name, out in zip(initiate_states.keys() + non_iters, outputs):
            doutputs[name] = out
        self.apply_ops('batchnorm', None, normalization, outputs=doutputs, step='update')
        return output, doutputs, updates

    def feed(self, X):
        sequences, initiate_states, non_iters, outputs_info, non_sequences, n_steps = self.pre_feed(X)
        step = self.apply_wrapper(sequences, initiate_states, non_iters, non_sequences)
        outputs, updates = kernel.scan(step, sequences=sequences.values(),
                                       initiate_states=initiate_states.values(),
                                       non_iters=[None] * len(non_iters),
                                       non_sequences=non_sequences.values(), n_steps=n_steps,
                                       go_backwards=self.backward)
        return self.su_feed(outputs, updates, initiate_states, non_iters)

    def build(self, ops_option):
        self.set_input(self.pre_layer.output)
        self.initiate(ops_option)
        self.output, self.outputs, update = self.feed(self.input)
        return self.output


class attention(fwdlinear):
    unit_name = ['AttentionContext', 'AttentionRecurrent', 'AttentionCombine']

    def __init__(self, attunit, ctxunit, recunit, attbiased, attgate=T.tanh, **kwargs):
        base.__init__(self, **kwargs)
        self.att_unit = attunit
        self.attgate = attgate
        self.att_units_dim = OrderedDict()
        self.get_att_units()
        self.rec_dim = recunit
        self.context_dim = ctxunit
        self.attbiased = attbiased
        self.att_inscan_param = []

    def get_att_units(self):
        self.att_inscan_unit = [attention.unit_name[1], attention.unit_name[2]]
        self.att_units_dim[attention.unit_name[2]] = (self.att_unit, 1), False
        if self.attgate not in [T.glu, T.gtu]:
            self.att_units_dim[attention.unit_name[1]] = (self.rec_dim, unit), False
            self.att_units_dim[attention.unit_name[0]] = (self.context_dim, unit), self.biased
        else:
            self.att_units_dim[attention.unit_name[1]] = (self.rec_dim, unit * 2), False
            self.att_units_dim[attention.unit_name[0]] = (self.context_dim, unit * 2), self.biased

    def init_all_att_params(self):
        for unit_name, unit_dim in self.att_units_dim.items():
            if unit_name in self.att_inscan_unit:
                self.init_att_params(unit_dim[0][0], unit_dim[0][1], unit_dim[1], inscan=True, unit_name=unit_name)
            else:
                self.init_att_params(unit_dim[0][0], unit_dim[0][1], unit_dim[1], unit_name=unit_name)

    def init_att_params(self, in_dim, unit_dim, biased, inscan=False, unit_name=''):
        self.allocate(param_init_functions.uniform, unit_name + 'Wt', weight, [in_dim, unit_dim])
        if biased:
            self.allocate(param_init_functions.zeros, unit_name + 'Bi', bias, [unit_dim])
        if inscan:
            self.att_inscan_param.append(unit_name + 'Wt')

    def att_dot(self, name, X, inputs=None):
        pname = name + 'Wt'
        unit_dim = self.shapes[pname][-1]
        if inputs is not None:
            X = self.apply_ops(name=name, tvar=X, ops=dropout)
        else:
            X = self.apply_ops(name=name, tvar=X, ops=dropout)
        w = self.apply_ops(name=name, tvar=self.params[pname], ops=weightnorm)
        o = T.layer_dot(X, w)
        if inputs is not None:
            o = self.apply_ops(name=name, tvar=o, ops=normalization, unit_dim=unit_dim,
                               pname=pname, step='apply', inputs=inputs)
        else:
            o = self.apply_ops(name=name, tvar=o, ops=normalization, unit_dim=unit_dim,
                               pname=pname, step='apply')
        if name + 'Bi' in self.params:
            o = o + self.params[name + 'Bi']
        return o

    def apply_pctx(self, Context, inputs=None):
        return self.att_dot(name=attention.unit_name[0], X=Context, inputs=inputs)

    def apply_rctx(self, Recurrent, inputs=None):
        return self.att_dot(name=attention.unit_name[1], X=Recurrent, inputs=inputs)

    def apply_att(self, PreAttention, inputs=None):
        return self.att_dot(name=attention.unit_name[2], X=PreAttention, inputs=inputs)

    def apply_attgate(self, X):
        return self.attgate(X)

    def apply_attention(self, pctx, rctx, mask):
        patt = self.apply_attgate(pctx + rctx)
        att = self.apply_att(patt)
        if self.att_units_dim[attention.unit_name[2]][0][1] == 1:
            att = T.reshape(att, att.shape[:-1], att.attr[:-1])
        eij = T.exp(att - att.max(0, keepdims=True))
        if self.att_units_dim[attention.unit_name[2]][0][1] == 1:
            if eij.ndim == 2:
                eij = eij * mask
            elif eij.ndim == 3:
                eij = eij * mask[:, :, None]
            elif eij.ndim == 4:
                eij = eij * mask[:, :, :, None]
        else:
            if eij.ndim == 3:
                eij = eij * mask[:, :, None]
            elif eij.ndim == 4:
                eij = eij * mask[:, :, :, None]
        return eij / eij.sum(0, keepdims=True)


class recoutput(rechidden, lookuptable, output):
    emb_name = 'TargetWord'
    initiator_name = 'BeginOfSentence'

    def __init__(self, unit, ctxunit, emb, vocab, biased, activation=T.tanh, initiator='biased_mean', masked=True,
                 **kwargs):
        base.__init__(**kwargs)
        super(recoutput, self).__init__(unit, biased, activation, masked=masked, **kwargs)
        self.emb_unit = {recoutput.emb_name: emb}
        self.initiator_unit = {recoutput.initiator_name: unit}
        self.vocab_dim = vocab
        self.initiator = initiator
        self.initiators = ['none', 'mean', 'final', 'biased_mean', 'biased_final']
        self.context = None
        self.context_dim = ctxunit
        self.is_first_recoutput_layer = False

    def init_initiator_param(self):
        self.allocate(param_init_functions.uniform, self.initiator_unit.keys()[0] + 'Wt', weight,
                      [self.context_dim, self.initiator_unit.keys()[1]])
        if self.initiator in ['biased_mean', 'biased_final']:
            self.allocate(param_init_functions.zeros, self.initiator_unit.keys()[0] + 'Bi', bias,
                          [self.initiator_unit.keys()[1]])

    def initiator_dot(self, X, name):
        if self.initiator in ['mean', 'final', 'biased_mean', 'biased_final']:
            if self.initiator in ['final', 'biased_final']:
                context = self.process_output(X, 'final', self.data['X_Mask'])
            else:
                context = self.process_output(X, 'mean', self.data['X_Mask'])
            unit_dim = self.shapes[name + 'Wt'][-1]
            context = self.apply_ops(name=name, tvar=context, ops=dropout)
            w = self.apply_ops(name=name, tvar=self.params[name + 'Wt'], ops=weightnorm)
            o = T.layer_dot(context, w)
            o = self.apply_ops(name=name, tvar=o, ops=normalization, unit_dim=unit_dim)
            if self.initiator in ['biased_mean', 'biased_final']:
                o = o + self.params[name + 'Bi']
        else:
            o = utils.zeros_initiator(self.batch_size, self.unit_dim)
        return o

    def init_params(self):
        super(recoutput, self).init_params()
        if self.initiator in ['mean', 'final', 'biased_mean', 'biased_final']:
            self.init_initiator_param()
        if self.is_first_recoutput_layer:
            self.init_lookuptable_params(self.vocab_dim, self.emb_unit[0], self.emb_unit[1])

    def get_batch_size(self):
        self.batch_size = self.data['Y'].shape[self.data['Y'].attr.index(batch)]

    def get_length(self):
        self.length = self.data['Y'].shape[self.data['Y'].attr.index(time)]

    def apply_initiate_states(self, X):
        super(recoutput, self).apply_initiate_states(X)
        self.initiator_state = self.initiator_dot(self.context, recoutput.initiator_name)
        self.initiate_states['h_'] = self.initiator_state

    def pre_feed(self, X):
        super(recoutput, self).pre_feed(X)
        if self.masked:
            self.sequences['m'] = self.data['Y_Mask']

    def feed(self, X):
        if self.is_first_recoutput_layer:
            groundtruth = self.layer_lookup(recoutput.emb_name, self.data['Y'])
            groundtruth_shift = T.zeros([self.data['Y'].shape + [self.emb_unit.values()[0]]], attr=[time, batch, unit])
            groundtruth_shift[1:] = groundtruth[:-1]
            self.groundtruth = groundtruth_shift
        return super(recoutput, self).feed(self.groundtruth)

    def pre_build(self, name, prelayer, data):
        self.set_name(name)
        self.set_pre_layer(prelayer)
        self.set_data(data)
        self.set_in_dim(prelayer.out_dim)
        self.get_out_dim()
        self.first_recoutput_layer = self
        self.last_rechidden_layer = self.pre_layer
        while isinstance(self.first_recoutput_layer.pre_layer, recoutput):
            self.first_recoutput_layer = self.first_recoutput_layer.pre_layer
            self.last_rechidden_layer = self.last_rechidden_layer.pre_layer
        self.first_recoutput_layer.is_first_recoutput_layer = True
        self.context = self.first_recoutput_layer.input
        self.init_params()


class conditional(attention, recoutput):
    def __init__(self, unit, att_unit, ctxunit, biased, attbiased=True, activation=T.tanh, attgate=T.tanh,
                 masked=True,
                 **kwargs):
        base.__init__(**kwargs)
        recoutput.__init__(self, unit, biased, activation, masked=masked, **kwargs)
        attention.__init__(self, att_unit, ctxunit, unit, attbiased, attgate)
        self.cond = ''
        self.cond_units_dim = OrderedDict()
        for unit_name, unit_dim in self.units_dim.items():
            self.cond_units_dim['Cond' + unit_name] = unit_dim
        self.seq_units_dim.update(self.cond_units_dim)
        self.trs_units_dim.update(self.cond_units_dim)
        self.attention_context = None

    def init_all_seq_params(self):
        for name, unit in self.seq_units_dim.items():
            if name not in self.cond_units_dim:
                self.init_seq_params(self.in_dim, unit, unit_name=name)
            else:
                self.init_seq_params(self.in_dim, unit, inscan=True, unit_name=name)

    def init_all_att_params(self):
        for unit_name, unit_dim in self.att_units_dim.items():
            self.init_att_params(unit_dim[0][0], unit_dim[0][1], unit_dim[1], unit_name=unit_name)

    def init_params(self):
        super(conditional, self).init_params()
        self.init_all_att_params()
        self.inscan_param.append(self.att_inscan_param)

    def get_unit_name(self, name):
        name = super(conditional, self).get_unit_name(name)
        return self.cond + name

    def apply_non_sequences(self, X):
        super(conditional, self).apply_non_sequences(X)
        pctx = self.apply_pctx(self.context)
        self.non_sequences['ctx'] = self.context
        self.non_sequences['pctx'] = pctx

    def apply(self, inputs):
        self.cond = ''
        first_recurrent_sublayer = super(conditional, self).apply(inputs)
        self.cond = 'Cond'
        for sequence_name in self.sequences.keys():
            if sequence_name != 'm':
                pass
        if not isinstance(first_recurrent_sublayer, (tuple, list)):
            first_recurrent_sublayer = tuple(first_recurrent_sublayer)
        for i, initiate_state_name in enumerate(self.initiate_states.keys()):
            inputs[initiate_state_name] = first_recurrent_sublayer[i]
        second_recurrent_sublayer = super(conditional, self).apply(inputs)
        self.cond = ''
        return second_recurrent_sublayer


class beamsearch(base):
    def __init__(self, **kwargs):
        base.__init__(self, **kwargs)

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
        beamchoice = kernel.placeholder('BeamChoice', [batch, search], kernel.config.catX)
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
        fn_contexts = kernel.compile([self.data['X'], self.data['X_Mask']], updates=contexts_updates, strict=False)
        chosen_initstates = OrderedDict()
        for k, v in initstates.items():
            raw_shape = v.shape[2:]
            raw_attr = v.attr[2:]
            raw_flatten = v.reshape([batchsize * beamsize] + raw_shape, [None] + raw_attr)
            chosen_initstate = raw_flatten[beamchoice.flatten()].reshape([batchsize, beamsize] + raw_shape,
                                                                         [batch, search] + raw_attr)
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


class emitter(itglinear, itglookup, output):
    unit_name = ['EmitterRecurrent', 'EmitterGlimpse', 'EmitterPeek', 'EmitterWord']

    def __init__(self, unit, ctxunit, emb, vocab, biased=True, tie=False, activation=T.tanh, **kwargs):
        base.__init__(self, **kwargs)
        output.__init__(self, **kwargs)
        self.last_conditional_recurrent_layer = None
        self.unit_dim = unit
        self.units_dim = OrderedDict()
        self.context_dim = ctxunit
        self.emb_dim = emb
        self.vocab_dim = vocab
        self.biased = biased
        self.tie = tie
        self.biased = biased
        self.activation = activation
        self.ewd_units_dim = OrderedDict()
        self.emt_units_dim = OrderedDict()
        self.get_units()
        self.units_dim.update(self.emt_units_dim)
        self.units_dim.update(self.ewd_units_dim)

    def get_units(self):
        if self.activation not in [T.glu, T.gtu, T.maxout]:
            self.emt_units_dim[emitter.unit_name[0]] = (self.in_dim, self.unit_dim), True
            self.emt_units_dim[emitter.unit_name[1]] = (self.context_dim, self.unit_dim), False
            self.emt_units_dim[emitter.unit_name[2]] = (self.emb_dim, self.unit_dim), self.biased
        else:
            self.emt_units_dim[emitter.unit_name[0]] = (self.in_dim, self.unit_dim * 2), True
            self.emt_units_dim[emitter.unit_name[1]] = (self.context_dim, self.unit_dim * 2), False
            self.emt_units_dim[emitter.unit_name[2]] = (self.emb_dim, self.unit_dim * 2), self.biased
        self.ewd_units_dim[emitter.unit_name[3]] = (self.unit_dim, self.vocab_dim), False

    def init_all_emt_params(self):
        for unit_name, unit_dim in self.emt_units_dim.items():
            self.init_emt_params(unit_dim[0][0], unit_dim[0][1], unit_dim[1], unit_name=unit_name)

    def init_emt_params(self, in_dim, unit_dim, biased, unit_name=''):
        self.allocate(param_init_functions.uniform, unit_name + 'Wt', weight, [in_dim, unit_dim])
        if biased:
            self.allocate(param_init_functions.zeros, unit_name + 'Bi', bias, [unit_dim])

    def init_all_emd_params(self):
        if not self.tie:
            self.init_all_lookup_params(self.ewd_units_dim)
        else:
            pname = self.ewd_units_dim.keys()[0] + 'Lookuptable'
            self.params[pname] = self.first_recoutput_layer.params[pname]

    def init_params(self):
        self.init_all_emt_params()
        self.init_all_emd_params()

    def apply_activation(self, tvar):
        if self.activation is None:
            return tvar
        else:
            return self.activation(tvar)

    def layer_dot(self, name, X):
        unit_dim = self.shapes[name + 'Wt'][-1]
        X = self.apply_ops(name=name, tvar=X, ops=dropout)
        w = self.apply_ops(name=name, tvar=self.params[name + 'Wt'], ops=weightnorm)
        o = T.layer_dot(X, w)
        o = self.apply_ops(name=name, tvar=o, ops=normalization, unit_dim=unit_dim)
        if name + 'Bi' in self.params:
            o = o + self.params[name + 'Bi']
        return o

    def apply(self, X):
        s = X
        actx = self.last_conditional_recurrent_layer.attention_context
        gdth = self.first_recoutput_layer.groundtruth
        preprobability = self.layer_dot(emitter.unit_name[0], s)
        preprobability += self.layer_dot(emitter.unit_name[1], actx)
        preprobability += self.layer_dot(emitter.unit_name[2], gdth)
        preprobability = self.apply_activation(preprobability)
        probability = self.layer_dot(emitter.unit_name[3], preprobability)
        probability = T.softmax(probability)
        return probability

    def apply_one_step(self):
        X = T.zeros([self.data['Y'].shape[0], self.data['Y'].shape[1]], [time, batch])
        for layer in self.recoutput_layers:
            inputs = OrderedDict()
            seq = self.first_recoutput_layer.apply_sequences(X)
            masked = self.first_recoutput_layer.masked
            inputs.update(seq)
            if masked:
                inputs['m'] = self.data['Y_Mask']
            self.first_recoutput_layer.apply_initiate_states(X)
            ins = self.first_recoutput_layer.initiate_states
            inputs.update(ins)
            self.first_recoutput_layer.apply_non_sequences(X)
            nseq = self.first_recoutput_layer.non_sequences
            inputs.update(nseq)
            X = layer.apply(inputs)[0]
        probability = self.apply(X)
        words = T.argmax(probability, 1)
        E = self.first_recoutput_layer.layer_lookup(recoutput.emb_name, words)
        return E

    def apply_loss(self, Y_True):
        '''
        get the cost of the model
        :param Y_True:
            the label of the model which used to evaluate the cost function(loss function)
        :return: tensor variable
            the cost of the model
        '''
        probability = self.output
        maxlen = probability.shape[0]
        batchsize = probability.shape[1]
        loss = T.log_likelihood(
            probability.reshape([maxlen * batchsize, self.vocab_dim], [None, unit]),
            Y_True.flatten()).reshape([maxlen, batchsize]) * self.data['Y_Mask']
        return loss.sum(0).mean(0)

    def apply_sample(self):
        '''
        get the predict of the model
        :return: tensor variable
            the predict of model
        '''
        return T.round(self.output)

    def apply_sample_loss(self, Y_True):
        probability = self.sample_probability
        maxlen = probability.shape[0]
        batchsize = probability.shape[1]
        loss = T.log_likelihood(
            probability.reshape([maxlen * batchsize, self.vocab_dim], [None, unit]),
            Y_True.flatten()).reshape([maxlen, batchsize]) * self.data['Y_Mask']
        return loss.sum(0).mean(0)

    def apply_sample_error(self, Y_True):
        return T.mean(T.neq(Y_True, self.sample))

    def apply_predict(self):
        return self.apply_sample()

    def pre_build(self, name, prelayer, data):
        self.set_name(name)
        self.set_pre_layer(prelayer)
        self.set_data(data)
        self.set_in_dim(prelayer.out_dim)
        self.get_out_dim()
        self.recoutput_layers = [self]
        self.first_recoutput_layer = self
        self.last_rechidden_layer = self.pre_layer
        while isinstance(self.first_recoutput_layer.pre_layer, recoutput):
            self.recoutput_layers = [self.first_recoutput_layer.pre_layer] + self.recoutput_layers
            if isinstance(self.first_recoutput_layer.pre_layer, conditional):
                self.last_conditional_recurrent_layer = self.first_recoutput_layer.pre_layer
            self.first_recoutput_layer = self.first_recoutput_layer.pre_layer
            self.last_rechidden_layer = self.last_rechidden_layer.pre_layer
        self.first_recoutput_layer.is_first_recoutput_layer = True
        self.context = self.first_recoutput_layer.input
        self.init_params()


class deeptransition(rechidden):
    def __init__(self, transition_depth, masked):
        base.__init__(self)
        self.transition_depth = transition_depth
        self.masked = masked
        self.deeploc = ''
        self.dt_units_dim = OrderedDict()
        for i in range(2, transition_depth + 1):
            for k, v in self.trs_units_dim.items():
                self.dt_units_dim[k + 'DT' + str(i)] = v
        self.trs_units_dim.update(self.dt_units_dim)
        units_dim = OrderedDict()
        units_dim.update(self.seq_units_dim)
        units_dim.update(self.trs_units_dim)
        self.units_dim = units_dim

    def get_unit_name(self, name):
        name = super(deeptransition, self).get_unit_name(name)
        return name + self.deeploc

    def apply(self, inputs):
        self.deeploc = ''
        last_output = super(deeptransition, self).apply(inputs)
        for i in range(2, self.transition_depth + 1):
            self.deeploc = 'DT' + str(i)
            for sequence_name in self.sequences:
                if sequence_name != 'm':
                    inputs[sequence_name] = self.params[self.get_unit_name(sequence_name) + 'Bi']
            if not isinstance(last_output, (tuple, list)):
                last_output = tuple([last_output])
            for i, initiate_state_name in enumerate(self.initiate_states):
                inputs[initiate_state_name] = last_output[i]
            last_output = super(deeptransition, self).apply(inputs)
        self.deeploc = ''
        return last_output


class altnative(rechidden):
    pass


class bidirectional(rechidden):
    def __init__(self):
        base.__init__(self)
        self.direction = 'Fwd'
        bi_seq_units_dim = OrderedDict()
        for k, v in self.seq_units_dim.items():
            bi_seq_units_dim['Fwd' + k] = v
            bi_seq_units_dim['Bwd' + k] = v
        bi_trs_units_dim = OrderedDict()
        for k, v in self.trs_units_dim.items():
            bi_trs_units_dim['Fwd' + k] = v
            bi_trs_units_dim['Bwd' + k] = v
        self.seq_units_dim = bi_seq_units_dim
        self.trs_units_dim = bi_trs_units_dim
        units_dim = OrderedDict()
        units_dim.update(self.seq_units_dim)
        units_dim.update(self.trs_units_dim)
        self.units_dim = units_dim

    def get_out_dim(self):
        self.out_dim = self.unit_dim * 2

    def get_unit_name(self, name):
        name = super(bidirectional, self).get_unit_name(name)
        return self.direction + name

    def feed(self, X):
        self.direction = 'Fwd'
        sequences, initiate_states, non_iters, outputs_info, non_sequences, n_steps = self.pre_feed(X)
        fwdstep = self.apply_wrapper(sequences, initiate_states, non_iters, non_sequences)
        fwdoutputs, fwdupdates = kernel.scan(fwdstep, sequences=sequences.values(),
                                             initiate_states=initiate_states.values(),
                                             non_iters=[None] * len(self.non_iters),
                                             non_sequences=non_sequences.values(), n_steps=n_steps,
                                             go_backwards=False)
        self.direction = 'Bwd'
        sequences, initiate_states, non_iters, outputs_info, non_sequences, n_steps = self.pre_feed(X)
        bwdstep = self.apply_wrapper(sequences, initiate_states, non_iters, non_sequences)
        bwdoutputs, bwdupdates = kernel.scan(bwdstep, sequences=sequences.values(),
                                             initiate_states=initiate_states.values(),
                                             non_iters=[None] * len(self.non_iters),
                                             non_sequences=non_sequences.values(), n_steps=n_steps,
                                             go_backwards=True)
        fwdoutput, fwddoutputs, fwdupdates = self.su_feed(fwdoutputs, fwdupdates, initiate_states, non_iters)
        bwdoutput, bwddoutputs, bwdupdates = self.su_feed(bwdoutputs, bwdupdates, initiate_states, non_iters)
        output = T.concatenate([fwdoutput, bwdoutput[::-1]], axis=fwdoutput.ndim - 1)
        updates = fwdupdates
        updates.update(bwdupdates)
        doutputs = fwddoutputs
        doutputs.update(bwddoutputs)
        return output, doutputs, updates


class rnn(rechidden, entity):
    def __init__(self, unit, biased=True, activation=T.tanh, masked=True, out='all', backward=False, **kwargs):
        rechidden.__init__(self, unit, biased, activation, masked, out, backward, **kwargs)


class gru(rechidden, entity):
    unit_name = ['Gate', 'Recurrent']

    def __init__(self, unit, biased=True, activation=T.tanh, masked=True, out='all', backward=False, **kwargs):
        rechidden.__init__(self, unit, biased, activation, masked, out, backward, **kwargs)

    def get_units(self, unit):
        self.units_dim[gru.unit_name[0]] = unit * 2
        self.units_dim[gru.unit_name[1]] = unit

    def apply_sequences(self, X, inputs=None):
        sequences = OrderedDict()
        xg = self.seq_layer_dot(self.units[0], X, weight, inputs)
        x = self.seq_layer_dot(self.units[1], X, weight, inputs)
        xg = self.apply_bias(self.units[0], xg, self.biased)
        x = self.apply_bias(self.units[1], x, self.biased)
        sequences[self.units[0]] = xg
        sequences[self.units[1]] = x
        return sequences

    def apply(self, inputs):
        gate = inputs[self.units[0]] + self.seq_layer_dot(gru.unit_name[0], inputs['h_'], trans, inputs)
        gate = T.sigmoid(gate)
        z_gate = utils.slice(gate, 0, self.unit_dim)
        r_gate = utils.slice(gate, 1, self.unit_dim)
        h_c = self.apply_activation(
            inputs[self.units[1]] + r_gate * self.seq_layer_dot(gru.unit_name[1], inputs['h_'], trans, inputs),
            self.activation)
        h = (1 - z_gate) * inputs['h_'] + z_gate * h_c
        h = self.apply_mask(h, 'h_', inputs)
        return h


class lstm(rechidden, entity):
    unit_name = 'Recurrent'

    def __init__(self, unit, biased=True, activation=T.tanh, masked=True, out='all', backward=False, **kwargs):
        rechidden.__init__(self, unit, biased, activation, masked, out, backward, **kwargs)

    def get_units(self, unit):
        self.units_dim[lstm.unit_name] = unit * 4

    def apply_initiate_states(self, X):
        initiate_states = OrderedDict()
        initiate_states['h_'] = utils.zeros_initiator(self.batch_size, self.unit_dim)
        initiate_states['c_'] = utils.zeros_initiator(self.batch_size, self.unit_dim)
        return initiate_states

    def apply(self, inputs):
        gate = inputs[self.units[0]] + self.seq_layer_dot(self.units[0], inputs['h_'], trans, inputs)
        gate = T.sigmoid(gate)
        f_gate = T.sigmoid(utils.slice(gate, 0, self.unit_dim))
        i_gate = T.sigmoid(utils.slice(gate, 1, self.unit_dim))
        o_gate = T.sigmoid(utils.slice(gate, 2, self.unit_dim))
        cell = self.apply_activation(utils.slice(gate, 3, self.unit_dim), T.tanh)
        c = f_gate * inputs['c_'] + i_gate * cell
        c = self.apply_mask(c, 'c_', inputs)
        h = o_gate * self.apply_activation(c, self.activation)
        h = self.apply_mask(h, 'h_', inputs)
        return h, c


class rnnout(rnn, recoutput, entity):
    def __init__(self, unit, biased=True, activation=T.tanh, masked=True, **kwargs):
        rnn.__init__(self, unit, biased, activation, masked, **kwargs)


class gruout(gru, recoutput, entity):
    def __init__(self, unit, biased=True, activation=T.tanh, masked=True, **kwargs):
        gru.__init__(self, unit, biased, activation, masked, **kwargs)


class lstmout(lstm, recoutput, entity):
    def __init__(self, unit, biased=True, activation=T.tanh, masked=True, **kwargs):
        lstm.__init__(self, unit, biased, activation, masked, **kwargs)


class birnn(bidirectional, rnn):
    def __init__(self, unit, biased=True, activation=T.tanh, masked=True, out='all', backward=False,
                 **kwargs):
        rnn.__init__(self, unit, biased, activation, masked, out, backward, **kwargs)
        bidirectional.__init__(self)


class dtrnn(deeptransition, rnn):
    def __init__(self, unit, transition_depth=2, biased=True, activation=T.tanh, masked=True, out='all', backward=False,
                 **kwargs):
        rnn.__init__(self, unit, biased, activation, masked, out, backward, **kwargs)
        deeptransition.__init__(self, transition_depth, masked)


class bidtrnn(bidirectional, deeptransition, rnn):
    def __init__(self, unit, transition_depth=2, biased=True, activation=T.tanh, masked=True, out='all', backward=False,
                 **kwargs):
        rnn.__init__(self, unit, biased, activation, masked, out, backward, **kwargs)
        deeptransition.__init__(self, transition_depth, masked)
        bidirectional.__init__(self)


class bigru(bidirectional, gru):
    def __init__(self, unit, biased=True, activation=T.tanh, masked=True, out='all', backward=False,
                 **kwargs):
        gru.__init__(self, unit, biased, activation, masked, out, backward, **kwargs)
        bidirectional.__init__(self)


class dtgru(deeptransition, gru):
    def __init__(self, unit, transition_depth=2, biased=True, activation=T.tanh, masked=True, out='all', backward=False,
                 **kwargs):
        gru.__init__(self, unit, biased, activation, masked, out, backward, **kwargs)
        deeptransition.__init__(self, transition_depth, masked)


class bidtgru(bidirectional, deeptransition, gru):
    def __init__(self, unit, transition_depth=2, biased=True, activation=T.tanh, masked=True, out='all', backward=False,
                 **kwargs):
        gru.__init__(self, unit, biased, activation, masked, out, backward, **kwargs)
        deeptransition.__init__(self, transition_depth, masked)
        bidirectional.__init__(self)


class bilstm(bidirectional, lstm):
    def __init__(self, unit, biased=True, activation=T.tanh, masked=True, out='all', backward=False,
                 **kwargs):
        lstm.__init__(self, unit, biased, activation, masked, out, backward, **kwargs)
        bidirectional.__init__(self)


class dtlstm(deeptransition, lstm):
    def __init__(self, unit, transition_depth=2, biased=True, activation=T.tanh, masked=True, out='all', backward=False,
                 **kwargs):
        lstm.__init__(self, unit, biased, activation, masked, out, backward, **kwargs)
        deeptransition.__init__(self, transition_depth, masked)


class bidtlstm(bidirectional, deeptransition, lstm):
    def __init__(self, unit, transition_depth=2, biased=True, activation=T.tanh, masked=True, out='all', backward=False,
                 **kwargs):
        lstm.__init__(self, unit, biased, activation, masked, out, backward, **kwargs)
        deeptransition.__init__(self, transition_depth, masked)
        bidirectional.__init__(self)
