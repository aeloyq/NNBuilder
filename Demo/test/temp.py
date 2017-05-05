import theano
import numpy
import theano.tensor as tensor
def build_full_sampler(tparams, options, use_noise, trng, greedy=False):

    x_mask = None

    x, ctx = build_encoder(tparams, options, trng, use_noise, x_mask, sampling=True)
    n_samples = x.shape[2]

    ctx_mean = ctx.mean(0)

    init_state = get_layer_constr('ff')(tparams, ctx_mean, options,
                                    prefix='ff_state', activ='tanh')



    k = tensor.iscalar("k")
    k.tag.test_value = 12
    init_w = tensor.alloc(numpy.int64(-1), k*n_samples)

    ctx = tensor.tile(ctx, [k, 1])

    init_state = tensor.tile(init_state, [k, 1])

    # projected context
    assert ctx.ndim == 3, 'Context must be 3-d: #annotation x #sample x dim'
    pctx_ = tensor.dot(ctx*ctx_dropout_d[0], tparams[pp('decoder', 'Wc_att')]) +\
        tparams[pp('decoder', 'b_att')]

    def decoder_step(y, init_state, ctx, pctx_, target_dropout, emb_dropout, rec_dropout, ctx_dropout, *shared_vars):

        # if it's the first word, emb should be all zero and it is indicated by -1
        decoder_embedding_suffix = '' if options['tie_encoder_decoder_embeddings'] else '_dec'
        emb = get_layer_constr('embedding')(tparams, y, suffix=decoder_embedding_suffix)
        emb = tensor.switch(y[:, None] < 0,
                            tensor.zeros((1, options['dim_word'])),
                            emb)
        emb *= target_dropout

        # apply one step of conditional gru with attention
        proj = get_layer_constr('gru_cond')(tparams, emb, options,
                                                prefix='decoder',
                                                mask=None,
                                                context=ctx,
                                                context_mask=x_mask,
                                                pctx_=pctx_,
                                                one_step=True,
                                                init_state=init_state,
                                                emb_dropout=emb_dropout,
                                                ctx_dropout=ctx_dropout,
                                                rec_dropout=rec_dropout,
                                                shared_vars=shared_vars,
                                                profile=profile)
        # get the next hidden state
        next_state = proj[0]

        # get the weighted averages of context for this target word y
        ctxs = proj[1]

        # alignment matrix (attention model)
        dec_alphas = proj[2]

        if options['use_dropout'] and options['model_version'] < 0.1:
            next_state_up = next_state * retain_probability_hidden
            emb *= retain_probability_emb
            ctxs *= retain_probability_hidden
        else:
            next_state_up = next_state

        logit_lstm = get_layer_constr('ff')(tparams, next_state_up, options,
                                        prefix='ff_logit_lstm', activ='linear')
        logit_prev = get_layer_constr('ff')(tparams, emb, options,
                                        prefix='ff_logit_prev', activ='linear')
        logit_ctx = get_layer_constr('ff')(tparams, ctxs, options,
                                    prefix='ff_logit_ctx', activ='linear')
        logit = tensor.tanh(logit_lstm+logit_prev+logit_ctx)

        if options['use_dropout'] and options['model_version'] < 0.1:
            logit *= retain_probability_hidden

        logit_W = tparams['Wemb' + decoder_embedding_suffix].T if options['tie_decoder_embeddings'] else None
        logit = get_layer_constr('ff')(tparams, logit, options,
                                prefix='ff_logit', activ='linear', W=logit_W)

        # compute the softmax probability
        next_probs = tensor.nnet.softmax(logit)

        if greedy:
            next_sample = next_probs.argmax(1)
        else:
            # sample from softmax distribution to get the sample
            next_sample = trng.multinomial(pvals=next_probs).argmax(1)

        # do not produce words after EOS
        next_sample = tensor.switch(
                      tensor.eq(y,0),
                      0,
                      next_sample)

        return [next_sample, next_state, next_probs[:, next_sample].diagonal()], \
               theano.scan_module.until(tensor.all(tensor.eq(next_sample, 0))) # stop when all outputs are 0 (EOS)