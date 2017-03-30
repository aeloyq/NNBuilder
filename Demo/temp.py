for i in range(10):
    data = prepare_data(train_X, train_Y, train_minibatches[i][1])
    train_result = train_model(*data)
    dict['train_result'] = 1
    train_cost = 1
    dict['iteration_total'] += 1
    for ex in extension_instance:   ex.after_iteration()
dict['stop'] = True