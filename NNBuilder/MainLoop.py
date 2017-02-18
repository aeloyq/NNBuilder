# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 19:37:12 2016

@author: aeloyq
"""
#TODO:类化，装饰器加入，处理extension
import timeit
import theano
import numpy as np

def Train(configuration, model_stream, datastream,extension):

    train_model,valid_model,test_model,sample_model,debug_model,model,classifier,n_train_batches,n_valid_batches,n_test_batches,theta,cost=model_stream
    start_time=timeit.default_timer()
    print "\r\nTrainning Model:\r\n"
    sample_data=[datastream[0],datastream[3]]
    batch_size=configuration['batch_size']
    max_epoches=configuration['max_epoches']
    patience=configuration['train_patience']
    valid_freq=configuration['valid_frequence']
    sample_freq=configuration['sample_frequence']
    sample_func=configuration['sample_func']
    n_sample=configuration['n_sample']
    imp_threshold=configuration['improvement_threshold']
    patience_increase=configuration['patience_increase']
    report_iter=configuration['report_per_literation']
    report_epoch=configuration['report_per_epoch']
    iteration_train_index=1
    iteration_total=1
    train_error=train_cost=valid_error=1.
    best_valid_error=1.
    best_iter=0
    epoches=0
    errors=[]
    costs=[]
    first=True
    # Main Loop
    print '        ','Training Start'
    debug_s_time = [timeit.default_timer()]
    while (True):
        train_model(iteration_train_index)
        iteration_train_index += 1
        if iteration_train_index > n_train_batches:

            iteration_train_index=1

            time_debug(debug_s_time)
            break
    while(False):
        # Train model iter by iter
        train_model(iteration_train_index)
        if first:
            first=False;errors.append(train_error);costs.append(train_cost)
        if report_iter:
            print "Iteration Report at Epoch:%d   Iteration:%d     Cost:%.4f"%(epoches,iteration_total,train_cost)
        iteration_train_index+=1
        iteration_total+=1
        if iteration_train_index>n_train_batches:
            train_error=np.mean([test_model(i) for i in range(1,n_test_batches+1)])
            errors.append(train_error)
            costs.append(train_cost)
            iteration_train_index=1
            epoches+=1
            time_debug(debug_s_time)
            if report_epoch:
                print '\r\n'
                print "◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆\r\n\r\nSingle Epoch Done:               \r\n\r\nEpoches:%d  \r\nIterations:%d         \r\nCost:%.4f   \r\nError:%.4f%%\r\n\r\n◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆"%(epoches,iteration_total-1,costs[-1],(train_error*100))
                print '\r\n'
        # Sample
        if iteration_total%sample_freq==0 and sample_freq!=-1:
            if sample_func != None:
                sample_X=sample_data[0][(iteration_train_index-1)*batch_size:(iteration_train_index-1)*batch_size+n_sample].eval()
                sample_Y=sample_data[1][(iteration_train_index-1)*batch_size:(iteration_train_index-1)*batch_size+n_sample].eval()
                sp_pred,sp_cost,sp_error=sample_model(sample_X,sample_Y)
                sample_func(sample_X,sp_pred)
                print '\r\n'
                print "☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆"
                print "Sample Cost:%.4f  Sample Error:%.4f%%  "%(sp_cost,(sp_error*100))
                print "☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆"
                print '\r\n'
        # Stop When Timeout
        if epoches > max_epoches - 1 and max_epoches != -1:
            best_iter = best_iter
            print "⊙Trainning Time out⊙"
            break
        # Stop When Sucess
        if train_error==0:
            if np.mean([test_model(i) for i in range(1,n_test_batches+1)])==0:
                best_iter = iteration_total
                print "\r\n●Trainning Sucess●\r\n"
                break
        #Early Stop
        if (iteration_total-1)%valid_freq==0:
            valid_error=np.mean([valid_model(i) for i in range(1,n_valid_batches+1)])
            if valid_error < best_valid_error:
                if valid_error < best_valid_error *imp_threshold:
                    patience = max(patience, iteration_total * patience_increase)
                best_valid_error = valid_error
                best_iter = iteration_total
                print '\r\n'
                print "★Better Model Detected at Epoches:%d  Iterations:%d  Cost:%.4f  Valid error:%.4f%%★"%(epoches,iteration_total,train_cost,(valid_error*100))
                print '\r\n' 
        if patience < iteration_total:
            print "\r\n▲NO Trainning Patience      Early Stopped▲\r\n"
            break
    print "Finall model:"
    test_error=np.mean([test_model(i) for i in range(1,n_test_batches+1)]);
    for params in theta:
        for param in params:
            print param.get_value()
    print "Trainning finished after epoch:",epoches
    print "Trainning finished at iteration:",iteration_total-1
    print "Best iteration:",best_iter
    print "Finall cost:",costs[-1]
    print "Finall error:%.4f%%"%(errors[-1]*100)
    print "Test error:%.4f%%"%(test_error*100)
    print "Best error:%.4f%%"%(best_valid_error*100)
    end_time=timeit.default_timer()
    time_used=end_time-start_time
    print "Time used:%.2fs"%time_used
    return epoches,errors,costs
def time_debug(st):
    usd=timeit.default_timer()-st[0]
    st[0]=timeit.default_timer()
    print "time used: %.2fs"%usd