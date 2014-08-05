function [dnn,output] = train_dnn(nbClasses,train_data,train_targets,test_data,test_targets,dbn_size)

%%  0. PATHS

%   truncate training set for batching 
    batchsize = 50;
    T = fix(size(train_data,1)/batchsize)*batchsize;
    train_data = train_data(1:T,:);
    train_targets = train_targets(1:T,:);
             
%%  [2] TRAIN DBN

%   OPTIONS DBN ( parameters empirically set, not tuned )
    dbn.sizes        = dbn_size  % 2 hidden layers with size 26 and 13
    opts.learnr_grbm = 0.01;   % learning rate for GRBM
    opts.moment_grbm = 0.01;   % momentum for GRBM 
    opts.alpha       = 0.08;   % learning rate for all upper RBM layers
    opts.momentum    = 0.01;   % momentum for all upper RBM layers
    opts.numepochs   = 20;
    opts.batchsize   = batchsize;
    
%   train DBN   
    dbn = dbnsetup(dbn, train_data, opts);
    %dbn = dbntrain(dbn, train_data, opts);
    
%%  [3] TRAIN DBN-DNN

%   unfold DBN to DNN
    dnn = dbnunfoldtonn(dbn, nbClasses);

%   OPTIONS DNN ( parameters empirically set, not tuned )
    dnn.learningRate    = 0.1;
    dnn.weightPenaltyL2 = 0;
    dnn.momentum        = 0;
    dnn.output          = 'linear'; %'softmax';
    dnn.cost            = 'cro';
    opts.numepochs      = 40;
    opts.batchsize      = 50;
    opts.silent         = 0;
    
%   train DNN   
    dnn = nntrain(dnn, train_data, train_targets, opts);

%%  [4] TEST DBN-DNN   
    [er, bad, output] = nntest(dnn, test_data, test_targets);
    
