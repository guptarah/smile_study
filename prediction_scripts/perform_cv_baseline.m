function [combo_dev,combo_test] = perform_cv_baseline(dbn_size)

%features_dir='../feature_preparation/fpoint_norm/RA*';
features_dir='../feature_preparation/to_use_features/RA*';
feature_files = dir(features_dir);

feature_store=cell(10,1);
counter=0;
for file = feature_files'
	counter = counter + 1;
%	to_load_file = strcat('../feature_preparation/fpoint_norm/',file.name);
	to_load_file = strcat('../feature_preparation//to_use_features/',file.name);
	feature_store{counter} = load(to_load_file);
end

lables_dir='../../ratings/RA*';
lables_file=dir(lables_dir);
lables_store=cell(10,1);
counter = 0;
for file = lables_file'
	counter = counter + 1;
	to_load_file = strcat('../../ratings/',file.name);
	lables_store{counter} = load(to_load_file);	
end

addpath(genpath('/home/rcf-proj/mv/guptarah/DeepLearnToolbox/'));
combo_dev=[];
combo_test=[];
for split_id = 1:10
	disp(split_id);
	% preparing test, dev and train partitions
	train_data = [];
	train_lables = [];
	for i = 1:10
		if i == split_id
			disp('test partition');
			disp(i);
			test_data = feature_store{i};
			test_lables = lables_store{i};
		elseif mod(i,10) == mod(split_id + 1,10)
			disp('dev partition');
			disp(i);
			dev_data = feature_store{i};
			dev_lables = lables_store{i};
		else
			train_data = [train_data;feature_store{i}];
			train_lables = [train_lables;lables_store{i}];
		end 
	end

	% normalize the lables and compute their average 
	[norm_train_lables,lab_mu,lab_sigma] = zscore(train_lables);
	target_train = sum(norm_train_lables,2)/size(norm_train_lables,2);
	norm_test_lables = (test_lables - repmat(lab_mu,size(test_lables,1),1))./(repmat(lab_sigma,size(test_lables,1),1));
	target_test = sum(norm_test_lables,2)/size(norm_train_lables,2);
	norm_dev_lables = (dev_lables - repmat(lab_mu,size(dev_lables,1),1))./(repmat(lab_sigma,size(dev_lables,1),1));
        target_dev = sum(norm_dev_lables,2)/size(norm_train_lables,2);

	% normalize the features
	[norm_train_feat,feat_mu,feat_sigma] = zscore(train_data);
	norm_test_feat = (test_data - repmat(feat_mu,size(test_data,1),1))./(repmat(feat_sigma,size(test_data,1),1));
	norm_dev_feat = (dev_data - repmat(feat_mu,size(dev_data,1),1))./(repmat(feat_sigma,size(dev_data,1),1));

	% get results on dev set
	nbClasses = 1;
%	dbn_size = [5];
	[dnn,dev_output] = train_dnn(nbClasses,norm_train_feat,target_train,norm_dev_feat,target_dev,dbn_size);
	dev_corr = corrcoef(dev_output,target_dev);
	combo_dev = [combo_dev;dev_output target_dev];
	disp('dev_corr');
	disp(dev_corr(1,2));

	% get results on the test set
	[~, ~, test_output] = nntest(dnn, norm_test_feat, target_test);	
	test_corr = corrcoef(test_output,target_test);
	combo_test = [combo_test;test_output target_test];
	disp('test corr');
	disp(test_corr(1,2));
end

disp(corrcoef(combo_test(:,1),combo_test(:,2)));
disp(corrcoef(combo_dev(:,1),combo_dev(:,2)));
