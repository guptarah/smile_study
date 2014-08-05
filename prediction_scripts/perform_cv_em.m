function [combo_test,all_storage] = perform_cv_em(dbn_size,max_iter,save_file)

%features_dir='../feature_preparation/fpoint_norm/RA*';
features_dir='../feature_preparation/to_use_features/RA*';
feature_files = dir(features_dir);

feature_store=cell(10,1);
counter=0;
for file = feature_files'
	counter = counter + 1;
	to_load_file = strcat('../feature_preparation/to_use_features/',file.name);
	cur_rating = load(to_load_file);
	feature_store{counter} = cur_rating(:,2:end); 
end

disp('dimensionality');
disp(size(feature_store{1}));

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
combo_dev=cell(10,1);
combo_test=cell(10,1);
all_storage = cell(10,1);
for split_id = 1:10
%for split_id = 1
	disp(split_id);
	% preparing test, dev and train partitions
	train_data = [];
	train_lables = [];
	for i = 1:10
		if i == split_id
			test_data = feature_store{i};
			test_lables = lables_store{i};
		elseif mod(i,10) == mod(split_id + 1,10)
			dev_data = feature_store{i};
			dev_lables = lables_store{i};
		else
			train_data = [train_data;feature_store{i}];
			train_lables = [train_lables;lables_store{i}];
		end 
	end

	% normalize the lables and compute their average 
	[norm_train_lables,lab_mu,lab_sigma] = zscore(train_lables);
	target_train = sum(norm_train_lables,2)/size(train_lables,2);
	norm_test_lables = (test_lables - repmat(lab_mu,size(test_lables,1),1))./(repmat(lab_sigma,size(test_lables,1),1));
	target_test = sum(norm_test_lables,2)/size(train_lables,2);
	norm_dev_lables = (dev_lables - repmat(lab_mu,size(dev_lables,1),1))./(repmat(lab_sigma,size(dev_lables,1),1));
        target_dev = sum(norm_dev_lables,2)/size(train_lables,2);

	% normalize the features
	[norm_train_feat,feat_mu,feat_sigma] = zscore(train_data);
	norm_test_feat = (test_data - repmat(feat_mu,size(test_data,1),1))./(repmat(feat_sigma,size(test_data,1),1));
	norm_dev_feat = (dev_data - repmat(feat_mu,size(dev_data,1),1))./(repmat(feat_sigma,size(dev_data,1),1));

	% train the model 
	nbClasses = 1;

	flag = 1;
	iter_id = 0;
%	max_iter = 2;
	params_iter = cell(max_iter,1);	
	while flag == 1
		iter_id = iter_id + 1;
		if iter_id > max_iter
			flag = 0; 
                        break;
		elseif iter_id == max_iter
			flag = 0;
                end 	
		
		[dnn,train_output] = train_dnn(nbClasses,norm_train_feat,target_train,norm_train_feat,target_train,dbn_size);
		[~, ~, dev_output] = nntest(dnn, norm_dev_feat, target_dev);	
		[~, ~, test_output] = nntest(dnn, norm_test_feat, target_test); 
			
		% determine RMSE on each annotator
		if iter_id == 1 % it means we started from taking the mean of all the annotators
			[RMSE_train,corr_coef_train] = compute_rmse_iter1(train_output,norm_train_lables);
			disp([mean(RMSE_train),mean(corr_coef_train)]);
			[RMSE_dev,corr_coef_dev] = compute_rmse_iter1(dev_output,norm_dev_lables);
			disp([mean(RMSE_dev),mean(corr_coef_dev)]);
			[RMSE_test,corr_coef_test] = compute_rmse_iter1(test_output,norm_test_lables);
                        disp([mean(RMSE_test),mean(corr_coef_test)]);
		else
			[RMSE_train,corr_coef_train] = compute_rmse(train_output,norm_train_lables,coefs);
			disp([mean(RMSE_train),mean(corr_coef_train)]);
			[RMSE_dev,corr_coef_dev] = compute_rmse(dev_output,norm_dev_lables,coefs);	
			disp([mean(RMSE_dev),mean(corr_coef_dev)]);
			[RMSE_test,corr_coef_test] = compute_rmse(test_output,norm_test_lables,coefs);      
                        disp([mean(RMSE_test),mean(corr_coef_test)]);

		end			

		
		params_iter{iter_id}.dnn = dnn;
		params_iter{iter_id}.RMSE_dev = RMSE_dev;
		params_iter{iter_id}.RMSE_test = RMSE_test;
		params_iter{iter_id}.corr_coef_dev = corr_coef_dev;
		params_iter{iter_id}.corr_coef_test = corr_coef_test;
		params_iter{iter_id}.dev_output = dev_output;
		params_iter{iter_id}.test_output = test_output;
		if iter_id > 1
			params_iter{iter_id}.coefs = coefs;
		end

		if flag == 1 % dont want to unnecesarily do in last iteration	
			% determining coefficients and the next set of target lables
        	        [obt_hidden_G,coefs] = determine_coefs(train_output,norm_train_lables);
			%[obt_hidden_G,coefs] = determine_coefs_on_dev(train_output,norm_train_lables,dev_output,norm_dev_lables);
                	target_train = obt_hidden_G; 
		end

	end

	% find the best dev result and take corresponding test outputs
	dev_results = zeros(max_iter,2);
	test_results = zeros(max_iter,2);
	for iter_id = 1:max_iter
		dev_results(iter_id,:) = [mean(params_iter{iter_id}.RMSE_dev) mean(params_iter{iter_id}.corr_coef_dev)];
		test_results(iter_id,:) = [mean(params_iter{iter_id}.RMSE_test) mean(params_iter{iter_id}.corr_coef_test)];
	end 
	
	[~,best_iter_id] = min(dev_results(:,1));
%	[~,best_iter_id] = max(dev_results(:,2));
	disp(['best_iter_id:',num2str(best_iter_id)]);
	disp('obtained');	
	disp([dev_results(best_iter_id,:),test_results(best_iter_id,:)]);
	disp('baseline');
	disp([dev_results(1,:),test_results(1,:)]);	
	
	combo_test{split_id}.tuned_test_output = params_iter{best_iter_id}.test_output;
	if best_iter_id > 1
		disp(['best_iter_id:',num2str(best_iter_id)]);
		disp('getting >1 iter results');
		combo_test{split_id}.coefs = params_iter{best_iter_id}.coefs;
	else
		disp(['best_iter_id:',num2str(best_iter_id)]);
		combo_test{split_id}.coefs = -999;	
	end
	combo_test{split_id}.baseline_test_output = params_iter{1}.test_output;
	combo_test{split_id}.target_test = norm_test_lables;
	all_storage{split_id} = params_iter;
	
	if nargin == 3
		save(save_file,'all_storage');
	end
 
end

% get results on the entire test set
%evaluate_results_baseline(combo_test);
