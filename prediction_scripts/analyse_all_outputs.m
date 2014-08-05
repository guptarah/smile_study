function [last_combo_test] = analyse_all_outputs(all_storage,combo_test,num_iter);

for i = 1:num_iter

	clc;
	em_iters = length(all_storage{1});
	vals = zeros(em_iters,4);
	for j = 1:em_iters
		cur_params = all_storage{i}{j};
		test_len_vector = length(cur_params.test_output);
		dev_len_vector = length(cur_params.dev_output);
		if j==1
		vals(j,:) = [mean(cur_params.RMSE_dev) mean(cur_params.RMSE_test) mean(cur_params.corr_coef_dev) mean(cur_params.corr_coef_test)];
		disp([mean(cur_params.RMSE_dev) mean(cur_params.RMSE_test) mean(cur_params.corr_coef_dev) mean(cur_params.corr_coef_test)])
		else
		vals(j,:) = [mean(cur_params.RMSE_dev)/sqrt(dev_len_vector) mean(cur_params.RMSE_test)/sqrt(test_len_vector) mean(cur_params.corr_coef_dev) mean(cur_params.corr_coef_test)];
                disp([mean(cur_params.RMSE_dev)/sqrt(dev_len_vector) mean(cur_params.RMSE_test)/sqrt(test_len_vector) mean(cur_params.corr_coef_dev) mean(cur_params.corr_coef_test)])
		end
%		pause;
	end

	[~,rmse_best] = min(vals(:,1));
	[~,corr_best] = max(vals(:,3));

	[~,oracle_best] = min(vals(:,2));

	disp([ rmse_best vals(rmse_best,:) ]);
	disp([ corr_best vals(corr_best,:)]);
	disp([oracle_best vals(oracle_best,:)]);
	pause


end


% get iteration wise results
em_iter_num = 4;
for i = 1:num_iter
        em_iters = length(all_storage{1});
        vals = zeros(em_iters,4);
	cur_params =  all_storage{i}{em_iter_num};
	last_combo_test{i}.tuned_test_output = cur_params.test_output;
	last_combo_test{i}.coefs = cur_params.coefs;
	last_combo_test{i}.baseline_test_output = combo_test{i}.baseline_test_output; 
	last_combo_test{i}.target_test = combo_test{i}.target_test; 
end 
