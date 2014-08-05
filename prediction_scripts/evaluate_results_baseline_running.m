function evaluate_results_baseline_running(combo_test,iter_len)

baseline_op = [];
best_op = [];
target_op = [];
for i = 1:iter_len 
	baseline_op = [baseline_op; combo_test{i}.baseline_test_output];
	target_op = [target_op; combo_test{i}.target_test];
	best_op = [best_op; combo_test{i}.tuned_test_output];
end

[RMSE,corr_coef] = compute_rmse_iter1(baseline_op,target_op);
disp('overall performance');
disp([mean(RMSE),mean(corr_coef)]);

disp('corr with mean lables');
corr_mat = corrcoef(baseline_op,mean(target_op,2));
disp(corr_mat(1,2));

RMSE_mean = RMSE;
corr_mean = corr_coef;

num_annt = size(combo_test{1}.target_test,2);
outputs = cell(num_annt,1);
for i =  1:iter_len 
	test_op = combo_test{i}.tuned_test_output;
	cur_coefs = combo_test{i}.coefs;
	if numel(cur_coefs) == 1
		for annt_id = 1:num_annt
			filtered_op = test_op; 
                        outputs{annt_id} = [outputs{annt_id};filtered_op];
		end
	else
		for annt_id = 1:num_annt
			filtered_op =  filter(cur_coefs(annt_id,:),1,test_op);
			outputs{annt_id} = [outputs{annt_id};filtered_op]; 
		end
	end
end

RMSE = zeros(num_annt,1);
corr_coef = zeros(num_annt,1);
to_show_mat = [];
for i = 1:num_annt
[RMSE_1,corr_coef_1] = compute_rmse_iter1(outputs{i},target_op(:,i));
RMSE(i) = RMSE_1;
corr_coef(i) = corr_coef_1;
to_show_mat=[to_show_mat [RMSE_mean(i) corr_mean(i);RMSE_1,corr_coef_1]];
end
disp(to_show_mat);
 
disp('overall performance');
disp([mean(RMSE),mean(corr_coef)]);
