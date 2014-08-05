function [RMSE,corr_coef] = compute_rmse(train_output,norm_train_lables,coefs)

num_annt = size(norm_train_lables,2);

RMSE = zeros(1,num_annt); 
corr_coef = zeros(1,num_annt);
for i = 1:num_annt
	% find the obtained target by convolution
	obt_op = filter(coefs(i,:),1,train_output);
	RMSE(i) =  sqrt((obt_op - norm_train_lables(:,i))'*(obt_op - norm_train_lables(:,i)));
	corr_coef_mat = corrcoef(norm_train_lables(:,i),obt_op);
	corr_coef(i) = corr_coef_mat(1,2);
end
