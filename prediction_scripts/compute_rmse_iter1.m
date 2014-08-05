function [RMSE,corr_coef] = compute_rmse_iter1(train_output,norm_train_lables)

% first determine the initial set of coefs used to transform
% C*target_train = norm_train_lables
% In this case the RMSE filter would be just assigning scaled outputs (by a factor of 1/no. of annt)

num_annt = size(norm_train_lables,2);
RMSE = zeros(1,num_annt);
corr_coef = zeros(1,num_annt);
for i = 1:num_annt
%	RMSE(i) = sqrt((norm_train_lables(:,i) - (1/num_annt)*train_output)'*(norm_train_lables(:,i) - (1/num_annt)*train_output));
%	corr_coef_mat = corrcoef(norm_train_lables(:,i),(1/num_annt)*train_output);

        RMSE(i) = sqrt(1/length(train_output))* sqrt((norm_train_lables(:,i) - (1)*train_output)'*(norm_train_lables(:,i) - (1)*train_output));
	
        corr_coef_mat = corrcoef(norm_train_lables(:,i),(1)*train_output);



	corr_coef(i) = corr_coef_mat(1,2); 
end

