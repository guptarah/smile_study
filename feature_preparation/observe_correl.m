function observe_correl(targets,features)

for i = 1:size(targets,2)
	correl_mat = zeros(1,size(features,2)); 
	for j = 1:size(features,2)
		corr_mat = corrcoef(targets(:,i),features(:,j));
		correl_mat(j) = corr_mat(1,2);
	end

	plot(correl_mat);
	pause;
end
