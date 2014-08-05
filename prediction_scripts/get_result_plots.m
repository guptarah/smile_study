function last_combo_test = get_result_plots(all_storage)

lables_dir='../../ratings/RA*';
lables_file=dir(lables_dir);
lables_store=cell(10,1);
counter = 0;  
for file = lables_file' 
        counter = counter + 1;
        to_load_file = strcat('../../ratings/',file.name);
        cur_lables = load(to_load_file);

        % Removing annotator 2 and 3. COMMENT THIS IF NOT WANTED 
%        cur_lables(:,3) = []; % remove annotator 2
%        cur_lables(:,3) = []; % remove annotator 3


        lables_store{counter} = cur_lables(:,2:end);

end

for split_id = 1:10
        disp(split_id);
        % preparing test, dev and train partitions
        train_data = [];
        train_lables = [];
        for i = 1:10
                if i == split_id
                        test_lables = lables_store{i};
                elseif mod(i,10) == mod(split_id + 1,10)
                        dev_lables = lables_store{i};
                else
                        train_lables = [train_lables;lables_store{i}];
                end
        end

	[norm_train_lables,lab_mu,lab_sigma] = zscore(train_lables);
	norm_test_lables = (test_lables - repmat(lab_mu,size(test_lables,1),1))./(repmat(lab_sigma,size(test_lables,1),1));

	i = split_id;
	em_iter_num = 4;
	em_iters = length(all_storage{1});
        cur_params =  all_storage{i}{em_iter_num};
        last_combo_test{i}.tuned_test_output = cur_params.test_output;
        last_combo_test{i}.coefs = cur_params.coefs;


	cur_params = all_storage{i}{1};
        last_combo_test{i}.baseline_test_output = cur_params.test_output; 
        last_combo_test{i}.target_test = norm_test_lables; 

end
 
