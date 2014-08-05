function run_em_algo_3()

dbn_size = 5;
save_file = 'outputs_2_iter';
max_iter = 3;
[combo_test,all_storage] = perform_cv_em_lr(dbn_size,max_iter,save_file);
save(save_file,'all_storage','combo_test');

evaluate_results_baseline(combo_test);
