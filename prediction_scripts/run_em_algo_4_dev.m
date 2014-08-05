function run_em_algo_4_dev()

dbn_size = 5;
save_file = 'outputs_3_iter_dev';
max_iter = 4;
[combo_test,all_storage] = perform_cv_em_lr_dev(dbn_size,max_iter,save_file);
save(save_file,'all_storage','combo_test');

evaluate_results_baseline(combo_test);
