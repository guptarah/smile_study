function concatenate_features()

input_dir1='fpoint_norm/';
input_dir2='../../lbp_per_raw_frame/';
input_dir3='../../lbp_per_face/';
save_dir = 'to_use_features/';


files_dir2 = dir(strcat(input_dir2,'RA*'));

for files2 = files_dir2'
	file_id = strtok(files2.name,'.');
	disp(file_id);	
	file1_to_load = strcat(input_dir1,file_id,'*');
	file1_to_load = dir(file1_to_load);
	file1_to_load = file1_to_load.name;
	file1_to_load = strcat(input_dir1,file1_to_load);
	features1 = load(file1_to_load);

	file2_to_load = strcat(input_dir2,files2.name);
	features2 = load(file2_to_load);
	features2 = features2.lbp_static_frame;
	features2 = features2';	

	file3_to_load = strcat(input_dir3,files2.name);
	features3 = load(file3_to_load);
        features3 = features3.lbp_static_face;
        features3 = features3';
	

	disp([size(features1,1) size(features2,1) size(features3,1)])


	features = [features1 features2 features3];
	save_file = strcat(save_dir,file_id,'.csv');
	csvwrite(save_file,features);
	
end 
