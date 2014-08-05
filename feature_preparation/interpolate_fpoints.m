function interpolate_fpoints(fpoint_dir,output_dir)

to_get_files = strcat(fpoint_dir,'/*.pnt');
fpoint_files = dir(to_get_files);
for file = fpoint_files'
	to_load_file = strcat(fpoint_dir,'/',file.name);
	data = load(to_load_file);
	disp(file.name);
	num_frames = data(end,1);

	nan_padded_xy = NaN(num_frames,size(data,2));	
	for i = 1:size(data,1)
		nan_padded_xy(data(i,1),:) = data(i,:); 		
	end
	
	for i = 1:size(data,2)
		cur_stream = nan_padded_xy(:,i);
		nanx = isnan(cur_stream);
		t = 1:numel(cur_stream);

		if nanx(1) 
			cur_stream(1,:) = data(1,i);
			nanx(1) = 0;
			if i == 1 
				disp('no 1st est');
			end
		end

		if nanx(end)
			cur_stream(end,:) = data(end,i);
			nanx(end) = 0;
			if i == 1
				disp('no last est');
			end
		end

		cur_stream(nanx) = interp1(t(~nanx), cur_stream(~nanx), t(nanx));
		nan_padded_xy(:,i) = cur_stream;	
	end

	% Normalize features
	to_subtract = repmat(nan_padded_xy(:,55:56),1,size(nan_padded_xy,2)/2);
	to_subtract(:,55) = mean(nan_padded_xy(:,55));
	to_subtract(:,56) = mean(nan_padded_xy(:,56));

	normalized_xy = nan_padded_xy - to_subtract;

	to_divide = repmat([abs(normalized_xy(:,3)) abs(normalized_xy(:,4))],1,size(nan_padded_xy,2)/2);
%	normalized_xy = normalized_xy./to_divide;

	% compute the deltas
	diff_normalized_xy = diff(normalized_xy,1);
	diff_normalized_xy = [diff_normalized_xy(1,:) ;diff_normalized_xy];
	normalized_xy = [normalized_xy diff_normalized_xy(:,3:end)];
	
	to_save_xy = normalized_xy(:,3:end);
	to_save_file = strcat(output_dir,'/',file.name);
	csvwrite(to_save_file,to_save_xy);	
	
end 
