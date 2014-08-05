function [obt_hidden_G,coefs] = determine_coefs(nn_output,annot_lables,nn_output_dev,annot_lables_dev,win_len)

if nargin < 5
	win_len = 10;
end

% determining coeffs
coefs = zeros(size(annot_lables_dev,2),length(-1*win_len:win_len));

X = zeros(size(nn_output_dev,1),length(-1*win_len:win_len));
for (i = -1*win_len:win_len)
        X(:,i+1+win_len) = circshift(nn_output_dev,i);
end

for annot_id = 1:size(annot_lables_dev,2)
	cur_Y = annot_lables_dev(:,annot_id);
	coefs_Y = pinv(X)*cur_Y;	
	coefs(annot_id,:) = coefs_Y';
end 

% determine new target G from A = BG

% Formulating B
% Divide data into chunks to find G
num_chunks = 10;
partiton_len = floor(size(annot_lables,1)/10);
obt_hidden_G = zeros(size(nn_output));
for i_chunk = 0:num_chunks-1
	disp ('chunk_id');
	disp(i_chunk);
	if i_chunk < num_chunks-1
	annot_lables_cur = annot_lables(i_chunk*partiton_len+1:(i_chunk+1)*partiton_len,:);
	else
	annot_lables_cur = annot_lables(i_chunk*partiton_len+1:end,:);
	end

	% preparing the matrices 
	B = zeros((size(annot_lables_cur,1)-2*win_len)*size(annot_lables_cur,2),size(annot_lables_cur,1));
	A = reshape(annot_lables_cur(win_len+1:end-win_len,:),(size(annot_lables_cur,1)-2*win_len)*size(annot_lables_cur,2),1);
	for i_annt = 0:size(annot_lables_cur,2)-1
        	start_point = i_annt*(size(annot_lables_cur,1)-2*win_len);
	        start_row = zeros(1,size(annot_lables_cur,1));
        	start_row(1:length(-1*win_len:win_len)) = coefs(i_annt+1,:);
	        for row_id = 1:(size(annot_lables_cur,1)-2*win_len)
        	        B(start_point+row_id,:) = circshift(start_row',row_id-1)';
	        end
	end     
        
	% compute G
	disp('computing pinv');
	G = pinv(B)*A;
	disp('computing gold truth');
	obt_hidden_G(i_chunk*partiton_len+1:i_chunk*partiton_len+length(G)) = G;

end

