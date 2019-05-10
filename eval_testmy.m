netmat = load('data/exp/net-epoch-1.mat');
net = dagnn.DagNN.loadobj(netmat.net);

%%
net.move('gpu') ;
net.mode = 'test' ;

% load and preprocess an image
im = imread('peppers.png') ;
im_ = single(im) ; % note: 0-255 range
im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
im_ = bsxfun(@minus, im_, net.meta.normalization.averageImage) ;
images = gpuArray(im_); %move to gpu
%%
inputs = {'data', images};
net.eval(inputs) ;
%%
% Gather results.
    % Gather results.
    index = strfind({net.layers.name}, 'softmax_global'); %softmax
    index = find(not(cellfun('isempty', index)));
    if isempty(index)
	index = strfind({net.layers.name}, 'softmax'); %softmax
    index = find(not(cellfun('isempty', index)));
    end
    npos = length(index);

