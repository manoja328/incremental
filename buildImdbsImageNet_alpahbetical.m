% Build ImageNet imdbs following the guidelines of iCaRL: Incremental Classifier and Representation Learning
% https://arxiv.org/abs/1611.07725

max_iters = 1;
nclasses = 1000;
%batch_sizes = [100];
%[[100] ; ones(90,1)*10]
batch_sizes = [100,10]; % for tylers 100 and then 10 , 10
imdb_path = 'data/ILSVRC2012/train';
imdbtest_path = 'data/ILSVRC2012/val';
val_ground_truth = 'data/ILSVRC2012/ILSVRC2012_validation_ground_truth.txt';
outdir = 'data/ImageNet_incremental';

%% Split into small im
classes_dirs = importdata('data/icarl_sorted.txt');
for nit=1:max_iters
    order = 1:nclasses;
    
    % Print order for TensorFlow.
    fprintf('order = np.array([');
    for o_ix = 1 : length(order) - 1
        fprintf('%d,', order(o_ix));
    end
    fprintf('%d])', order(end));
    
    for i=1:length(batch_sizes) % Batch_sizes
        nParts = nclasses / batch_sizes(i);
        for j=1:nParts
            %% Cut data
            % Find positions
            in = (j-1) * batch_sizes(i) + 1;
            en = in + batch_sizes(i) - 1;
            
            % Build imdb.images
            classes = order(in:en); 
            
            %% Initialize imdb
            imdb.images.data = {};
            imdb.images.folders = {};
            imdb.images.classes = [];
            imdb.images.labels = [];
            imdb.images.set = [];
            
            for c_ix = in : en
                % Train files
                subdir = classes_dirs{order(c_ix)};               
                files = dir(fullfile(imdb_path, subdir, '*.JPEG'));
                
                
                data = {files(:).name};
                [folders{1:length(files)}] = deal(subdir);
                labels = zeros(1, length(files)) + c_ix;
                set = zeros(1, length(files)) + 1;
                
                % Cat
                imdb.images.data = cat(2, imdb.images.data, data);
                imdb.images.folders = cat(2, imdb.images.folders, folders);
                imdb.images.classes = cat(2, imdb.images.classes, labels);
                imdb.images.labels = cat(2, imdb.images.labels, labels);
                imdb.images.set = cat(2, imdb.images.set, set);
                
                clear('data', 'folders', 'labels', 'set');
                
                % Test files               
                                
                files = dir(fullfile(imdbtest_path,subdir, '*.JPEG'));
                labs = load(val_ground_truth);
                pos = find(labs == c_ix);

                data = {files(:).name};
                [folders{1:length(pos)}] = deal(subdir);
                labels = zeros(1, length(pos)) + c_ix;
                set = zeros(1, length(pos)) + 3;
                
                % Cat
                imdb.images.data = cat(2, imdb.images.data, data);
                imdb.images.folders = cat(2, imdb.images.folders, folders);
                imdb.images.classes = cat(2, imdb.images.classes, labels);
                imdb.images.labels = cat(2, imdb.images.labels, labels);
                imdb.images.set = cat(2, imdb.images.set, set);
                
                clear('data', 'folders', 'labels', 'set');
            end
            
            % Build imdb.meta
            imdb.meta.sets = {'train', 'val', 'test'} ;
            imdb.meta.classes = classes;
            imdb.meta.dataMean = [123.68 116.779 103.939]; % Same as in ICARL.
            imdb.meta.meanType = 'channels';
            imdb.meta.whitenData = 0;
            imdb.meta.contrastNormalization = 0;
            imdb.meta.trainRoot = imdb_path;
            imdb.meta.testRoot = imdbtest_path;
            
            %% Save imdb
            outname = sprintf('imagenet-%d-%02d-%02d-%02d.mat', nclasses, batch_sizes(i), j, nit);
            outpath = fullfile(outdir, outname);
            save(outpath, 'imdb');
            clear imdb;
        end
    end
end

disp('done.....')
