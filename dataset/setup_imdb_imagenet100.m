function imdb = setup_imdb_imagenet100(imdbs_dir,varargin)
opts.seed = 1 ;
opts.joint = 0;
opts.datasetName = 'ILSVRC2012_base';
opts.imdbs_dir = imdbs_dir;
opts.imdb_pattern = 'imagenet-1000-100-01-01.mat';
opts = vl_argparse(opts, varargin) ;
imdbPath = fullfile(opts.imdbs_dir, opts.imdb_pattern);
exemplars = load(imdbPath);
imdb = exemplars.imdb;
imdb.meta.sets = {'train', 'val', 'test'};
fprintf('%d classes found! \n', numel(imdb.meta.classes));

% images
imdb.images.name    = {};
imdb.images.class   = imdb.images.classes;
imdb.images.set     = imdb.images.set;

for c = 1:length(imdb.images.data),
   file = imdb.images.data{c};
   setn = imdb.images.set(c);
   folder =  imdb.images.folders(c);
   if setn == 1 % train
       curr_name = fullfile('data/ILSVRC2012','train',folder,file);
   else   % test ( which is val in imagenet)  
       curr_name = fullfile('data/ILSVRC2012','val',folder,file);	
   end
   %imdb.images.name{c} = curr_name; 
   imdb.images.name = cat(2,imdb.images.name,curr_name);

end

fprintf('done ...........\n');
