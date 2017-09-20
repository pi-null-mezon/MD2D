% Taranov Alex, thanks to https://github.com/lvdmaaten/bhtsne
clc;
disp('t-SNE for the multidimensional data');
% Let's upload descriptions first
Features = csvread('descriptions.txt');
%lets upload labels (it may seems bulky, but this method works for the latin and cyrillic symbols)
[f,msg] = fopen('labels.txt','r','n','UTF-8');
SL = textscan(f,'%[^\n]','delimiter','\n');
Labels = SL{1};

% Now we can make t-SNE
outDims = 2; pcaDims = 128; perplexity = 30; theta = 0; alg = 'svd';
disp('Data prepared. t-SNE started...');
map = fast_tsne(Features, outDims, pcaDims, perplexity, theta, alg, 5000);
disp('Prepare plot data...');
tic
figure
gscatter(map(:,1), map(:,2), Labels);
grid on
legend off
figure
coords = zeros(size(map));
maxc = max(map);
minc = min(map);
a(1,1) = 1.0 / (maxc(1,1) - minc(1,1));
b(1,1) = - minc(1,1) * a(1,1);
a(1,2) = 1.0 / (maxc(1,2) - minc(1,2));
b(1,2) = - minc(1,2) * a(1,2);
for i = 1:size(coords,1)
    coords(i,1) = 0.9*(map(i,1)*a(1,1) + b(1,1)) + 0.05;
    coords(i,2) = 0.9*(map(i,2)*a(1,2) + b(1,2)) + 0.05;
end
for i = 1:size(Labels,1)
 axes('pos',[coords(i,1), coords(i,2), 0.015, .02]);
 try
    I = imresize(imread(Labels{i}),[100 NaN]);
    imshow(I);
 catch
    warning(['Can not load ' Labels{i}]); 
 end
end
toc
disp('Saving plot on hard drive as png file...'); 
set(gcf, 'PaperUnits', 'inches', 'PaperPosition', [0 0 8000 6400]/600);
print -dpng -r600 VGGFace_tsne_graph.png
disp('Work has been finished');

