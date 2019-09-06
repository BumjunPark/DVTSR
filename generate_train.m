clear;
close all;
clc;

%% Train Set
data_folder_origin = 'F:\AIM2019\data\TSR_data\val\val_15fps';
label1_folder_origin = 'F:\AIM2019\data\TSR_data\val\val_30fps';
label02_folder_origin = 'F:\AIM2019\data\TSR_data\val\val_60fps';

train_data0_savepath = 'tsr_train_data0.h5';
train_data1_savepath = 'tsr_train_data1.h5';
train_label0_savepath = 'tsr_train_label0.h5';
train_label1_savepath = 'tsr_train_label1.h5';
train_label2_savepath = 'tsr_train_label2.h5';

% val_data0_savepath = 'tsr_val_data0.h5';
% val_data1_savepath = 'tsr_val_data1.h5';
% val_label0_savepath = 'tsr_val_label0.h5';
% val_label1_savepath = 'tsr_val_label1.h5';
% val_label2_savepath = 'tsr_val_label2.h5';

patch_size_h = 720;
patch_size_w = 1280;

%% generate train data
count = 0;
data0 = zeros(patch_size_h, patch_size_w, 3, 1350, 'uint8');
folder = dir(data_folder_origin);
n = length(folder);
for i = 3 : n
    folder_path = strcat(data_folder_origin, '\', folder(i).name);
    file = dir(folder_path);
    m = length(file);
    for j = 3 : m-1
        file_path = strcat(folder_path, '\', file(j).name);        
        origin = imread(file_path);        
        [hei, wid, c] = size(origin);        
        for x = 1  : patch_size_h : hei-patch_size_h+1
            for y = 1  :patch_size_w : wid-patch_size_w+1
                count = count+1;
                subim_origin = origin(x : x+patch_size_h-1, y : y+patch_size_w-1, :);                
                data0(:, :, :, count) = subim_origin;                
            end
        end
    end
    display(100*(i-2)/(n-2));display('percent complete(data0)');
end
order = randperm(count);
data0 = data0(:, :, :, order);
h5create(train_data0_savepath, '/data', size(data0), 'Datatype', 'uint8');
h5write(train_data0_savepath, '/data', data0);
clearvars data0;

count = 0;
data1 = zeros(patch_size_h, patch_size_w, 3, 1350, 'uint8');
for i = 3 : n
    folder_path = strcat(data_folder_origin, '\', folder(i).name);
    file = dir(folder_path);
    m = length(file);
    for j = 3 : m-1        
        file_path = strcat(folder_path, '\', file(j+1).name);        
        origin = imread(file_path);
        [hei, wid, c] = size(origin);
        for x = 1  : patch_size_h : hei-patch_size_h+1
            for y = 1  :patch_size_w : wid-patch_size_w+1
                count = count+1;                
                subim_origin = origin(x : x+patch_size_h-1, y : y+patch_size_w-1, :);                
                data1(:, :, :, count) = subim_origin;
            end
        end
    end
    display(100*(i-2)/(n-2));display('percent complete(data1)');
end

data1 = data1(:, :, :, order);
h5create(train_data1_savepath, '/data', size(data1), 'Datatype', 'uint8');
h5write(train_data1_savepath, '/data', data1);
clearvars data1;

%% generate label data
count = 0;
label1 = zeros(patch_size_h, patch_size_w, 3, 1350, 'uint8');
folder = dir(label1_folder_origin);
n = length(folder);
for i = 3 : n
    folder_path = strcat(label1_folder_origin, '\', folder(i).name);
    file = dir(folder_path);
    m = length(file);
    for j = 4 : 2 : m-1
        file_path = strcat(folder_path, '\', file(j).name);
        origin = imread(file_path);
        [hei, wid, c] = size(origin);        
        for x = 1  : patch_size_h : hei-patch_size_h+1
            for y = 1  :patch_size_w : wid-patch_size_w+1
                count = count+1;
                subim_origin = origin(x : x+patch_size_h-1, y : y+patch_size_w-1, :);
                label1(:, :, :, count) = subim_origin;
            end
        end
    end
    display(100*(i-2)/(n-2));display('percent complete(label1)');
end
label1 = label1(:, :, :, order);
h5create(train_label1_savepath, '/data', size(label1), 'Datatype', 'uint8');
h5write(train_label1_savepath, '/data', label1);
clearvars label1;

count = 0;
label0 = zeros(patch_size_h, patch_size_w, 3, 1350, 'uint8');
folder = dir(label02_folder_origin);
n = length(folder);
for i = 3 : n
    folder_path = strcat(label02_folder_origin, '\', folder(i).name);
    file = dir(folder_path);
    m = length(file);
    for j = 4 : 4 : m-3
        file_path = strcat(folder_path, '\', file(j).name);
        origin = imread(file_path);
        [hei, wid, c] = size(origin);        
        for x = 1  : patch_size_h : hei-patch_size_h+1
            for y = 1  :patch_size_w : wid-patch_size_w+1
                count = count+1;
                subim_origin = origin(x : x+patch_size_h-1, y : y+patch_size_w-1, :);
                label0(:, :, :, count) = subim_origin;
            end
        end
    end
    display(100*(i-2)/(n-2));display('percent complete(label0)');
end
label0 = label0(:, :, :, order);
h5create(train_label0_savepath, '/data', size(label0), 'Datatype', 'uint8');
h5write(train_label0_savepath, '/data', label0);
clearvars label0;

count = 0;
label2 = zeros(patch_size_h, patch_size_w, 3, 1350, 'uint8');
folder = dir(label02_folder_origin);
n = length(folder);
for i = 3 : n
    folder_path = strcat(label02_folder_origin, '\', folder(i).name);
    file = dir(folder_path);
    m = length(file);
    for j = 6 : 4 : m-1
        file_path = strcat(folder_path, '\', file(j).name);
        origin = imread(file_path);
        [hei, wid, c] = size(origin);        
        for x = 1  : patch_size_h : hei-patch_size_h+1
            for y = 1  :patch_size_w : wid-patch_size_w+1
                count = count+1;
                subim_origin = origin(x : x+patch_size_h-1, y : y+patch_size_w-1, :);
                label2(:, :, :, count) = subim_origin;
            end
        end
    end
    display(100*(i-2)/(n-2));display('percent complete(label2)');
end
label2 = label2(:, :, :, order);
h5create(train_label2_savepath, '/data', size(label2), 'Datatype', 'uint8');
h5write(train_label2_savepath, '/data', label2);
clearvars label2;