function [data, labels] = loadMNIST(path_to_digits, path_to_labels)

    % Open files
    fid1 = fopen(path_to_digits, 'r','b');
    
    % The labels file
    fid2 = fopen(path_to_labels, 'r','b');
    
    %% Reading the images

    magicNum = fread(fid1,1,'uint32');
    numImgs = fread(fid1,1,'uint32');
    imageHeight = fread(fid1,1,'uint32');
    imageWidth = fread(fid1,1,'uint32');

    %% Reading rawdata

    rawImageData = uint8(fread(fid1,imageHeight * imageWidth * numImgs,'uint8'));
    data = double(reshape(rawImageData,[imageHeight * imageWidth, numImgs]));
    size(data)
    %% Reading Labels
    magicNum = fread(fid2,1,'uint32');
    numLabels = fread(fid2,1,'uint32');
    labels_vec = fread(fid2,numLabels,'uint8');

    labels = zeros(10,numLabels);
    for i = 1:numLabels
        labels(labels_vec(i)+1,i) = 1;
    end

    fclose(fid1);
    fclose(fid2);
end