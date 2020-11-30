clear;
load train
load testFaces
load testFacePhotos

% x = size(testFaces, 1);
% y = size(testFaces, 2);
% z = size(testFaces, 3);
% 
% results = zeros(x, y, z);
% 
% for i = 1: z
%     results(:,:,i) = boosted_multiscale_search(testFaces(:,:,i), 1, boosted_classifier, weak_classifiers, [41, 41]);
%     
% end
% 
% correct = 0;
% for q = 1: z
%     tmp = results(:,:,q);
%     for x = 1: 41
%         for y = 1: 41
%             tmp(x,y) = results(41+x, 41+y, q);
%         end
%     end
%     tmp = (tmp > 4);
%     count = 0;
%     for x = 1: 41
%         for y = 1: 41
%             if(tmp(x,y) == 1)
%                 count = count + 1;
%             end
%         end
%     end
%     
%     if (count > 50)
%         correct = correct + 1;
%     end
% end
% 
% AdaBoostAccuracy = (correct / z) * 100;
% 
% %%% skin detection





negative_histogram = read_double_image('negatives.bin');
positive_histogram = read_double_image('positives.bin');

image = testFacePhotos{1,3};

%%%%%%% Extracting 100x100 chunk sized windows
for i=1:99: size(image,1)
    if ((i+99) > size(image,1))
            continue
    end
    for q=1:99: size(image,2)
        if ((q+99) > size(image,2))
            continue
        end
        % Getting the window 
        window = (image(i:i+99, q:q+99, :));
        imshow(window, []);
        % find skin for the window
        result = detect_skin(window, positive_histogram, negative_histogram);
        % if theres a 75% chance that pixel is a skin pixel we keep it
        result = (result > .75);
        
        check = 0;
        
        % looping over the result to see if theres enough skin pixels
        % to classify as a face
        for m=1: size(result,1)
            for q=1: size(result,2)
                if(result(m,q) == 1)
                    check = check + 1;
                end
            end
        end
        
        % if there are 75 skin pixels we classify as a face
        if (check > 75)
            colorFaceResults{i} = boosted_multiscale_search(result, 1, boosted_classifier, weak_classifiers, [41, 41]);
        end
    end
end

for i=1: size(colorFaceResults, 2)
    tmp = colorFaceResults{1,i};
    if(~isempty(tmp))
        imshow(tmp,[]);
    end
end
  
