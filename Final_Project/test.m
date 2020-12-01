clear;
load train
load testFaces
load testFacesPhotos
% 
% x = size(testFaces, 1);
% y = size(testFaces, 2);
% 
% 
% 
x = testFaces(:,:,43);
imshow(x, []);
% imshow(x, []);
% for i = 1: 50
%     results(:,:,i) = boosted_multiscale_search(testFaces(:,:,i), 1, boosted_classifier, weak_classifiers, [41, 41]);
%     f = 8;
% end
% 
% imshow(results(:,:,32)>4);
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
%         imshow(tmp, []);
%     end
% end
% 
% AdaBoostAccuracy = (correct / z) * 100;

%%% skin detection

negative_histogram = read_double_image('negatives.bin');
positive_histogram = read_double_image('positives.bin');

testimage = testFacesPhotos{1,17};

windowNum = 1;
foundFaces = 0;
correct = 0;
skinCount = 0;

skinimage = detect_skin(testimage, positive_histogram, negative_histogram);
skinimage = (skinimage > .75);
imshow(skinimage,[]);
face_size = [50,50]
face_vertical = face_size(1);
face_horizontal = face_size(2);

vertical_size = size(testimage, 1);
horizontal_size = size(testimage, 2);

result = zeros(vertical_size, horizontal_size);


for i=1:100: size(skinimage,1)
    
    if ((i+100) > size(skinimage,1))
            continue
    end
    for q=1:100: size(skinimage,2)
        
        if ((q+100) > size(skinimage,2))
            continue
        end
        window = (skinimage(i:i+100, q:q+100, :));
       
        
        
        check = 0;
        imshow(window, []);
        for m=1: size(window,1)
            for n=1: size(window,2)
                if(window(m,n) == 1)
                    check = check + 1;
                end
            end
        end
        if (check > 1000)
           
           
           
             imshow(testimage, []);
             thistThis = rgb2gray(testimage);
             thistThis = double(thistThis);
             [result, boxes] =  boosted_detector_demo(thistThis, 1,  boosted_classifier, ...
                          weak_classifiers, [77, 77], 2);
             imshow(result, []);
        end
        windowNum = windowNum + 1;
    end
end


% for i = 1 : size(faceWindows,2)
%     photo = rgb2gray(faceWindows{1,i});
%     photo = double(photo);
%     
%     imshow(photo, [])
%     
% 
%     tmp =boosted_multiscale_search(photo, 2, boosted_classifier, weak_classifiers, [41, 41]);
%     figure(3);
%     imshow(max((tmp > 4) * 255, photo * 0.5), [])
% 
%     imshow(tmp, []);
%     count = 0;
%     for x = 1: size(tmp,1)
%         for y = 1: size(tmp, 2)
%             if(tmp(x,y) == 1)
%                 count = count + 1;
%             end
%         end
%     end
%     count
%     if (count > 50)
%         correctFaces(:, i) = facePosition(:,i);
%     end
%     if((count < 50))
%         foundFaces = foundFaces -1;
%         
%     end
% 
% end
% 
% 
% for i =1: foundFaces
%     result = draw_rectangle2(testimage, correctFaces(1, i)+49, correctFaces(2, i)+49, 75, 75);
%     figure(i);
%     imshow(result);
%     p = 1;
% end

  
