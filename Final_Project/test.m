clear;
load train
load testFaces
load testFacesPhotos




x = size(testFaces, 1);
y = size(testFaces, 2);
z = size(testFaces, 3);

results = zeros(x, y, z);

for i = 1: z
    results(:,:,i) = boosted_multiscale_search(testFaces(:,:,i), 3, boosted_classifier, weak_classifiers, [41, 41]);  
end

correct = 0;
for q = 1: z
    tmp = results(:,:,q);
    
    for x = 1: 41
        for y = 1: 41
            tmp(x,y) = results(41+x, 41+y, q);
        end
    end
    tmp = (tmp > 2);
    count = 0;
    for x = 1: 41
        for y = 1: 41
            if(tmp(x,y) == 1)
                count = count + 1;
            end
        end
    end
    
    if (count > 25)
        correct = correct + 1;
    end
    
end

accuracy = (correct / z) * 100;% %%% skin detection

load trainBootstrap

x = size(testFaces, 1);
y = size(testFaces, 2);
z = size(testFaces, 3);

results = zeros(x, y, z);

for i = 1: z
    results(:,:,i) = boosted_multiscale_search(testFaces(:,:,i), 3, boosted_classifier, weak_classifiers, [41, 41]);  
end

correct = 0;
for q = 1: z
    tmp = results(:,:,q);
    
    for x = 1: 41
        for y = 1: 41
            tmp(x,y) = results(41+x, 41+y, q);
        end
    end
    tmp = (tmp > 2);
    count = 0;
    for x = 1: 41
        for y = 1: 41
            if(tmp(x,y) == 1)
                count = count + 1;
            end
        end
    end
    
    if (count > 25)
        correct = correct + 1;
    end
    
end

BootstrapAccuracy = (correct / z) * 100;% %%% skin detection



negative_histogram = read_double_image('negatives.bin');
positive_histogram = read_double_image('positives.bin');


for l = 3:  size(testFacesPhotos,2)
    testimage = testFacesPhotos{1,l};

    windowNum = 1;
    foundFaces = 0;
    correct = 0;
    skinCount = 0;

    skinimage = detect_skin(testimage, positive_histogram, negative_histogram);
    skinimage = (skinimage > .55);
    face_size = [50,50]
    face_vertical = face_size(1);
    face_horizontal = face_size(2);


    vertical_size = size(testimage, 1);
    horizontal_size = size(testimage, 2);

    result = zeros(vertical_size, horizontal_size);

    foundFaces = 0;
    for i=1:100: size(skinimage,1)

        if ((i+100) > size(skinimage,1))
                continue
        end
        for q=1:100: size(skinimage,2)

            if ((q+100) > size(skinimage,2))
                continue
            end
            window = (skinimage(i:i+100, q:q+100, :));
            window2 = (testimage(i:i+100, q:q+100, :));

            %result = cascade(window2);


            check = 0;
            
            for m=1: size(window,1)
                for n=1: size(window,2)
                    if(window(m,n) == 1)
                        check = check + 1;
                    end
                end
            end
            if (check > 1000)
               foundFaces = foundFaces + 1; 
                 break;
            end
            windowNum = windowNum + 1;
        end
    end

    thistThis = rgb2gray(testimage);
    thistThis = double(thistThis);
    [ans, boxes] =  boosted_detector_demo(thistThis, 2,  boosted_classifier, ...
              weak_classifiers, [41, 41], foundFaces);
    
    figure(l);
    imshow(ans, []);
end

% based off human judgment with skin detection bootstrapping and adaBoost
totalAccuracy = 14/35
