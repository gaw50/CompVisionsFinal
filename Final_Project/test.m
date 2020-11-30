clear;
load train
load testFaces
load testFacePhotos


%%% starting test for plain adaBoost %%%
x = size(testFaces, 1);
y = size(testFaces, 2);
z = size(testFaces, 3);

results = zeros(x, y, z);

for i = 1: z
    results(:,:,i) = boosted_multiscale_search(testFaces(:,:,i), 1, boosted_classifier, weak_classifiers, [41, 41]);
    
end

correct = 0;
for q = 1: z
    tmp = results(:,:,q);
    for x = 1: 41
        for y = 1: 41
            tmp(x,y) = results(41+x, 41+y, q);
        end
    end
    tmp = (tmp > 4);
    count = 0;
    for x = 1: 41
        for y = 1: 41
            if(tmp(x,y) == 1)
                count = count + 1;
            end
        end
    end
    
    if (count > 50)
        correct = correct + 1;
    end
end

accuracy = (correct / z) * 100;

%%% add skin detection under here %%%
