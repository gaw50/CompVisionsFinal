%%%%%%%% TRAIN.m %%%%%%%%
clear;
load trainingFaces
load trainingNonFaces

num = 1000;
startingAccuracy = 50;
for i = 1: num
        bootstrapped(:,:,i) = trainingFaces(:,:,i);
end
roundCount = 0;
isIncreasing = 1;
incorrectResults = 0;
bestAccuracy = 0;
while(isIncreasing)
    image = trainingFaces(:,:,1);
    if(roundCount >= 1)
        num = size(bootstrapped, 3) + 25
    end
    incorrectCount = 1;
    for i = num-24:num
        if(~incorrectResults)
            break
        end   
       i
       num-i
       incorrectResults(incorrectCount)
        bootstrapped(:,:,i) = trainingFaces(:,:, incorrectResults(incorrectCount));
        incorrectCount = incorrectCount + 1;
    end

    x = size(image, 1);
    y = size(image, 2);
    face_integrals = zeros(41, 41, num);
    for i=1:num
        image = bootstrapped(:,:,i);
        image = image(41:41, 41:41);

        x = size(image, 1);
        y = size(image, 2);
        face_integrals(:,:,i)= integral_image(image);
    end

    tnf = size(trainingNonFaces,3);
    image = trainingNonFaces(:,:,1);
    nonface_integrals = zeros(41, 41, tnf);

    for i=1:tnf
        image = trainingNonFaces(:,:,i);
        image = image(35:75, 30:70);

        x = size(image, 1);
        y = size(image, 2);
        nonface_integrals(:,:,i)= integral_image(image);

    end

    face_horizontal = 41;
    face_vertical = 41;

    number = 1000;
    % Generate Weak Classifiers 
    weak_classifiers = cell(1, number);
    for i = 1:number
        weak_classifiers{i} = generate_classifier(face_vertical, face_horizontal);
    end

    example_number = num + size(trainingNonFaces, 3);
    labels = zeros(example_number, 1);
    labels (1:size(bootstrapped, 3)) = 1;
    labels((size(bootstrapped, 3)+1):example_number) = -1;
    examples = zeros(face_vertical, face_horizontal, example_number);
    examples (:, :, 1:num) = face_integrals;
    examples(:, :, num+1:example_number) = nonface_integrals;

    classifier_number = numel(weak_classifiers);

    responses =  zeros(classifier_number, example_number);

    for example = 1:example_number
        integral = examples(:, :, example);
        for feature = 1:classifier_number
            classifier = weak_classifiers {feature};
            responses(feature, example) = eval_weak_classifier(classifier, integral);
        end
    end



    boosted_classifier = AdaBoost(responses, labels, 15);

    x = size(trainingFaces, 1);
    y = size(trainingFaces, 2);
    z = size(trainingFaces, 3);

    results = zeros(x, y, num);

    for i = 1: z
        results(:,:,i) = boosted_multiscale_search(trainingFaces(:,:,i), 1, boosted_classifier, weak_classifiers, [41, 41]);  
    end

    correct = 0;
    incorrect = 0;
    for q = 1: z
        tmp = results(:,:,q);
        for x = 1: 41
            for y = 1: 41
                tmp(x,y) = results(41+x, 41+y, q);
            end
        end
        tmp = (tmp > 1);
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
        if (count < 50)
            incorrect = incorrect + 1;
            incorrectResults(incorrect) = q;
        end
    end


    tmpAccuracy = (correct / z) * 100
    
    if(tmpAccuracy > startingAccuracy)
        
        if(tmpAccuracy >bestAccuracy)
            bestAccuracy = tmpAccuracy;
        end
        prevAccuracy = tmpAccuracy;
      
    end
    
    if(tmpAccuracy > 98.99)
        bestAccuracy = tmpAccuracy
        
    end

    roundCount = roundCount + 1
    if (roundCount == 5)
        isIncreasing = 0;
    end
end

save trainBootstrap 
