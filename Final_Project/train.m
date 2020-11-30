%%%%%%%% TRAIN.m %%%%%%%%

clear;
load trainingFaces
load trainingNonFaces

num = size(trainingFaces,3);
image = trainingFaces(:,:,1);
x = size(image, 1);
y = size(image, 2);
face_integrals = zeros(41, 41, num);
for i=1:num
    image = trainingFaces(:,:,i);
    image = image(35:75, 30:70);
    
    x = size(image, 1);
    y = size(image, 2);
    face_integrals(:,:,i)= integral_image(image);
end

num = size(trainingNonFaces,3);
image = trainingNonFaces(:,:,1);
nonface_integrals = zeros(41, 41, num);

for i=1:num
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

example_number = size(trainingFaces, 3) + size(trainingNonFaces, 3);
labels = zeros(example_number, 1);
labels (1:size(trainingFaces, 3)) = 1;
labels((size(trainingFaces, 3)+1):example_number) = -1;
examples = zeros(face_vertical, face_horizontal, example_number);
examples (:, :, 1:size(trainingFaces, 3)) = face_integrals;
examples(:, :, (size(trainingFaces, 3)+1):example_number) = nonface_integrals;

classifier_number = numel(weak_classifiers);

responses =  zeros(classifier_number, example_number);

for example = 1:example_number
    integral = examples(:, :, example);
    for feature = 1:classifier_number
        classifier = weak_classifiers {feature};
        responses(feature, example) = eval_weak_classifier(classifier, integral);
    end
end

boosted_classifier = AdaBoost(responses, labels, 10);

% save train 