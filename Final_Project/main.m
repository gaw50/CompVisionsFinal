clear;
load images
load examples1000
load faces1000



imshow(nonface_integrals(:,:,1), [])
for i=1:50
    
    x = face_integrals(:,:,i);
    imshow(x, [])
end
face_horizontal = face_size(2);
face_vertical = face_size(1);

number = 1000;

% Generate Weak Classifiers 
weak_classifiers = cell(1, number);
for i = 1:number
    weak_classifiers{i} = generate_classifier(face_vertical, face_horizontal);
    
end

example_number = size(faces, 3) + size(nonfaces, 3);
labels = zeros(example_number, 1);
labels (1:size(faces, 3)) = 1;
labels((size(faces, 3)+1):example_number) = -1;
examples = zeros(face_vertical, face_horizontal, example_number);
examples (:, :, 1:size(faces, 3)) = face_integrals;
examples(:, :, (size(faces, 3)+1):example_number) = nonface_integrals;

classifier_number = numel(weak_classifiers);

responses =  zeros(classifier_number, example_number);

for example = 1:example_number
    integral = examples(:, :, example);
    for feature = 1:classifier_number
        classifier = weak_classifiers {feature};
        responses(feature, example) = eval_weak_classifier(classifier, integral);
    end
    disp(example)
end



boosted_classifier = AdaBoost(responses, labels, 15);
photo2 = read_gray('training_test_data/test_cropped_faces/04200d71.bmp');
result = boosted_multiscale_search(photo2, 1, boosted_classifier, weak_classifiers, [31, 25]);
result = boosted_detector(photo2, 1:0.5:3, boosted_classifier, weak_classifiers, [31, 25], 1);
