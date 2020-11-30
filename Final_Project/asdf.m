clear;
addpath C:\Users\Reagan\Desktop\training_faces
addpath C:\Users\Reagan\Desktop\training_nonfaces
addpath C:\Users\Reagan\Desktop\test_cropped_faces
addpath C:\Users\Reagan\Desktop\test_face_photos

% load all the training images
%cd C:\Users\Reagan\Desktop\training_nonfaces
% Get list of all BMP files in this directory
% DIR returns as a structure array.  You will need to use () and . to get
% the file names.
cd C:\Users\Reagan\Desktop\test_face_photos
imagefiles = dir();      
nfiles = length(imagefiles); %Number of files found
nfiles = nfiles-1;
currentfilename = imagefiles(3).name;
currentimage = imread(currentfilename);
count = 0
for i=3:nfiles
   currentfilename = imagefiles(i).name;
   currentimage = imread(currentfilename);
   
   x = size(currentimage, 2);
   y = size(currentimage, 1);
   
   testFacePhotos{i} = currentimage;
   count = count + 1;
   x = 4;
end

save testFacePhotos testFacePhotos

%cd C:\Users\Reagan\Desktop\training_nonfaces
trainingNonFaces = zeros(x, y, nfiles);
imagefiles = dir();      
nfiles = length(imagefiles); %Number of files found
nfiles = nfiles-1;
currentfilename = imagefiles(3).name;
currentimage = imread(currentfilename);
x = size(currentimage, 1);
y = size(currentimage, 2);
trainingFaces = zeros(x, y, nfiles);
trainingNonFaces = zeros(x, y, nfiles);
for i=3:nfiles
   currentfilename = imagefiles(i).name;
   currentimage = imread(currentfilename);
   
   trainingNonFaces{i} = currentimage;
   
   x = 4;
end


%save trainingNonFaces


cd C:\Users\Reagan\Desktop\test_cropped_faces
% Get list of all BMP files in this directory
% DIR returns as a structure array.  You will need to use () and . to get
% the file names.

imagefiles = dir();      
nfiles = length(imagefiles); %Number of files found
nfiles = nfiles-1;
currentfilename = imagefiles(3).name;
currentimage = imread(currentfilename);
x = size(currentimage, 1);
y = size(currentimage, 2);
testFaces = zeros(x, y, nfiles);
for i=3:nfiles
   currentfilename = imagefiles(i).name;
   currentimage = imread(currentfilename);
   
   testFaces(:, :, i-2) = currentimage;
   
   
end

save testFaces testFaces



