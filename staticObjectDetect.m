%% Object Detection using Point Feature Matching

% Simulation Code for Final Project, Math 214

% University of Michigan College of Engineering

% By: Allison Bruns, Andy Forche, Jessie Houghton, Kevin Yan, Gillian Minnehan

% MathWork (2019) Object Detection in a Cluttered Scene Using Point Feature Matching
% Source: https://www.mathworks.com/help/vision/examples/object-detection-in-a-cluttered-scene-using-point-feature-matching.html


close all;
clear all;

% Read the reference image containing the object of interest.
colorstickerImage = imread('sticker.jpg');
stickerImage = rgb2gray(colorstickerImage);
figure;
imshow(stickerImage);
title('Image of a Sticker');

% Read the target image containing a cluttered scene.
colorsceneImage = imread('computer.jpg'); 
sceneImage = rgb2gray(colorsceneImage);
figure;
imshow(sceneImage);
title('Image of a Cluttered Scene');

%% Detect Feature Points

% Detect feature points in both images.
boxPoints = detectSURFFeatures(stickerImage);
scenePoints = detectSURFFeatures(sceneImage);

% Visualize the strongest feature points found in the reference image.
figure;
imshow(stickerImage);
title('100 Strongest Feature Points from Sticker Image');
hold on;
plot(selectStrongest(boxPoints, 100));

% Visualize the strongest feature points found in the target image.
figure;
imshow(sceneImage);
title('300 Strongest Feature Points from Scene Image');
hold on;
plot(selectStrongest(scenePoints, 300));

%%% Extract Feature Descriptors

[boxFeatures, boxPoints] = extractFeatures(stickerImage, boxPoints);
[sceneFeatures, scenePoints] = extractFeatures(sceneImage, scenePoints);

%%% Find Putative Point Matches

% Match the features using their descriptors.
boxPairs = matchFeatures(boxFeatures, sceneFeatures);

% Display putatively matched features.
matchedBoxPoints = boxPoints(boxPairs(:, 1), :);
matchedScenePoints = scenePoints(boxPairs(:, 2), :);
figure;
showMatchedFeatures(stickerImage, sceneImage, matchedBoxPoints, ...
    matchedScenePoints, 'montage');
title('Putatively Matched Points (Including Outliers)');

%%% Locate the Object in the Scene Using Putative Matches

% estimateGeometricTransform calculates the transformation relating the matched points, 
% while eliminating outliers. 
% This transformation allows us to localize the object in the scene.
[tform, inlierBoxPoints, inlierScenePoints] = ...
    estimateGeometricTransform(matchedBoxPoints, matchedScenePoints, 'affine');

% Display the matching point pairs with the outliers removed
figure;
showMatchedFeatures(stickerImage, sceneImage, inlierBoxPoints, ...
    inlierScenePoints, 'montage');
title('Matched Points (Inliers Only)');

% Get the bounding polygon of the reference image.
boxPolygon = [1, 1;...                           % top-left
        size(stickerImage, 2), 1;...                 % top-right
        size(stickerImage, 2), size(stickerImage, 1);... % bottom-right
        1, size(stickerImage, 1);...                 % bottom-left
        1, 1];                   % top-left again to close the polygon
    
newBoxPolygon = transformPointsForward(tform, boxPolygon);

figure;
imshow(sceneImage);
hold on;
line(newBoxPolygon(:, 1), newBoxPolygon(:, 2), 'Color', 'y');
title('Detected Sticker');

