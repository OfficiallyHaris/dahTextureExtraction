clear; 

numImages = 116;
imageFolder = './ball_frames';

% Texture Features Extraction
% Initialize arrays to store texture features for each ball type
% Format: [ball_type][channel][feature][orientation]
ASM = cell(3, 3);     % Angular Second Moment
Contrast = cell(3, 3); % Contrast
Correlation = cell(3, 3); % Correlation

% Arrays to store average values across orientations
ASM_avg = cell(3, 3);
Contrast_avg = cell(3, 3);
Correlation_avg = cell(3, 3);

% Arrays to store ranges across orientations
ASM_range = cell(3, 3);
Contrast_range = cell(3, 3);
Correlation_range = cell(3, 3);

% Process each image
for i = 54:numImages
    % Load RGB and indexed images
    rgbImg = imread(fullfile(imageFolder, sprintf('frame-%d_rgb.png', i)));
    idxImg = imread(fullfile(imageFolder, sprintf('frame-%d_indexed.png', i)));
    
    % For each ball type (1-3)
    for ballType = 1:3
        % Create mask for this ball
        ballMask = (idxImg == ballType);
        
        % Skip if this ball type is not in this image
        if sum(ballMask(:)) == 0
            continue;
        end
        
        % Get bounding box of the ball region
        stats = regionprops(ballMask, 'BoundingBox');
        bbox = round(stats.BoundingBox);
        
        % Extract region for each channel and compute GLCM
        for channel = 1:3
            % Extract this channel
            channelImg = rgbImg(:,:,channel);
            
            % Create a masked image (set background to 0)
            maskedImg = channelImg .* uint8(ballMask); % probably a built-in method 
            
            % Get the region of interest within the bounding box
            xStart = max(1, bbox(1));
            yStart = max(1, bbox(2));
            xEnd = min(size(maskedImg,2), bbox(1)+bbox(3)-1);
            yEnd = min(size(maskedImg,1), bbox(2)+bbox(4)-1);
            roi = maskedImg(yStart:yEnd, xStart:xEnd);
            roiMask = ballMask(yStart:yEnd, xStart:xEnd);
            
            % Skip if ROI is too small for GLCM (needs at least 2x2 pixels)
            if sum(roiMask(:)) < 4
                continue;
            end
            
            % Normalize to 8 levels to reduce computation 
            % and increase statistical significance
            roi = uint8(double(roi) / 256 * 8);
            
            % Define offsets for the four orientations: 0°, 45°, 90°, 135°
            offsets = [0 1; -1 1; -1 0; -1 -1]; 
            
            % Calculate GLCM for all orientations
            glcm = graycomatrix(roi, 'Offset', offsets, 'NumLevels', 8,'Symmetric', true, 'GrayLimits', [0 7]);
            
            % Calculate Haralick features
            stats = graycoprops(glcm, {'energy', 'contrast', 'correlation'});
            
            % Store features for each orientation
            for o = 1:4
                if isempty(ASM{ballType, channel})
                    ASM{ballType, channel} = [];
                    Contrast{ballType, channel} = [];
                    Correlation{ballType, channel} = [];
                end
                
                % Angular Second Moment (Energy in MATLAB)
                ASM{ballType, channel}(end+1, o) = stats.Energy(o);
                
                % Contrast
                Contrast{ballType, channel}(end+1, o) = stats.Contrast(o);
                
                % Correlation
                Correlation{ballType, channel}(end+1, o) = stats.Correlation(o);
            end
            
            % Calculate average and range for this instance
            instance_idx = size(ASM{ballType, channel}, 1);
            
            % Averages across orientations % Creating Rotation invariance
            ASM_avg{ballType, channel}(instance_idx) = mean(ASM{ballType, channel}(instance_idx, :));
            Contrast_avg{ballType, channel}(instance_idx) = mean(Contrast{ballType, channel}(instance_idx, :));
            Correlation_avg{ballType, channel}(instance_idx) = mean(Correlation{ballType, channel}(instance_idx, :));
            
            % Ranges across orientations
            ASM_range{ballType, channel}(instance_idx) = max(ASM{ballType, channel}(instance_idx, :)) - min(ASM{ballType, channel}(instance_idx, :));
            Contrast_range{ballType, channel}(instance_idx) = max(Contrast{ballType, channel}(instance_idx, :)) - min(Contrast{ballType, channel}(instance_idx, :));
            Correlation_range{ballType, channel}(instance_idx) = max(Correlation{ballType, channel}(instance_idx, :)) - min(Correlation{ballType, channel}(instance_idx, :));
        end
    end
end

% Plot distributions of selected texture features (one from each channel)
% Here I'm selecting Angular Second Moment (Energy) from Red channel,
% Contrast from Green channel, and Correlation from Blue channel
featureNames = {'ASM (Red Channel)', 'Contrast (Green Channel)', 'Correlation (Blue Channel)'};
ballNames = {'Ball Type 1', 'Ball Type 2', 'Ball Type 3'};
channelNames = {"Red", "Green", "Blue"};
featureData = {ASM_avg, Contrast_avg, Correlation_avg};
for feature = 1:3
    figure;
    
    % Loop through color channels (Red, Green, Blue)
    for colour_channel = 1:3
        subplot(1,3,colour_channel);
        hold on;
        
        % Plot for each ball type
        for ball = 1:3
            histogram(featureData{feature}{ball, colour_channel}, 'Normalization', 'probability');
        end
        
        title([channelNames{colour_channel} ' Channel: ' featureNames{feature}]);
        if colour_channel == 1
            legend(ballNames);
        end
        xlabel([featureNames{feature} ' Value']);
        ylabel('Probability');
        hold off;
    end
    
    sgtitle([featureNames{feature} ' Feature Distribution by Ball Type and Color Channel']);
end


% 
% colour_channel = 0;
% figure;
% % Plot ASM from Red channel
% subplot(1,3,1);
% hold on;
% % colour_channel = 0;
% for ball = 1:3
% 
%     colour_channel = colour_channel + 1;
%     histogram(ASM_avg{ball, 1}, 'Normalization', 'probability');
% end
% title(featureNames{1});
% legend(ballNames);
% hold off;
% 
% % Plot Contrast from Green channel
% subplot(1,3,2);
% hold on;
% for ball = 1:3
%     histogram(Contrast_avg{ball, 2}, 'Normalization', 'probability');
% end
% title(featureNames{2});
% legend(ballNames);
% hold off;
% 
% % Plot Correlation from Blue channel
% subplot(1,3,3);
% hold on;
% for ball = 1:3
%     histogram(Correlation_avg{ball, 3}, 'Normalization', 'probability');
% end
% title(featureNames{3});
% legend(ballNames);
% hold off;