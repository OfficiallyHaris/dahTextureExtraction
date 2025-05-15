%Laaraiedh, M. (n.d.). Implementation of Kalman Filter with Python Language. 
%[online] Available at: https://arxiv.org/pdf/1204.0375.
%

clear;


x = csvread('x.csv');
y = csvread('y.csv');

na = csvread('na.csv');
nb = csvread('nb.csv');

% Plot real vs noisy trajectories
figure;
plot(x, y, 'g-', 'LineWidth', 2); hold on;
plot(na, nb, 'rx--', 'LineWidth', 1);
legend('Ground Truth', 'Noisy Observations');
xlabel('X'); ylabel('Y');
title('Ball Trajectory: Ground Truth vs Noisy');
grid on;

% Step 2: Define Kalman filter parameters
dt = 0.5;

% State transition matrix (constant velocity model)
F = [1 0 dt 0;
     0 1 0 dt;
     0 0 1  0;
     0 0 0  1];

% Cartesian Observation matrix (we observe only x and y)
H = [1 0 0 0;
     0 1 0 0];

% Initial state estimate (use first noisy point with 0 velocity)
x_k = [na(1); nb(1); 0; 0];

% Initial covariance matrix
P = eye(4) * 500;

% Process noise covariance
Q = [0.16 0 0 0;
     0 0.36 0 0; 
     0 0 0.8 0; 
     0 0 0 0.16];

% Measurement noise covariance
R = [0.25 0; 
     0 0.5];

% Step 3: Run Kalman Filter
N = length(na);
estimated = zeros(N, 2); % Store estimated positions

for k = 1:N
    % Measurement at current time
    z = [na(k); nb(k)];

    % Predict
    x_pred = F * x_k;
    P_pred = F * P * F' + Q;

    % Kalman Gain
    K = P_pred * H' / (H * P_pred * H' + R);

    % Update
    x_k = x_pred + K * (z - H * x_pred);
    P = (eye(4) - K * H) * P_pred;

    % Store estimate
    estimated(k, :) = x_k(1:2)';
end

figure;
plot(x, y, 'g-', 'LineWidth', 2); hold on;
plot(na, nb, 'rx--', 'LineWidth', 1);
plot(estimated(:,1), estimated(:,2), 'b-', 'LineWidth', 2);
legend('Ground Truth', 'Noisy Observations', 'Kalman Estimate');
xlabel('X'); ylabel('Y');
title('Kalman Filter Tracking Results');
grid on;



% Step 4.1: RMSE calculations
% Ensure all arrays are same length
min_len = min([length(x), size(estimated,1)]);
x = x(:);
y = y(:);
na = na(:);
nb = nb(:);


gt = [x(1:min_len), y(1:min_len)];
noisy = [na(1:min_len), nb(1:min_len)];
estimated = estimated(1:min_len, :);


rmse_noisy = sqrt(mean(sum((noisy - gt).^2, 2)));
rmse_estimated = sqrt(mean(sum((estimated - gt).^2, 2)));

% Also compute standard deviation of errors
err_noisy = sqrt(sum((noisy - gt).^2, 2));
err_estimated = sqrt(sum((estimated - gt).^2, 2));

std_noisy = std(err_noisy);
std_estimated = std(err_estimated);

fprintf('RMSE (Noisy): %.3f, Std Dev: %.3f\n', rmse_noisy, std_noisy);
fprintf('RMSE (Kalman Estimate): %.3f, Std Dev: %.3f\n', rmse_estimated, std_estimated);