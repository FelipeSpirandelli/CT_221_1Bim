clear
clc

% Configuracao
hidden_layers = 2;
r = 25;
activation_func1 = 'poslin';
p = 25;
activation_func2 = 'poslin';
train_function = 'trainbr';
epochs = 1000;
config_name = "config6";

% Importar dados
petr = readtable('./dados/PETR3.SA.csv');
embr = readtable('./dados/EMBR3.SA.csv');
vale = readtable('./dados/VALE3.SA.csv');

% Processando dados
lookback = 10;
numRows = height(petr) - lookback*2;
inputs = zeros(numRows, 3*lookback);
outputs_petr = zeros(numRows, lookback);
outputs_embr = zeros(numRows, lookback);
outputs_vale = zeros(numRows, lookback);

for i = 1:numRows
    inputs(i, 1:lookback) = petr.AdjClose(i:i+lookback-1);
    inputs(i, lookback+1:2*lookback) = embr.AdjClose(i:i+lookback-1);
    inputs(i, 2*lookback+1:3*lookback) = vale.AdjClose(i:i+lookback-1);
    outputs_petr(i, :) = petr.AdjClose(i+lookback:i+2*lookback-1);
    outputs_embr(i, :) = embr.AdjClose(i+lookback:i+2*lookback-1);
    outputs_vale(i, :) = vale.AdjClose(i+lookback:i+2*lookback-1);
end

train_inputs = inputs(1:1300, :);
train_outputs_petr = outputs_petr(1:1300, :);
train_outputs_embr = outputs_embr(1:1300, :);
train_outputs_vale = outputs_vale(1:1300, :);

% Treinando a rede para vale
[net_petr, tr_petr] = train_nn(epochs, hidden_layers, r, p, train_inputs, train_outputs_petr, activation_func1, activation_func2, train_function);
[net_embr, tr_embr] = train_nn(epochs, hidden_layers, r, p, train_inputs, train_outputs_embr, activation_func1, activation_func2, train_function);
[net_vale, tr_vale] = train_nn(epochs, hidden_layers, r, p, train_inputs, train_outputs_vale, activation_func1, activation_func2, train_function);

% Simular
simulated_petr = sim(net_petr, inputs');
simulated_embr = sim(net_embr, inputs');
simulated_vale = sim(net_vale, inputs');

figure;
hold on
plot(petr.Date(21:end), outputs_petr(:, 1), 'r');
plot(petr.Date(21:end), simulated_petr(1, :)', 'b');
grid
xlabel('Tempo (dia)', 'FontSize', 14, 'Interpreter', 'latex');
ylabel('Valor (R\$)', 'FontSize', 14, 'Interpreter', 'latex');
title('A\c{c}\~{a}o da Petrobras em fun\c{c}\~{a}o do tempo', 'FontSize', 14, 'Interpreter', 'latex');
legend({'Real', 'Previs\~{a}o', }, 'Location', 'northwest', 'FontSize', 14, 'interpreter', 'latex');
set(gca, 'FontSize', 14);
set(gca, 'TickLabelInterpreter', 'latex');
filename_petr = sprintf('petr_%s.eps', config_name);
print('-depsc2', filename_petr);
hold off

figure;
hold on
plot(embr.Date(21:end), outputs_embr(:, 1), 'r');
plot(embr.Date(21:end), simulated_embr(1, :)', 'b');
grid
xlabel('Tempo (dia)', 'FontSize', 14, 'Interpreter', 'latex');
ylabel('Valor (R\$)', 'FontSize', 14, 'Interpreter', 'latex');
title('A\c{c}\~{a}o da Embrear em fun\c{c}\~{a}o do tempo', 'FontSize', 14, 'Interpreter', 'latex');
legend({'Real', 'Previs\~{a}o', }, 'Location', 'northwest', 'FontSize', 14, 'interpreter', 'latex');
set(gca, 'FontSize', 14);
set(gca, 'TickLabelInterpreter', 'latex');
filename_embr = sprintf('embr_%s.eps', config_name);
print('-depsc2', filename_embr);
hold off

figure;
hold on
plot(vale.Date(21:end), outputs_vale(:, 1), 'r');
plot(vale.Date(21:end), simulated_vale(1, :)', 'b');
grid
xlabel('Tempo (dia)', 'FontSize', 14, 'Interpreter', 'latex');
ylabel('Valor (R\$)', 'FontSize', 14, 'Interpreter', 'latex');
title('A\c{c}\~{a}o da Vale em fun\c{c}\~{a}o do tempo','FontSize', 14, 'Interpreter', 'latex');
legend({'Real', 'Previs\~{a}o', }, 'Location', 'northwest', 'FontSize', 14, 'interpreter', 'latex');
set(gca, 'FontSize', 14);
set(gca, 'TickLabelInterpreter', 'latex');
filename_vale = sprintf('vale_%s.eps', config_name);
print('-depsc2', filename_vale);
