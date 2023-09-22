clear
clc

% Configuracao

camadas = 1;
r = 15;
transformacao_1 = 'tansig';
p = 10;
transformacao_2 = 'tansig';
trainFunction = 'trainlm';
epochs = 1000;



% 1) Importar dados
petr = readtable('./dados/PETR3.SA.csv');
embr = readtable('./dados/EMBR3.SA.csv');
vale = readtable('./dados/VALE3.SA.csv');


% 2) Processando dados
lookback = 10;
numRows = height(petr) - lookback*2;
inputs = zeros(numRows, 3*lookback);
outputs = zeros(numRows, lookback);

for i = 1:numRows
    inputs(i, 1:lookback) = petr.AdjClose(i:i+lookback-1);
    inputs(i, lookback+1:2*lookback) = embr.AdjClose(i:i+lookback-1);
    inputs(i, 2*lookback+1:3*lookback) = vale.AdjClose(i:i+lookback-1);
    outputs(i, :) = petr.AdjClose(i+lookback:i+2*lookback-1);
end


% 3) Criar uma arquitetura MLP
net = feedforwardnet(camadas);  % duas camadas escondidas
net = configure(net, inputs, outputs);

net.input.processFcns = {'mapminmax'};
net.output.processFcns = {'mapminmax'};

% 4) Dividir padrões em treinamento, validação e teste
net.divideFcn = 'divideind';
net.divideParam.trainInd = 1:1283;
net.divideParam.valInd = 1283:1283;
net.divideParam.testInd= 1283:1363;


% 5) Inicializa pesos
net=init(net);

% 6) Treinando a rede
net.trainParam.showWindow=true;
net.layers{1}.dimensions=r;
if(camadas == 2)
    net.layers{2}.dimensions=p;
    net.layers{2}.transferFcn = transformacao_2; % Funcao de ativacao da camada interna 2
end
net.layers{1}.transferFcn = transformacao_1; % Funcao de ativacao da camada interna 1
net.layers{end}.size = 10;


net.trainFcn = trainFunction;
net.trainParam.epochs = epochs;
net.trainParam.time = 1000;
net.trainParam.lr = 0.1;
net.trainParam.min_grad = 10^-8; % Criterio minimo de parado segundo professor
net.trainParam.max_fail = 1000;

[net, tr] = train(net, inputs', outputs');

% 7) Simular resposta
simulated = sim(net, inputs');
plot(outputs(:, 1), '+r');
hold on
plot(simulated(1, :)', 'b');
grid
xlabel('Tempo');
ylabel('Valor acao');
title('Acao em funcao do tempo');
