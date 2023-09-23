function [net, tr] = train_nn(epochs, hidden_layers, r, p, inputs, outputs, activation_func1, activation_func2, train_function)
    % Criar uma arquitetura MLP
    net = feedforwardnet(hidden_layers);
    net = configure(net, inputs, outputs);

    net.input.processFcns = {'mapminmax'};
    net.output.processFcns = {'mapminmax'};

    % Dividir padrões em treinamento, validação e teste
    net.divideFcn = 'divideind';
    net.divideParam.trainInd = 1:1320;
    net.divideParam.valInd = 1320:1320;
    net.divideParam.testInd= 1320:1320;

    % Inicializa pesos
    net=init(net);

    % Treinando a rede
    net.trainParam.showWindow=true;
    net.layers{1}.dimensions=r;
    net.layers{1}.transferFcn = activation_func1; % Funcao de ativacao da camada interna 1
    if(hidden_layers == 2)
        net.layers{2}.dimensions=p;
        net.layers{2}.transferFcn = activation_func2; % Funcao de ativacao da camada interna 2
    end
    net.layers{end}.size = 10;

    net.trainFcn = train_function;
    net.trainParam.epochs = epochs;
    net.trainParam.time = 1000;
    net.trainParam.lr = 0.1;
    net.trainParam.min_grad = 10^-8; % Criterio minimo de parado segundo professor
    net.trainParam.max_fail = 1000;

    [net, tr] = train(net, inputs', outputs');
end