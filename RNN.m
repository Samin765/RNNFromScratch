[book_data, book_chars] = readData();
[char_to_ind, ind_to_char] = getContainers(book_chars);

%##########PARAMS###########%
m = 5; % number of nodes in hidden state
sig = 0.01; 
K = length(book_chars); 
x_0 = 's'; % first character in sequence
h_0 = zeros(m, 1);
n = 20; % sequence length
epochs = 200;
eta = 0.001;

X_chars = book_data(1: n); % Input
Y_chars = book_data(2: n+1); % Labels


% Assume you have a cell array with your vectors
vectorsY = cell(1, n); 
vectorsX = cell(1, n); 

for i = 1:n
    vectorsY{i} = char_to_vector(Y_chars(i) , K, char_to_ind); 
    vectorsX{i} = char_to_vector(X_chars(i) , K, char_to_ind); 
end

% Initialize the matrix to hold all vectors

Y = zeros(K, n);
X = zeros(K, n);

% Fill the matrix with each vector from the cell array
for i = 1:n
    Y(:, i) = vectorsY{i}; 
    X(:, i) = vectorsX{i}; 

end



%#####################%


%########MAIN#############%

x_0 = char_to_vector(x_0 , K, char_to_ind);
RNN = init_RNN(K ,m ,sig, eta);
Gradients = init_Gradients(K, m, n);

num_grads = ComputeGradsNum(X,Y , RNN , 1e-4);

disp(num_grads.W);

RNNwGradient(RNN, h_0, x_0, n, K, ind_to_char, char_to_ind, epochs, X, Y, Gradients);
%#####################%

function num_grads = ComputeGradsNum(X, Y, RNN, h, Gradients)

    for f = fieldnames(RNN)'
        disp('Computing numerical gradient for')
        disp(['Field name: ' f{1} ]);
        num_grads.(f{1}) = ComputeGradNumSlow(X, Y, f{1}, RNN, h);


    end

end

function grad = ComputeGradNumSlow(X, Y, f, RNN, h)

    n = numel(RNN.(f)(1,:));
    grad = zeros(size(RNN.(f)));
    hprev = zeros(size(RNN.W, 1), 1);
    
    for i=1:n
        RNN_try = RNN;

        RNN_try.(f)(:,i) = RNN.(f)(:,i) - h;
        l1 = sum(computeCost(X, Y));

        RNN_try.(f)(:,i) = RNN.(f)(:,i) + h;
        l2 = sum(computeCost(X, Y));

        
        grad(i,:) = (l2-l1)/(2*h);
                

    end
end


function RNNwGradient(RNN, h_0, x_0, n, K, ind_to_char, char_to_ind, epochs, X, Y, Gradients)

    lossDataEpochs = zeros(epochs,1);

    for epoch = 1:epochs
        [generated_sentence, a, h ,o, P , x, Y_t, lossDataTime, Y_time, h_time,a_time,x_time] = vanilla_RNN(RNN, h_0, x_0, n, K, ind_to_char, char_to_ind, Y);
        lossDataEpochs(epoch) = sum(lossDataTime);
        [Gradients] = computeGradients(RNN, h_0, x_0, n, K, ind_to_char, char_to_ind, Gradients, Y_t, P, Y_time, h_time, a_time,x_time);
        
        %disp(size(Gradients.W));
        %disp(size(RNN.W));
        RNN.W = RNN.W - RNN.eta .* Gradients.W;
        RNN.V = RNN.V - RNN.eta .* Gradients.V;
        RNN.U = RNN.U - RNN.eta .* Gradients.U;
        RNN.b = RNN.b - RNN.eta .* Gradients.b;
        RNN.c = RNN.c - RNN.eta .* Gradients.c;

        

    end
    display(generated_sentence);

    plot(lossDataEpochs);

end

% Compute Gradients %
function [Gradients] = computeGradients(RNN, h_0, x_0, n, K, ind_to_char, char_to_find, Gradients, Y_t, P, Y_time, h_time,a_time,x_time)
    Gradients.o = -(Y_time - P);
    %disp(size(Gradients.V));
    
    for t = 1:n
        
        Gradients.V = Gradients.V + Gradients.o(:,t) * h_time(:,t)';
        
    end
    

    %disp(size(Gradients.a));
    %disp(size(h_time));
    %disp(size(Gradients.U));
    %disp(size(RNN.V));

    Gradients.h(:,n) = Gradients.o(:,n)' * RNN.V;
    Gradients.a(:,n) = Gradients.h(:,n)' * diag(1 - (tanh(a_time(:,n))).^2);

    for t = n-1:-1:1
        Gradients.h(:,t) = Gradients.o(:,t)' * RNN.V + Gradients.a(:, t+1)' * RNN.W;
        Gradients.a(:,t) = Gradients.h(:,t)' * diag(1-(tanh(a_time(:,t))).^2);
    end

    for t = 2:n
        Gradients.W = Gradients.W + Gradients.a(:,t) * h_time(:, t-1)';
    end
    
    for t = 1:n
        Gradients.U = Gradients.U + Gradients.a(:,t) * x_time(:,t)';
    end

    for t = 1:n
        Gradients.c = Gradients.c + Gradients.o(:, t); % For output bias c
    end
    
    for t = 1:n
        Gradients.b = Gradients.b + Gradients.a(:, t); % For hidden layer bias b
    end

end

function [loss] = computeCost(P, Y)
    isYCell = iscell(Y);
    isPCell = iscell(P);

    if isYCell
        Y = cell2mat(Y);  
    end

    if isPCell
        P = cell2mat(P);  
    end


    loss = -log(sum(Y .* P, 1) + 1e-8);

    
end

% VANILLA RNN %
function [sampled_char, a, h ,o, P , x, Y_t, lossDataTime, Y_time, h_time, a_time, x_time] = vanilla_RNN(RNN, h_0 ,x_0, n , K, ind_to_char , char_to_ind, Ylabels)
    
    a = cell(n,1); % activation function
    h = cell(n,1); % hidden states
    o = cell(n,1); % outputs
    P = zeros(K, n); % Probabilities (matrix)
    x = cell(n,1); % inputs
    lossDataTime = zeros(n,1);



    sampled_index = cell(n,1); % generated index 
    sampled_char = cell(1,n); % generated character

    Y_t = cell(K, n); % one-hot encoding for generated character

    % new matrices cause im dumb and used cells instead of matrices  :)%
    Y_time = zeros(K,n);
    h_time = zeros(RNN.m,n);
    a_time = zeros(RNN.m, n);
    x_time = zeros(K,n);

    %h{1} = h_0;
    x{1} = x_0;
    x_time(:,1) = x{1};

    a{1} = RNN.W*h_0 + RNN.U * x{1} + RNN.b; 
    a_time(:,1) = a{1};

    h{1} = tanh(a{1});
    h_time(:,1) = h{1};
    
    o{1} = RNN.V*h{1} + RNN.c;
    o{1} = o{1} - max(o{1});

    P(:, 1) = exp(o{1}) ./ sum(exp(o{1})); % Softmax (size Kx1)
    %disp(size(P(:, 1)));
    %disp(size(Ylabels(:, 1)));
    lossDataTime(1) = computeCost(P(:, 1), Ylabels(:, 1));

    ii = sample_Index(P, 1);
    sampled_index{1} = ii;

    Y_t{1} = char_to_vector(ind_to_char(sampled_index{1}), K , char_to_ind);
   
    Y_time(:,1) = Y_t{1};

    x{2} = Y_t{1};
    x_time(:,2) = x{2};
    sampled_char{1} = ind_to_char(sampled_index{1});

    for t = 2:n
        a{t} = RNN.W*h{t-1} + RNN.U*x{t} + RNN.b;
        a_time(:,t) = a{t};
        h{t} = tanh(a{t});
        h_time(:,t) = h{t};
        o{t} = RNN.V*h{t} + RNN.c;
        o{t} = o{t} - max(o{t});

        P(:, t) = exp(o{t}) ./ sum(exp(o{t})); % Softmax (size Kx1)
        lossDataTime(t) = computeCost(P(:, t), Ylabels(:, t));

        ii = sample_Index(P, t);
        sampled_index{t} = ii;
        Y_t{t} = char_to_vector(ind_to_char(sampled_index{t}), K , char_to_ind);
        Y_time(:,t) = Y_t{t};
        x{t+1} = Y_t{t};
        x_time(:,t+1) = x{t+1}; %% kanske fel t-1?
        sampled_char{t} = ind_to_char(sampled_index{t});
    end


end

function index = sample_Index(P, t)
       cp = cumsum(P(:, t)); % Use regular parentheses for indexing
       a_rand = rand; 
       ixs = find(cp-a_rand >0);
       %disp(size(ixs));
       index = ixs(1);
end

% Function to transform from word to one hot vector
function x_0 = char_to_vector(x_0chars, K, char_to_ind)
    x_0 = zeros(K,1);
    for i = 1:length(x_0chars)
        x_0(char_to_ind(x_0chars(i))) = 1;
    end

end

% Function to initalize RNN Hyper Parameters

function RNN = init_RNN(K, m, sig, eta)
    RNN.W = randn(m, m) * sig;
    RNN.V = randn(K, m) * sig;
    RNN.b = zeros(m, 1);
    RNN.c = zeros(K, 1);
    RNN.U = randn(m, K) * sig;


    RNN.eta = eta;
    RNN.seq_length = 25;
        RNN.m = m;

end

function Gradients = init_Gradients(K, m , n)

    Gradients.W = zeros(m, m);
    Gradients.V = zeros(K, m);
    Gradients.b = zeros(m, 1);
    Gradients.c = zeros(K, 1);
    Gradients.U = zeros(m, K);

    Gradients.a = zeros(m, n);
    Gradients.o = zeros(n, 1);
    Gradients.h = zeros(m, n);
end

%Function to get transformation between char/ind
function [char_to_ind, ind_to_char] = getContainers(book_chars)
    char_to_ind = containers.Map('KeyType', 'char', 'ValueType', 'int32');
    ind_to_char = containers.Map('KeyType', 'int32', 'ValueType', 'char');
    
    for i = 1:length(book_chars)
        char_to_ind(book_chars(i)) = i;
        ind_to_char(i) = book_chars(i);
    end
end

%Function to read Book Data and Characters
function [book_datas, book_chars] = readData()

    book_fname = 'goblet_book.txt';
    fid = fopen(book_fname, 'r');
    book_datas = fscanf(fid, '%c');
    fclose(fid);
    book_chars = unique(book_datas);
end

function [matrix,m] = AdaGrad(matrix,gradient,m,eta)
    m = m + gradient .^ 2;
    matrix = matrix - (eta ./ sqrt(m + eps)) .* gradient;
end



