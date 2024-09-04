
[book_data, book_chars] = readData();
[char_to_ind, ind_to_char] = getContainers(book_chars);

%##########PARAMS###########%
m = 100; % number of nodes in hidden state
sig = 0.01; 
K = length(book_chars); 
x_0 = 's'; % first character in sequence
h_0 = zeros(m, 1);
n = 5; % sequence length
epochs = 200;

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
RNN = init_RNN(K ,m ,sig);
Gradients = init_Gradients(K, m, n);

RNNwGradient(RNN, h_0, x_0, n, K, ind_to_char, char_to_ind, epochs, X, Y, Gradients);
%#####################%


function RNNwGradient(RNN, h_0, x_0, n, K, ind_to_char, char_to_ind, epochs, X, Y, Gradients)

    lossDataEpochs = zeros(epochs,1);

    for epoch = 1:epochs
        [generated_sentence, ~, ~ ,~, ~ , ~, ~, lossDataTime] = vanilla_RNN(RNN, h_0, x_0, n, K, ind_to_char, char_to_ind, Y);
        lossDataEpochs(epoch) = sum(lossDataTime);

    end
    display(generated_sentence);

    plot(lossDataEpochs);

end

function computeGradients(RNN, h_0, x_0, n, K, ind_to_char, char_to_find, Gradients)
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


     loss = -log(sum(Y.*P, 1));

    
end

function [sampled_char, a, h ,o, P , x, Y_t, lossDataTime] = vanilla_RNN(RNN, h_0 ,x_0, n , K, ind_to_char , char_to_ind, Ylabels)
    
    a = cell(n,1); % activation function
    h = cell(n,1); % hidden states
    o = cell(n,1); % outputs
    P = zeros(K, n); % Probabilities (matrix)
    x = cell(n,1); % inputs
    lossDataTime = zeros(n,1);


    sampled_index = cell(n,1); % generated index 
    sampled_char = cell(1,n); % generated character

    Y_t = cell(K, n); % one-hot encoding for generated character

    h{1} = h_0;
    x{1} = x_0;

    a{1} = RNN.W*h{1} + RNN.U * x{1} + RNN.b; 
    h{1} = tanh(a{1});
    o{1} = RNN.V*h{1} + RNN.c;

    P(:, 1) = exp(o{1}) ./ sum(exp(o{1})); % Softmax (size Kx1)
    %disp(size(P(:, 1)));
    %disp(size(Ylabels(:, 1)));
    lossDataTime(1) = computeCost(P(:, 1), Ylabels(:, 1));

    ii = sample_Index(P, 1);
    sampled_index{1} = ii;

    Y_t{1} = char_to_vector(ind_to_char(sampled_index{1}), K , char_to_ind);
    x{2} = Y_t{1};
    sampled_char{1} = ind_to_char(sampled_index{1});

    for t = 2:n
        a{t} = RNN.W*h{t-1} + RNN.U*x{t} + RNN.b;
        h{t} = tanh(a{t});
        o{t} = RNN.V*h{t} + RNN.c;

        P(:, t) = exp(o{t}) ./ sum(exp(o{t})); % Softmax (size Kx1)
        lossDataTime(t) = computeCost(P(:, t), Ylabels(:, t));

        ii = sample_Index(P, t);
        sampled_index{t} = ii;
        Y_t{t} = char_to_vector(ind_to_char(sampled_index{t}), K , char_to_ind);
        x{t+1} = Y_t{t};
        sampled_char{t} = ind_to_char(sampled_index{t});
    end

end

function index = sample_Index(P, t)
       cp = cumsum(P(:, t)); % Use regular parentheses for indexing
       a_rand = rand; 
       ixs = find(cp-a_rand >0);
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
function RNN = init_RNN(K, m, sig)
    RNN.m = m;
    RNN.eta = 0.1;
    RNN.seq_length = 25;
    RNN.b = zeros(m, 1);
    RNN.c = zeros(K, 1);
    RNN.U = randn(m, K) * sig;
    RNN.W = randn(m, m) * sig;
    RNN.V = randn(K, m) * sig;
end

function Gradients = init_Gradients(K, m , n)
    Gradients.b = zeros(m, 1);
    Gradients.c = zeros(K, 1);
    Gradients.U = zeros(m, K);
    Gradients.W = zeros(m, m);
    Gradients.V = zeros(K, m);
    Gradients.a = zeros(n, 1);
    Gradients.o = zeros(n, 1);
    Gradients.h = zeros(n, 1);
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



