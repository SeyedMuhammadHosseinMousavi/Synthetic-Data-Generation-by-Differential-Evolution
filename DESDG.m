%% Synthetic Data Generation by Differential Evolution (DE)

clc;
clear;
close all;
load fisheriris.mat;
Target(1:50)=1;Target(51:100)=2;Target(101:150)=3;Target=Target'; % Original labels
y_data=reshape(meas,1,[]); % Preprocessing - convert matrix to vector
ss=size(y_data); SS= ss (1,2);
x_data = 1:SS;

%% Problem Definition
nVar=10;            % Number of Decision Variables
VarSize=[1 nVar];   % Decision Variables Matrix Size
VarMin=-10;          % Lower Bound of Decision Variables
VarMax= 10;          % Upper Bound of Decision Variables

%% DE Parameters
MaxIt=100;      % Maximum Number of Iterations
nPop=15;        % Population Size

beta_min=0.2;   % Lower Bound of Scaling Factor
beta_max=0.8;   % Upper Bound of Scaling Factor
pCR=0.2;        % Crossover Probability

% %% Initialization
Runs = 6; % 6 means synthetic samples 6 times more than original samples
for ii=1: Runs
empty_individual.Position=[];
empty_individual.Cost=[];
BestSol.Cost=inf;
pop=repmat(empty_individual,nPop,1);
agents = rand(nPop, nVar) * 10 - 5;  % Random initial positions
for i=1:nPop
agent = agents(i, :);
CostFunction=@(x) SDG(agent, x_data, y_data);    % Cost Function
pop(i).Position=unifrnd(VarMin,VarMax,VarSize);
pop(i).Cost=CostFunction(pop(i).Position);
if pop(i).Cost<BestSol.Cost
BestSol=pop(i);
end
end
BestCost=zeros(MaxIt,1);

%% DE Main Loop
for it=1:MaxIt
for i=1:nPop
agents = rand(nPop, nVar) * 10 - 5;  % Random initial positions
agent = agents(i, :);
CostFunction=@(x) SDG(agent, x_data, y_data);    % Cost Function
x=pop(i).Position;
A=randperm(nPop);
A(A==i)=[];
a=A(1);
b=A(2);
c=A(3);

% Mutation
beta=unifrnd(beta_min,beta_max,VarSize);
y=pop(a).Position+beta.*(pop(b).Position-pop(c).Position);
y = max(y, VarMin);
y = min(y, VarMax);
% Crossover
z=zeros(size(x));
j0=randi([1 numel(x)]);
for j=1:numel(x)
if j==j0 || rand<=pCR
z(j)=y(j);
else
z(j)=x(j);
end
end
NewSol.Position=z;
NewSol.Cost=CostFunction(NewSol.Position);

if NewSol.Cost<pop(i).Cost
pop(i)=NewSol;
if pop(i).Cost<BestSol.Cost
BestSol=pop(i);
% if agents(i, :) < agents(i-1, :)
% agents(i, :) = agents(i, :);
% end
end
end
end
% Update Best Cost
BestCost(it)=BestSol.Cost;
% Show Iteration Information
disp(['Iteration ' num2str(it) ': Best Cost = ' num2str(BestCost(it))]);
end

% Generate synthetic data using the optimized polynomial curve
synthetic_x = linspace(min(x_data), max(x_data), numel(x_data));
synthetic_y = polyval(agents(end, :), synthetic_x);
syn=rescale(synthetic_y,min(y_data),max(y_data)).*y_data;
syn=rescale(syn,min(y_data),max(y_data));
SyntheticData{ii}=syn';
end

%% Converting cell to matrix
SyntheticData=cell2mat(SyntheticData);
S = size(SyntheticData(Runs)); SO = size (meas); SF = SO (1,2); SO = SO (1,1); 
for i=1:Runs
Syn2{i}=reshape(SyntheticData(:,i),[SO,SF]);
Syn2{i}(:,end+1)=Target; 
end
Synthetic3 = cell2mat(Syn2');
SyntheticData=Synthetic3(:,1:end-1);
SyntheticLbl=Synthetic3(:,end);

%% Plot data and classes
Feature1=1;
Feature2=3;
f1=meas(:,Feature1); % feature1
f2=meas(:,Feature2); % feature 2
ff1=SyntheticData(:,Feature1); % feature1
ff2=SyntheticData(:,Feature2); % feature 2
figure('units','normalized','outerposition',[0 0 1 1])
subplot(3,2,1)
area(meas, 'linewidth',1); title('Original Data');
ax = gca; ax.FontSize = 12; ax.FontWeight='bold'; grid on;
subplot(3,2,2)
area(SyntheticData, 'linewidth',1); title('Synthetic Data');
ax = gca; ax.FontSize = 12; ax.FontWeight='bold'; grid on;
subplot(3,2,3)
gscatter(f1,f2,Target,'rkgb','.',20); title('Original');
ax = gca; ax.FontSize = 12; ax.FontWeight='bold'; grid on;
subplot(3,2,4)
gscatter(ff1,ff2,SyntheticLbl,'rkgb','.',20); title('Synthetic');
ax = gca; ax.FontSize = 12; ax.FontWeight='bold'; grid on;
subplot(3,2,[5 6])
semilogy(BestCost,'k', 'LineWidth', 2); title('DE Last Cost');
xlabel('Iteration');
ylabel('Best Cost');
grid on;
hold off;

%% Train and Test
% Training Synthetic dataset by SVM
Mdlsvm  = fitcecoc(SyntheticData,SyntheticLbl); CVMdlsvm = crossval(Mdlsvm); 
SVMError = kfoldLoss(CVMdlsvm); SVMAccAugTrain = (1 - SVMError)*100;
% Predict new samples (the whole original dataset)
[label5,score5,cost5] = predict(Mdlsvm,meas);
% Test error and accuracy calculations
sizlbl=size(Target); sizlbl=sizlbl(1,1);
countersvm=0; % Misclassifications places
misindexsvm=0; % Misclassifications indexes
for i=1:sizlbl
if Target(i)~=label5(i)
misindex(i)=i; countersvm=countersvm+1; end; end
% Testing the accuracy
TestErrAugsvm = countersvm*100/sizlbl; SVMAccAugTest = 100 - TestErrAugsvm;
% Result SVM
AugResSVM = [' Synthetic Train SVM "',num2str(SVMAccAugTrain),'" Test on Original Dataset"', num2str(SVMAccAugTest),'"'];
disp(AugResSVM);
