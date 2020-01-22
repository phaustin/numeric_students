function sample()

%%%% This is a sample that will generate two matrices, and graph
%%%% the matrices 

hold on

%%%% generate the matrices
for i = 1:10
   matrix1(i,1) = i;
   matrix1(i,2) = i + 1;
   matrix1(i,3) = i + 2;
end,

k = 1
  
%%%% numbers incremented by 3
for j = 1:3:28 
     matrix2(k,1) = -j;
     matrix2(k,2) = 1 - j;
     matrix2(k,3) = 2 - j;
     k=k+1;
end,

%%% displaying the two matrices generated
matrix1
matrix2

%% set view angle
  view ([135,45]);

%%% plotting the two matrices, set the title and label the x and y axes
  mesh (matrix1),title "matrix1&matrix2",xlabel "x",ylabel "y",zlabel "z";
mesh(matrix2);

% putting a label at the origin
text (0,0,0,'origin');

% the octave version also adds some arrows.  In matlab the easiest way to add
% arrows is using the interactive editor.
