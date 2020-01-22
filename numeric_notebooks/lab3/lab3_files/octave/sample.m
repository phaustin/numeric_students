function sample()

%%%% This is a sample that will generate two matrices, and graph
%%%% the matrices with different line styles
%%%% 
%%%% It also plots two arrows

%%%% set parametric mode 
gset parametric
hold on

%%%% setting some arrows
%%%% arrows will not show up until something is plotted 
%%%%   in the plot window
gset arrow 1 to 0,0,-4;
gset arrow 2 to 0,1,7;

%%% putting a label at the origin
gset label 1 "origin" at 0,0,0;

%%% labeling the axes
gset xlabel "x"; gset ylabel "y"; gset zlabel "z";

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
     k++;
end,

%%% displaying the two matrices generated
matrix1
matrix2

%%% plotting the two matrices
gsplot matrix1 with lines 1 title "matrix1", matrix2 with lines 2 title "matrix2";
