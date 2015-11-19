function [B,x,f,Hz] = schauderbasis (T,Q,N)
% Make Schauder basis
%
% INPUT: 
% 
% T - length that Z will have
% Q - defines the range of frequencies covered by Z
% N - period (Z will have 0.5*T/N periods)
%
% OUTPUT
%
% B - schauder basis
% x - frequencies 
% f - to which frequency does each each basis in B belong? 
% Hz - Sampling frequency
% Author: Diego Vidaurre, OHBA, University of Oxford

if nargin<2
    Q = T/2;
end;
if nargin<3
    N = 1;
end;

x = 0:1/(T-1):(N + (N-1) / (T-1));
f = [1:Q; 1:Q]; f = [0 f(:)']; Hz = length(x)/max(x);
B = ones(T*N,1);  
 
for i=1:Q 
    B = [B sqrt(2)*sin(2*i*pi*x)' sqrt(2)*cos(2*i*pi*x)'];
end; 
