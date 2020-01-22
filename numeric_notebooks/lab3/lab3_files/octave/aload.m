%
% aload.m: This file from the Octave mailing list archive, item #301 (1995). 
%

function [X]=aload(filename,nr)
% function [X]=aload(filename,nr) loads a flat ASCII file into X. If the
% number of rows nr is not specified, will read in the whole file.

if (~exist(filename))
        error("File not found")
end

nc=shell_cmd(["head -n 1 " filename " | wc -w | awk '{printf $1}'"],1);
if nargin<2
  nr=shell_cmd(["wc -l " filename " | awk '{printf $1}'"],1);
end

tmpfile = sprintf("/tmp/oct%s", shell_cmd("echo -n $$"));
fid=fopen(tmpfile,"w");
fprintf(fid,"# name: localX\n# type: matrix\n");
fprintf(fid,["# rows: " nr "\n# columns: " nc "\n"]);
fclose(fid);

if nargin<2
  shell_cmd(["/bin/cat " filename " >> " tmpfile]);
else
  shell_cmd(["/usr/bin/head -n " num2str(nr) " " filename  ...
	  " >> " tmpfile]);
end
  
eval (["load -force " tmpfile]);

shell_cmd(["/bin/rm -f " tmpfile]);

X=localX; 

%% 
%% From help-octave-request@che.utexas.edu  Tue Jun  6 21:20:29 1995
%% Received: (daemon@localhost) by bullwinkle.che.utexas.edu (8.6.10/8.6) id VAA23396 for help-octave-outgoing; Tue, 6 Jun 1995 21:18:19 GMT
%% Received: from morpheus (root@morpheus.wustl.edu [128.252.139.1]) by bullwinkle.che.utexas.edu (8.6.10/8.6) with SMTP id QAA38495 for <help-octave@che.utexas.edu>; Tue, 6 Jun 1995 16:18:18 -0500
%% Received: from morpheus.wustl.edu by morpheus  with smtp
%% 	(Linux Smail3.1.28.1 #14) id m0sJ60v-000MrGC; Tue, 6 Jun 95 16:18 CDT
%% Message-Id: <m0sJ60v-000MrGC@morpheus>
%% X-Mailer: exmh version 1.5.3 12/28/94
%% To: help-octave@che.utexas.edu
%% Subject: Reading ASCII data of unknown length (summary)
%% Mime-Version: 1.0
%% Content-Type: text/plain; charset="us-ascii"
%% Date: Tue, 06 Jun 1995 16:18:16 -0500
%% From: Tim Kerwin <tkerwin@morpheus.wustl.edu>
%% Sender: help-octave-request@che.utexas.edu
%% 
%% Thanks to all who responded to my query about reading ASCII files of unknown 
%% length into octave.  Most of the solutions offered involved invoking "wc" on 
%% the file (via system()) in order to get the number of lines in the file.  Some 
%% proceed with something like:
%% 
%%    for i=1:nlines
%%       [x(i) ... ] = fscanf(...
%% 
%% but this is very slow.  Better appears to be to tack an octave header on to 
%% the ASCII file, then use "load", which is much faster than repeated fscanf.  
%% Here's the function "aload" (adapted from one sent by Eyal Doron 
%% <doron@mickey.mpi-hd.mpg.de> -- I just changed the temporary file name to use 
%% a process id number so users don't stomp on each other's files):
%% 
%% 