function percentScore=CVC(subjID,howmany,audflag,catchflag)
% e.g. type: CVC('test');      for default 84 trials (1 run) with
%                                  default no feedback no catch trials
%  or type: CV4ns6r('test',10);    for 10 trials (incomplete)
%  or type: CVC('test',108,10,0);    to play stimuli dichotically (default)
%  or type: CV4ns6r('test',108,1);    to correct for HL thresholds
%
%Modified xxxx

if nargin<4
    catchflag=0;
    maxN=84;
else
    if catchflag==1
        maxN=108;   %add 16 diotic catch stimuli (8 at ~100 and ~150 F0)
    else
        maxN=84;
    end
end
if nargin<3, 
    audflag=0;
end
if nargin<2, howmany=maxN; end
    
wh=cd; 
global shp;
close all;

%Instructions 
%Instructions(8);

fs=44100;
bits=24; %double check number of bits in CVC stimuli and set correctly

%whichVow=['ah'; 'ae'; 'ee'; 'oo'];
whichVow=['AH'; 'AE'; 'EE'; 'OO'; 'xx'; 'xx'; 'xx'; 'xx'; '10'; '14'; '26'; '35'; '52'; '56'; '72'; '78'];
%whichVowR=['AH'; 'UH'; 'OO'; 'OU'; 'AE'; 'EH'; 'IH'; 'EE'];  %Added 6/14/2019
whichVowR=['AH'; 'UH'; 'OO'; 'AE'; 'IH'; 'EE'];   %Modified 6/27/2023 LR/KP/MM for 6 vowel version
%whichF0=['120_0'; '151_2'];
%whichF0=['120_0'; '127_1'];
%whichF0=['106.9'; '151.2'; '201.8'];  %Modified 7/3/19
%whichF0=['106.9'; '151.2'; '201.8'; '109.0'; '155.0'];  %Modified 7/3/19
whichF0=['106.9'; '151.2'; '201.8'; '109.0'; '155.0'];  %Modified 6/27/2023 MM & LR
if catchflag==0
    stimset(:,1)=[1*ones(15,1); 2*ones(15,1); 3*ones(15,1); 4*ones(15,1); 1*ones(6,1); 2*ones(6,1); 3*ones(6,1); 4*ones(6,1)];
    stimset(:,2)=[repmat([ones(9,1); [2 3 2 3 2 3]'],4,1); repmat([2 2 2 3 3 3]',4,1)];
    stimset(:,3)=[2 2 2 3 3 3 4 4 4 2 2 3 3 4 4 1 1 1 3 3 3 4 4 4 1 1 3 3 4 4 1 1 1 2 2 2 4 4 4 1 1 2 2 4 4 1 1 1 2 2 2 3 3 3 1 1 2 2 3 3 2 3 4 2 3 4 1 3 4 1 3 4 1 2 4 1 2 4 1 2 3 1 2 3]';
    stimset(:,4)=[repmat([1 2 3 1 2 3 1 2 3 1 1 1 1 1 1]',4,1); repmat([2 2 2 3 3 3]',4,1)];
elseif catchflag==1
    stimset(:,1)=[1*ones(15,1); 2*ones(15,1); 3*ones(15,1); 4*ones(15,1); 1*ones(6,1); 2*ones(6,1); 3*ones(6,1); 4*ones(6,1); repmat([9:16]',3,1)];
    stimset(:,2)=[repmat([ones(9,1); [2 3 2 3 2 3]'],4,1); repmat([2 2 2 3 3 3]',4,1); 1*ones(8,1); 2*ones(8,1); 3*ones(8,1)];
    stimset(:,3)=[2 2 2 3 3 3 4 4 4 2 2 3 3 4 4 1 1 1 3 3 3 4 4 4 1 1 3 3 4 4 1 1 1 2 2 2 4 4 4 1 1 2 2 4 4 1 1 1 2 2 2 3 3 3 1 1 2 2 3 3 2 3 4 2 3 4 1 3 4 1 3 4 1 2 4 1 2 4 1 2 3 1 2 3 repmat([9:16]',3,1)']';
    stimset(:,4)=[repmat([1 2 3 1 2 3 1 2 3 1 1 1 1 1 1]',4,1); repmat([2 2 2 3 3 3]',4,1); 1*ones(8,1); 2*ones(8,1); 3*ones(8,1)];
end

%soundPath='/SoundFiles/ConcurrVowels';
soundPath='/SoundFiles/ConcurrVowels/New8vowels/';    %Modified 7/3/2019
pathSave=['/Experiments/Data/' subjID '/'];
controlCh='%s  %5.2f%%  %s, %2d:%2d:%2d.  %d\n';

if exist(pathSave)~=7   %if file directory doesn't exist, create new dir
    success=mkdir(['/Experiments/Data/'],subjID);
    if success==0
        disp('Create directory failed!  Aborting...'); 
        return; 
    end
end
feval('cd',pathSave);


scalefactor=10^(-31/20);
soundsystem='Babyface + Headphones';


rand('state',sum(100*clock)); %for random seed resetting
pickaNum=mod(randperm(ceil(maxN)),maxN)+1;
scoreCum=0;
telapsedCum=0;

if audflag==1
    %load frequency range and attenuations for both ears from file here
    fidR=fopen(sprintf('/Experiments/Data/%s/Audiogram.txt',subjID));
    freqread=fscanf(fidR,'%i',[8,1])';
    aud=fscanf(fidR,'%f',[8,2])';    %first row is left, second row is right
    fclose(fidR);
end

flist=dir; 
flist=flist(3:end); 
subjID=GetFolder(subjID);
outFile=fileExistCheck(flist,[subjID '_cvc_0.txt']); %updated 7/3/2019
fid=fopen(outFile,'wt');
% if audflag==1
%     fprintf(fid,'Audiogram left input: [%i %i %i %i %i %i %i %i]\n',aud(:,1));
%     fprintf(fid,'Audiogram right input: [%i %i %i %i %i %i %i %i]\n',aud(:,2));
% else
%     fprintf(fid,'Audiogram left input: NA\n');
%     fprintf(fid,'Audiogram right input: NA\n');
% end
%Code fixed 3/24/2017 LR to document attenuation used and fix audiogram
%file bug
fprintf(fid,'Sound system: %s',soundsystem);
if audflag==1
    fprintf(fid,'Audiogram left input: [');
    for i=1:8
        fprintf(fid,'%i ',aud(1,i));
    end
    fprintf(fid,']\n');
    fprintf(fid,'Audiogram right input: [');
    for i=1:8
        fprintf(fid,'%i ',aud(2,i));
    end
    fprintf(fid,']\n');
else
    fprintf(fid,'Audiogram left input: NA\n');
    fprintf(fid,'Audiogram right input: NA\n');
end
fprintf(fid,'vowID1 vowF02 vowID1 vowF02 answer1 answer2 ncorrect nrep time_elapsed\n');
fprintf('saved to %s \n',[pathSave outFile])


disp('Starting by subject''s pressing a button...')
cr=status_bar;  % initializing the current run status bar
%loading and executing the "new" GUI derived from matlab:
load C:/Development/Matlab/ark1.mat
h=ark1v6;

buttonv6(9).name='Press any button to start.';
bcontrol(h,1,buttonv6,9,'w',20);    

shp=0;
waitButton; %light_response2((1:24),0); 

 
bcontrol(h,6,buttonv6,0,blue,40);  %turn all on
pause(1);
bcontrol(h,6,buttonv6,0,red,30); % turn all off
pause(1);
	
N=[0 0 0];
ncorrect=[0 0 0];   %total correct
ncorrect1=[0 0 0];  %how many times only 1 correct
ncorrect2=[0 0 0];  %how many times both correct
%trial begins...
for nn=1:howmany
	set(cr,'String',num2str(nn));
	%matCom(7) %hide cursor
 	soundNameL=[soundPath whichVow(stimset(pickaNum(nn),1),:) '_' whichF0(stimset(pickaNum(nn),2),:) '.wav'];
 	soundNameR=[soundPath whichVow(stimset(pickaNum(nn),3),:) '_' whichF0(stimset(pickaNum(nn),4),:) '.wav'];    
	%yL=wavread(soundNameL);
	%yR=wavread(soundNameR);
    yL=audioread(soundNameL);
	yR=audioread(soundNameR);

    yL=[zeros(2000,1); yL; zeros(2000,1)];
    yR=[zeros(2000,1); yR; zeros(2000,1)];
    if audflag==1
        [targetL rms_y db_y] = ampstim(yL, fs, aud(1,:));
        [targetR rms_y db_y] = ampstim(yR, fs, aud(2,:));
        targetL=targetL/(10^(28/20));
        targetR=targetR/(10^(28/20));
        %Code added by LR/SS 11/28/2016 as safeguard for too large wave files for
        %too big hearing losses
        if max(abs(targetL))>1 | max(abs(targetR))>1
            disp('WARNING!!!  Wave file will exceed allowable values of +-1.  Please use audiogram with lower values!  ');
        end
    elseif audflag==0
        targetL=yL;
        targetR=yR;
    end
    z=[targetL targetR];
	%y=target*1.982;
    z=z*scalefactor;    %6/8/2022 reduce by 31 dB only if using Babyface

    buttonv6(9).name=sprintf('Playing trial %i of %i...',nn,howmany);
    bcontrol(h,1,buttonv6,9,'w',20);    
    pause(0.7);
	%sound(z,fs)
    a=audioplayer(z,fs,bits);   %change to sound instead of audioplayer/play for 24 bits
    playblocking(a);

	pause(0.3);
        
	vowIDL = transID(stimset(pickaNum(nn),1));     %Modified 7/3/2019 to convert vowID to number within response set
    vowF0L = stimset(pickaNum(nn),2);
    vowIDR = transID(stimset(pickaNum(nn),3));     %Modified 7/3/2019 to convert vowID to number within response set
    vowF0R = stimset(pickaNum(nn),4);
		
		
	%matCom(7); %hide cursor		
    buttonv6(9).name='Which vowel did you hear?';
    %buttonv6(7).name=sprintf('Vowels %i, %i',vowIDL,vowIDR);;
    %bcontrol(h,1,buttonv6,7,'w',20);    
    bcontrol(h,1,buttonv6,9,'w',20);    %Modified 6/14/2019
 	shp=0; 
	tstart=tic;     %Added 10/3/2011 to save time elapsed during data collection - LR
    
    %vowelstatus=zeros(1,4);
    %vowelstatus=zeros(1,8); %Modified 6/14/2019
    vowelstatus=zeros(1,6); %Modified 6/27/2023 for 6 options, LR/KP/MM
    shpdone=0;
    nrepeat=0;
    while (shpdone==0)
        waitButton;
        answer=shp;
        %if answer==6    
        if answer==8    %Modified 6/14/2019
            answerind=find(vowelstatus==1);
            if length(answerind)<1 | length(answerind)>2
                if length(answerind)<1
                    buttonv6(9).name='Please pick at least 1 vowel!';   %Modified 6/14/2019
                    bcontrol(h,1,buttonv6,9,'w',20);   %Modified 6/14/2019
                elseif length(answerind)>2
                    buttonv6(9).name='Do not pick more than 2 vowels! Pick 1 or 2.';  %Modified 6/14/2019
                    bcontrol(h,1,buttonv6,9,'w',20);   %Modified 6/14/2019
                end
            else %only allow to exit if 1 or 2 vowels selected
                telapsed=toc(tstart);   %Added 10/3/2011 to save time elapsed during data collection - LR
                shpdone=1;
                bcontrol(h,6,buttonv6,0,red,30); % turn all off
            end
        %elseif answer<=4
        %elseif answer<=8    %Modified 6/14/2019
        elseif answer<=6    %Modified 6/27/2023 for 6 options, LR/KP/MM
            if vowelstatus(answer)==0  %change button color to blue for correct
                bcontrol(h,1,buttonv6,answer,blue,30);    
                vowelstatus(answer)=1;
            else vowelstatus(answer)==1  %change button color back to red for incorrect
                bcontrol(h,1,buttonv6,answer,red,30);    
                vowelstatus(answer)=0;
            end
        %elseif answer==5            
        elseif answer==7            %Modified 6/14/2019
            if nrepeat<2    %Modified 7/8/2022 to limit to 2 repeats
                pause(0.3);
                sound(z,fs);
                nrepeat=nrepeat+1;
            else
                buttonv6(9).name='No more than 2 repeats permitted.';  %Added 7/8/2022
                bcontrol(h,1,buttonv6,9,'w',20);   %     
            end
        end
    end
    f0diff=abs(vowF0L-vowF0R);
    if f0diff<=0, 
        f0diffind=1;
    elseif f0diff<=1
        f0diffind=2;
    else
        f0diffind=3;
    end
    N(f0diffind)=N(f0diffind)+1;
    
    answer1=answerind(1);
        
    if length(answerind)<2
        answer2=0;
    else
        answer2=answerind(2);
    end
    
    %Modified 6/14/2019 to convert answer to original format
    ncorrectVowIDL=0;
    ncorrectVowIDR=0;
    correct=0;
    if answer1==vowIDL | answer2==vowIDL
        ncorrect(f0diffind)=ncorrect(f0diffind)+1;
        ncorrectVowIDL=1;
        correct=correct+1;
    end
    if answer1==vowIDR | answer2==vowIDR
        ncorrect(f0diffind)=ncorrect(f0diffind)+1;
        ncorrectVowIDR=1;
        correct=correct+1;
    end
    if ncorrectVowIDL==1 & ncorrectVowIDR==1
        ncorrect2(f0diffind)=ncorrect2(f0diffind)+1;
    elseif (ncorrectVowIDL==1 & ncorrectVowIDR==0) | (ncorrectVowIDL==0 & ncorrectVowIDR==1)
        ncorrect1(f0diffind)=ncorrect1(f0diffind)+1;
    end

    
   	fprintf(fid,'%3d\t %3d\t %3d\t %3d\t %3d\t %3d\t %i\t %i\t %.4f\n',vowIDL,...     %Updated 10/3/2011 to save time data
        vowF0L,vowIDR,vowF0R,answer1, answer2, correct,nrepeat, telapsed);
    telapsedCum=telapsedCum+telapsed;
    
	
end

buttonv6(9).name=sprintf('Run finished.');   %Modified 6/14/2019
bcontrol(h,1,buttonv6,9,'w',20);    %Modified 6/14/2019

percentScore=ncorrect./N/2*100;
totalScore=sum(ncorrect)/sum(N)/2*100;
responseTime=telapsedCum/howmany;
fprintf(1,'Score is %5.2f%%, %5.2f%%, %5.2f%% for f0 difference = 0, 50, and 105 Hz\n',percentScore(1), percentScore(2), percentScore(3));
fprintf(1,'Total score is %5.2f%%\n',totalScore);

percentScore=ncorrect2./N*100;
totalScore=sum(ncorrect2)/sum(N)*100;
fprintf(1,'Percent of both correct is %5.2f%%, %5.2f%%, %5.2f%% for f0 difference = 0, 50, and 105 Hz\n',percentScore(1), percentScore(2), percentScore(3));
fprintf(1,'Total percent both correct is %5.2f%%\n',totalScore);

percentScore=ncorrect1./N*100;
totalScore=sum(ncorrect1)/sum(N)*100;
fprintf(1,'Percent of 1/2 correct is %5.2f%%, %5.2f%%, %5.2f%% for f0 difference = 0, 50, and 105 Hz\n',percentScore(1), percentScore(2), percentScore(3));
fprintf(1,'Total percent 1/2 correct is %5.2f%%\n',totalScore);


fprintf(1,'Average response time is %5.2f sec\n',responseTime);
cl=clock;


	
fclose('all');
	
feval('cd',wh); 

% end of experiment
for i=1:3
       bcontrol(h,6,buttonv6,0,blue,30);  %turn all on	%Modified 6/14/2019
       pause(0.8);
       bcontrol(h,6,buttonv6,0,red,20); % turn all off  %Modified 6/14/2019
       pause(0.8);
end

%matCom(8); %show cursor 
close all;

closereq;
closereq;
	
clear;

function NewID=transID(ID)

% whichVow=['AW'; 'AE'; 'EE'; 'OO'];
% whichVowR=['AH'; 'UH'; 'OO'; 'OU'; 'AE'; 'EH'; 'IH'; 'EE'];  %Added 6/14/2019
% if ID==1
%     NewID=1;
% elseif ID==2
%     NewID=5;
% elseif ID==3
%     NewID=8;
% elseif ID==4
%     NewID=3;
% elseif ID>4
%     NewID=ID;
% end

% whichVow=['AW'; 'AE'; 'EE'; 'OO'];
% whichVowR=['AH'; 'UH'; 'OO'; 'AE'; 'IH'; 'EE'];   %Modified 6/27/2023 LR/KP/MM for 6 vowel version
if ID==1
    NewID=1;
elseif ID==2
    NewID=4;
elseif ID==3
    NewID=6;
elseif ID==4
    NewID=3;
elseif ID>4
    NewID=ID; % ID>4 accounts for catch stimuli. 06/27/2023 MM    
end

return

