clear all
close all
addpath("../src")
paths = get_paths()
addpath(sprintf('%s/Code/data-loading-code-peterbranch', paths.livingstone_lab))
addpath(sprintf('%s/Code/matpl', paths.livingstone_lab))
addpath(sprintf('%s/Code/npy-matlab-master', paths.livingstone_lab))
addpath(sprintf('%s/Stimuli/fewerOO', paths.livingstone_lab))
addpath(genpath(sprintf('%s/Code/umapAndEppFileExchange_4_5', paths.livingstone_lab)))
addpath(sprintf('%s/Data/Data-Ephys-Raw', paths.livingstone_lab))

%% Parameters
% data locations
data_formatted = sprintf('%s/Data/Data-Formatted/', paths.livingstone_lab);
data_neuropixel = sprintf('%s/Data/Data-Neuropixels-Preprocessed/', paths.livingstone_lab);
image_dir = sprintf('%s/Stimuli/faceswap_4/', paths.livingstone_lab);
addpath(genpath(image_dir));
colorjet=colormap(jet);
%% Parameters
% data locations
exp0_name = 'paul_250914';
exp_name = 'temp';
chanpos_exp_name = 'paul_250914';  % use a day when all 383 chans were present and IMRO table was the same

%% Load data

exp_path0 = fullfile(data_formatted,[exp0_name,'_experiment.mat']);
load(exp_path0)
mua0_path = fullfile(data_neuropixel,[exp0_name,'/catgt_',exp0_name,'_g0/',exp0_name,'_g0_imec1/',exp0_name,'-imec1-mua_cont.h5']);
mua0 = h5read(mua0_path, '/mua_cont');  % size (nchan x time_ms)
Stimulia=Stimuli;

clear Stimuli noisI



clear mua0_path mua11_path mua1_path mua2_path mua3_path mua4_path mua5_path mua6_path mua7_path mua9_path mua8_path mua10_path

chanpos_path = fullfile(data_neuropixel,[chanpos_exp_name,'/catgt_',chanpos_exp_name,'_g0/',chanpos_exp_name,'_g0_imec1/']);

load(fullfile(chanpos_path,'channel_positions.mat'));
sel = [1:191 193:384];
chan_pos2 = chan_pos(sel,:);
channel_depth = chan_pos2(:,2)/1e3;
[~, I] = sort(channel_depth);
channel_depth_sorted = channel_depth(I);

figure; plot(channel_depth_sorted)
filename='depths.jpg';
imtosave = getframe(gcf);
imwrite(imtosave.cdata, ['/Users/tizianocausin/figimages/',exp_name,'/',filename], 'jpg')
close all


%% Make rasters (clusters x time x presentations)
mua0=(mua0(I,:));
fps=30;
frametime=1000/fps;
exp_name='temp';
%% Load data

range=[1 40
    41 80
    81 120
    121 160
    161 200
    201 240
    241 280
    281 320
    321 360
    361 383
    ];

%% Make rasters (units x time x presentations)
eps = 10e-9;
for vidno=1:size(Stimulia,1)
    allmovienames{vidno}=Stimulia(vidno).filename;
end

movienames=unique(allmovienames);

rastercount=0;
for movieno=1:size(movienames,2)
    fn2load = sprintf('%s',image_dir,movienames{movieno});
    % load video header, and reset and prepare optical flow vector
    videoHeader = VideoReader(fn2load);
    numframes=0;
    while hasFrame(videoHeader) %loops over all the frames of the movie part
        frame=readFrame(videoHeader);
        numframes=numframes+1;
        frameTime_perframe_vh(numframes) = 1000*videoHeader.CurrentTime;
        vidframe(:,:,:,numframes)=imresize(frame,[108 192]);;
    end
    rasters=nan(383,10100,size(Stimulia,1));
    thismoviecount=0;
    fixloc=[];
    for videono=1:size(Stimulia,1)
        fixlocs = [0 0];
        if Trials(Stimulia(videono).trial_number).success==1
            trialend=Stimulia(videono).stop_time;
        else
            trialend=Trials(Stimulia(videono).trial_number).stop_time;
        end;
        trialstart=Stimulia(videono).start_time;
        if Trials(Stimulia(videono).trial_number).success==1 && strcmp(Stimulia(videono).filename, movienames(movieno))
            thismoviecount=thismoviecount+1;
            vidnos(thismoviecount)=videono;
            stimdelay=round(trialstart-Trials(Stimulia(videono).trial_number).start_time);
            stimduration=trialend-trialstart;
            startframe=1;
            alleyeposns= Trials(Stimulia(videono).trial_number).eye_data;
            alleyeposns(abs(alleyeposns)>20)=nan;
            eyeposns=alleyeposns(stimdelay:min([stimdelay+stimduration size(alleyeposns,1)]),:);
            endframe=find(frameTime_perframe_vh>=stimduration,1);
            bin = round(trialstart) : round(trialend);
            numframes_ML=endframe-startframe;
            frameTime_perframe_ML =1+ frameTime_perframe_vh(1,startframe:endframe)-frameTime_perframe_vh(1,1);
            rastersa = mua0(:,squeeze(bin));
            rastercount=rastercount+1;
            rasters(:,1:size(rastersa,2),rastercount)=rastersa;
            clear alleyeposns
            for vframe=1:numframes_ML-1
                frameduration=round(frameTime_perframe_ML(1,vframe)):floor(frameTime_perframe_ML(1,vframe+1)+eps);
                for rnge=1:10
                    firingrateperframe(rnge,thismoviecount,vframe)=squeeze(nanmean(nanmean(rastersa(range(rnge,1):range(rnge,2),frameduration),1),2));
                    firingrateinbin(rnge,thismoviecount,vframe,:)=squeeze(nanmean(rastersa(range(rnge,1):range(rnge,2),frameduration),1));
                end
                frameduration=round(frameTime_perframe_ML(1,vframe)):round(frameTime_perframe_ML(1,vframe))+33;
                fixlocs = eyeposns(frameduration,:);
                fixloc(thismoviecount,vframe,:)=nanmean(fixlocs,1);
            end
        end
    end

    if thismoviecount>0
        figure
        for rnge=1:10
            for mve1=1:thismoviecount
                subplot(5,2,rnge); hold on
                plot(smoothdata(squeeze(firingrateperframe(rnge,mve1,1:275)),'gaussian',[5 5]),'color',colorjet(round(mve1*255/thismoviecount),:),'linew',1)
            end
            subplot(5,2,rnge); hold on
            plot(smoothdata(squeeze(nanmean(firingrateperframe(rnge,:,1:275),2)),'gaussian',[5 5]),'k','linew',2)
            set(gca,'tickdir','out','linew',2); box on
            filename=([movienames{movieno},' byrange.jpg']);
            set(gca,'tickdir','out','linew',2); box on
            imtosave = getframe(gcf);
            imwrite(imtosave.cdata, ['/Users/tizianocausin/figimages/',exp_name,'/',filename], 'jpg')
        end
        firingrateperframeavg=median(firingrateperframe,1);
        firingrateinbinavg=squeeze(nanmean(nanmean(firingrateinbin,1),2));
        fixlocavg=squeeze(median(fixloc,2));
        for vino=1:thismoviecount
            vidframe_short=vidframe(:,:,:,1:numframes_ML);
            for vframe=1:numframes_ML-1
                xposn=round(96+fixloc(vino,vframe,1)*3.2);
                yposn=round(54-fixloc(vino,vframe,2)*3.2);
                if xposn<191 & xposn>1 &yposn>1 & yposn<107
                    vidframe_short(yposn-1:yposn+1,xposn-1:xposn+1,1,vframe)=255;
                    vidframe_short(yposn-1:yposn+1,xposn-1:xposn+1,2,vframe)=0;
                    vidframe_short(yposn-1:yposn+1,xposn-1:xposn+1,3,vframe)=0;
                end
            end
        end

        vidframe_short=vidframe(:,:,:,1:numframes_ML);
        for vidno=1:thismoviecount
            for vframe=1:numframes_ML-1
                xposn=round(96+fixloc(vidno,vframe,1)*3.2);
                yposn=round(54-fixloc(vidno,vframe,2)*3.2);
                if xposn<191 & xposn>1 &yposn>1 & yposn<107
                    vidframe_short(yposn-1:yposn+1,xposn-1:xposn+1,1,vframe)=255*colorjet(round(vidno*255/size(fixloc,2)),1);
                    vidframe_short(yposn-1:yposn+1,xposn-1:xposn+1,2,vframe)=255*colorjet(round(vidno*255/size(fixloc,2)),2);
                    vidframe_short(yposn-1:yposn+1,xposn-1:xposn+1,3,vframe)=255*colorjet(round(vidno*255/size(fixloc,2)),3);
                end
            end
        end


        % audiovideofilename=(['/Users/tizianocausin/figimages/temp/250914-depths/',movienames{movieno},'.avi']);
        % writerObj = vision.VideoFileWriter(audiovideofilename,'AudioInputPort',true);
        % for k = 1:1:numframes_ML-1
        %     tosave=squeeze(vidframe_short(:,:,:,k));
        %     num_bins = 1628;
        %     % Generate a vector of uniform random numbers
        %     random_numbers = rand(1, num_bins);
        %     spike_train=interp(squeeze(firingrateinbinavg(k,:)),44);
        %     spike_train(spike_train<0.075)=0;
        %     spike_train(spike_train>=0.075)=1;
        %     step(writerObj,tosave,spike_train')
        % end
        % release(writerObj)


        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        figure(5)
        for mve1=1:thismoviecount
            subplot(7,2,movieno); hold on
            plot(smoothdata(squeeze(fixloc(mve1,:,1)),'gaussian',[5 5]),smoothdata(squeeze(fixloc(mve1,:,2)),'gaussian',[5 5]),'color',colorjet(round(mve1*255/thismoviecount),:),'linew',1)
        end
        subplot(7,2,movieno); hold on
        plot(smoothdata(squeeze(nanmean(fixloc(:,:,1),1)),'gaussian',[5 5]),smoothdata(squeeze(nanmean(fixloc(:,:,2),1)),'gaussian',[5 5]),'k','linew',2)
    end
    clear firingrateinbinavg firingrateinbin firingrateperframeavg firingrateperframe
end
        filename=('eyeposns.jpg');
        set(gca,'tickdir','out','linew',2); box on
        imtosave = getframe(gcf);
        imwrite(imtosave.cdata, ['/Users/tizianocausin/figimages/',exp_name,'/',filename], 'jpg')
         

figure;
imagesc(squeeze(zscore(nanmean(rasters,3),[],2)))
xlabel('Time from stim on, ms')
ylabel('Channel depth, mm')
filename='rastersZbynum.png';
imtosave = getframe(gcf);
imwrite(imtosave.cdata, ['/Users/tizianocausin/figimages/',exp_name,'/',filename], 'jpg');
close gcf


figure;
imagesc(squeeze(nanmean(rasters,3)))
xlabel('Time from stim on, ms')
ylabel('Channel depth, mm')
filename=('rastersbynum.png');
imtosave = getframe(gcf);
imwrite(imtosave.cdata, ['/Users/tizianocausin/figimages/',exp_name,'/',filename], 'jpg');
close gcf
