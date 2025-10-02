clear all
close all
addpath('/n/data2/hms/neurobio/livingstone/Code/data_loading_code_peter_branch')
addpath('/n/data2/hms/neurobio/livingstone/Code/matpl')
addpath('/n/data2/hms/neurobio/livingstone/marge/margemonkeys/complexities')
addpath('/n/data2/hms/neurobio/livingstone/Code/npy-matlab-master')
addpath('/n/data2/hms/neurobio/livingstone/Stimuli/fewerOO')
addpath(genpath('/n/data2/hms/neurobio/livingstone/Code/umapAndEppFileExchange_4_5'))
addpath('/n/data2/hms/neurobio/livingstone/Data/Ephys-Raw')

%% Parameters
% data locations
data_formatted = '/n/data2/hms/neurobio/livingstone/Data/Formatted/';
data_neuropixel = '/n/data2/hms/neurobio/livingstone/Data/Npx-Preprocessed/';
% addpath('./npy-matlab-master/npy-matlab/')
% [meta,rasters,lfps,Trials] = loadFormattedData('dat123879001.plx', 'expControlFN', '200201_red_screening_omniplex.bhv2', ...
%     'expControl','ML','equipment','PLEXON', 'rasterWindow',[0 300], 'savepsth',1,'alignToPhotodiode',0,'continuous',0);
image_dir = '/n/data2/hms/neurobio/livingstone/Stimuli/faceswap_4/';
addpath(genpath(image_dir));
% goodch=[1 3 10 12 23 36 45 61 62];
colorjet=colormap(jet);
%% Parameters
% data locations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fmt_data_dir = '/n/data2/hms/neurobio/livingstone/Data/Formatted/';

expa_name='paul_20250914';
exp_name = 'temp';

%% Load data

fmt_data_patha = fullfile(fmt_data_dir, [expa_name '_experiment.mat']);
load(fmt_data_patha)
rasters_patha = fullfile(fmt_data_dir, [expa_name '-rasters.h5']);
long_rastersa = h5read(rasters_patha, '/rasters');  % size (n_units, time_ms)
unit_namesa = h5read(rasters_patha, '/unit_names');  % size (n_units, 1)
Stimulia=Stimuli; clear Stimuli
goodch=[3 12 13 15 16 24 32 36 45 55 58 64];
%% Make rasters (clusters x time x presentations)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
fps=30;
frametime=1000/fps;
exp_name='temp';
%% Load data

%% Make rasters (units x time x presentations)
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
    % rasters=nan(size(goodch,1),10100,size(Stimulia,1));
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
             rastersa = long_rastersa(:,bin);
            % rastercount=rastercount+1;
            % rasters(:,1:size(rastersa,2),rastercount)=rastersa;
            clear alleyeposns
            for vframe=1:numframes_ML-1
                frameduration=round(frameTime_perframe_ML(1,vframe)):floor(frameTime_perframe_ML(1,vframe+1));
                for sitee=1:64
                    site=sitee;
                    firingrateperframe(sitee,thismoviecount,vframe)=squeeze(nanmean(nanmean(rastersa(site,frameduration),1),2));
                    firingrateinbin(sitee,thismoviecount,vframe,:)=squeeze(nanmean(rastersa(site,frameduration),1));
                end
                frameduration=round(frameTime_perframe_ML(1,vframe)):round(frameTime_perframe_ML(1,vframe))+33;
                fixlocs = eyeposns(frameduration,:);
                fixloc(thismoviecount,vframe,:)=nanmean(fixlocs,1);
            end
        end
    end

    if thismoviecount>0
        figure
        for sitee=1:8
            site=sitee;
            for mve1=1:thismoviecount
                subplot(4,2,sitee); hold on
                plot(smoothdata(squeeze(firingrateperframe(sitee,mve1,1:275)),'gaussian',[5 5]),'color',colorjet(round(mve1*255/thismoviecount),:),'linew',1)
            end
            subplot(4,2,sitee); hold on
            plot(smoothdata(squeeze(nanmean(firingrateperframe(sitee,:,1:275),2)),'gaussian',[5 5]),'k','linew',2)
            set(gca,'tickdir','out','linew',2); box on
            filename=([movienames{movieno},' sites1to8.jpg']);
            set(gca,'tickdir','out','linew',2); box on; axis off
            imtosave = getframe(gcf);
            imwrite(imtosave.cdata, ['/n/data2/hms/neurobio/livingstone/marge/figimages/',exp_name,'/',filename], 'jpg')
            % close all
        end
figure
        for sitee=9:16
            site=sitee;
            for mve1=1:thismoviecount
                subplot(4,2,sitee-8); hold on
                plot(smoothdata(squeeze(firingrateperframe(sitee,mve1,1:275)),'gaussian',[5 5]),'color',colorjet(round(mve1*255/thismoviecount),:),'linew',1)
            end
            subplot(4,2,sitee-8); hold on
            plot(smoothdata(squeeze(nanmean(firingrateperframe(sitee,:,1:275),2)),'gaussian',[5 5]),'k','linew',2)
            set(gca,'tickdir','out','linew',2); box on
            filename=([movienames{movieno},' sites9to16.jpg']);
            set(gca,'tickdir','out','linew',2); box on; axis off
            imtosave = getframe(gcf);
            imwrite(imtosave.cdata, ['/n/data2/hms/neurobio/livingstone/marge/figimages/',exp_name,'/',filename], 'jpg')
            % close all
        end
figure
for sitee=17:24
            site=sitee;
            for mve1=1:thismoviecount
                subplot(4,2,sitee-16); hold on
                plot(smoothdata(squeeze(firingrateperframe(sitee,mve1,1:275)),'gaussian',[5 5]),'color',colorjet(round(mve1*255/thismoviecount),:),'linew',1)
            end
            subplot(4,2,sitee-16); hold on
            plot(smoothdata(squeeze(nanmean(firingrateperframe(sitee,:,1:275),2)),'gaussian',[5 5]),'k','linew',2)
            set(gca,'tickdir','out','linew',2); box on
            filename=([movienames{movieno},' sites17to24.jpg']);
            set(gca,'tickdir','out','linew',2); box on; axis off
            imtosave = getframe(gcf);
            imwrite(imtosave.cdata, ['/n/data2/hms/neurobio/livingstone/marge/figimages/',exp_name,'/',filename], 'jpg')
            % close all
        end

figure
for sitee=25:32
            site=sitee;
            for mve1=1:thismoviecount
                subplot(4,2,sitee-24); hold on
                plot(smoothdata(squeeze(firingrateperframe(sitee,mve1,1:275)),'gaussian',[5 5]),'color',colorjet(round(mve1*255/thismoviecount),:),'linew',1)
            end
            subplot(4,2,sitee-24); hold on
            plot(smoothdata(squeeze(nanmean(firingrateperframe(sitee,:,1:275),2)),'gaussian',[5 5]),'k','linew',2)
            set(gca,'tickdir','out','linew',2); box on
            filename=([movienames{movieno},' sites25to32.jpg']);
            set(gca,'tickdir','out','linew',2); box on; axis off
            imtosave = getframe(gcf);
            imwrite(imtosave.cdata, ['/n/data2/hms/neurobio/livingstone/marge/figimages/',exp_name,'/',filename], 'jpg')
            % close all
        end

figure
for sitee=33:40
            site=sitee;
            for mve1=1:thismoviecount
                subplot(4,2,sitee-32); hold on
                plot(smoothdata(squeeze(firingrateperframe(sitee,mve1,1:275)),'gaussian',[5 5]),'color',colorjet(round(mve1*255/thismoviecount),:),'linew',1)
            end
            subplot(4,2,sitee-32); hold on
            plot(smoothdata(squeeze(nanmean(firingrateperframe(sitee,:,1:275),2)),'gaussian',[5 5]),'k','linew',2)
            set(gca,'tickdir','out','linew',2); box on
            filename=([movienames{movieno},' sites33to40.jpg']);
            set(gca,'tickdir','out','linew',2); box on; axis off
            imtosave = getframe(gcf);
            imwrite(imtosave.cdata, ['/n/data2/hms/neurobio/livingstone/marge/figimages/',exp_name,'/',filename], 'jpg')
            % close all
        end

figure
for sitee=41:48
            site=sitee;
            for mve1=1:thismoviecount
                subplot(4,2,sitee-40); hold on
                plot(smoothdata(squeeze(firingrateperframe(sitee,mve1,1:275)),'gaussian',[5 5]),'color',colorjet(round(mve1*255/thismoviecount),:),'linew',1)
            end
            subplot(4,2,sitee-40); hold on
            plot(smoothdata(squeeze(nanmean(firingrateperframe(sitee,:,1:275),2)),'gaussian',[5 5]),'k','linew',2)
            set(gca,'tickdir','out','linew',2); box on
            filename=([movienames{movieno},' sites41to48.jpg']);
            set(gca,'tickdir','out','linew',2); box on; axis off
            imtosave = getframe(gcf);
            imwrite(imtosave.cdata, ['/n/data2/hms/neurobio/livingstone/marge/figimages/',exp_name,'/',filename], 'jpg')
            % close all
        end


figure
for sitee=49:56
            site=sitee;
            for mve1=1:thismoviecount
                subplot(4,2,sitee-48); hold on
                plot(smoothdata(squeeze(firingrateperframe(sitee,mve1,1:275)),'gaussian',[5 5]),'color',colorjet(round(mve1*255/thismoviecount),:),'linew',1)
            end
            subplot(4,2,sitee-48); hold on
            plot(smoothdata(squeeze(nanmean(firingrateperframe(sitee,:,1:275),2)),'gaussian',[5 5]),'k','linew',2)
            set(gca,'tickdir','out','linew',2); box on
            filename=([movienames{movieno},' sites49to56.jpg']);
            set(gca,'tickdir','out','linew',2); box on; axis off
            imtosave = getframe(gcf);
            imwrite(imtosave.cdata, ['/n/data2/hms/neurobio/livingstone/marge/figimages/',exp_name,'/',filename], 'jpg')
            % close all
        end

figure
for sitee=57:64
            site=sitee;
            for mve1=1:thismoviecount
                subplot(4,2,sitee-56); hold on
                plot(smoothdata(squeeze(firingrateperframe(sitee,mve1,1:275)),'gaussian',[5 5]),'color',colorjet(round(mve1*255/thismoviecount),:),'linew',1)
            end
            subplot(4,2,sitee-56); hold on
            plot(smoothdata(squeeze(nanmean(firingrateperframe(sitee,:,1:275),2)),'gaussian',[5 5]),'k','linew',2)
            set(gca,'tickdir','out','linew',2); box on
            filename=([movienames{movieno},' sites57to64.jpg']);
            set(gca,'tickdir','out','linew',2); box on; axis off
            imtosave = getframe(gcf);
            imwrite(imtosave.cdata, ['/n/data2/hms/neurobio/livingstone/marge/figimages/',exp_name,'/',filename], 'jpg')
            % close all
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


        audiovideofilename=(['/n/data2/hms/neurobio/livingstone/marge/figimages/temp/250914-depths',movienames{movieno},'.avi']);
        writerObj = vision.VideoFileWriter(audiovideofilename,'AudioInputPort',true);
        for k = 1:1:numframes_ML-1
            tosave=squeeze(vidframe_short(:,:,:,k));
            num_bins = 1628;
            % % Generate a vector of uniform random numbers
            random_numbers = rand(1, num_bins);
            spike_train=interp(squeeze(firingrateinbinavg(k,:)),44);%was 44
            spike_train(spike_train<0.075)=0;
            spike_train(spike_train>=0.075)=1;
            step(writerObj,tosave,spike_train')
        end
        release(writerObj)


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
        imwrite(imtosave.cdata, ['/n/data2/hms/neurobio/livingstone/marge/figimages/',exp_name,'/',filename], 'jpg')
         close all


