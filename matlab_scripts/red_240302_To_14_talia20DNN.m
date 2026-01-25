clear all
close all
cfg = [];
cfg.livingstone_lab = "/Volumes/LivingstoneLab";
image_dir = sprintf('%s/Stimuli/talia_20each/', cfg.livingstone_lab);
addpath(genpath(image_dir));

bkgwindow=65:85;
evkwindow=110:440;
bkgwindow1=65:85;
evkwindow1=110:440;
bkgwindow2=65:85;
evkwindow2=110:440;

rasterlength=550;
binsize=10;
myflag.categorieseachsite=0;
myflag.montage=0;
myflag.montagegoodch=0;
myflag.categorymontages=0;
myflag.rollingaverage=0;
myflag.rollingaverage1=0;
myflag.PCAcluster=0;
myflag.plotUmap=0;
myflag.commonnoise=0;
myflag.plotcategories=0;
fixation=[0 1];
myflag.rolling2=0;
myflag.split=0;
myflag.rolling3=0;
myflag.plotsigmoids=0;
myflag.plottimes=0
myflag.rollingaveragebypos=0;
myflag.plotsizeposn=0;
myflag.categories=0;
myflag.UMAP=0;
myflag.Umap=0;
myflag.colorbal=0;
myflag.montagebyposn=0;
myflag.rolling1=0;
myflag.xcorrbyposn=0;
myflag.rollingrank=0;
myflag.times=0;


%% Parameters
% data locations
fmt_data_dir  = sprintf('%s/Data/Data-Formatted/', cfg.livingstone_lab);
raster_window = [1 550];
expa_name='red_20240302';
expb_name='red_20240304';
expc_name='red_20240305';
expd_name='red_20240306';
expe_name='red_20240307';
expf_name='red_20240308';
expg_name='red_20240309';
exph_name='red_20240310';
expi_name='red_20240311';
expj_name='red_20240312';
expk_name='red_20240313';
expl_name='red_20240314';

exp_name='red_240302_to_14_DNN';
%% Load data
% - task + stimuli info
fmt_data_patha = fullfile(fmt_data_dir, [expa_name '_experiment.mat']);
load(fmt_data_patha)
rasters_patha = fullfile(fmt_data_dir, [expa_name '-rasters.h5']);
long_rastersa = h5read(rasters_patha, '/rasters');  % size (n_units, time_ms)
unit_namesa = h5read(rasters_patha, '/unit_names');  % size (n_units, 1)
Stimulia=Stimuli; clear Stimuli

fmt_data_pathb = fullfile(fmt_data_dir, [expb_name '_experiment.mat']);
load(fmt_data_pathb)
rasters_pathb = fullfile(fmt_data_dir, [expb_name '-rasters.h5']);
long_rastersb = h5read(rasters_pathb, '/rasters');  % size (n_units, time_ms)
unit_namesb = h5read(rasters_pathb, '/unit_names');  % size (n_units, 1)
Stimulib=Stimuli; clear Stimuli

fmt_data_pathc = fullfile(fmt_data_dir, [expc_name '_experiment.mat']);
load(fmt_data_pathc)
rasters_pathc = fullfile(fmt_data_dir, [expc_name '-rasters.h5']);
long_rastersc = h5read(rasters_pathc, '/rasters');  % size (n_units, time_ms)
unit_namesc = h5read(rasters_pathc, '/unit_names');  % size (n_units, 1)
Stimulic=Stimuli; clear Stimuli

fmt_data_pathd = fullfile(fmt_data_dir, [expd_name '_experiment.mat']);
load(fmt_data_pathd)
rasters_pathd = fullfile(fmt_data_dir, [expd_name '-rasters.h5']);
long_rastersd = h5read(rasters_pathd, '/rasters');  % size (n_units, time_ms)
unit_namesd = h5read(rasters_pathd, '/unit_names');  % size (n_units, 1)
Stimulid=Stimuli; clear Stimuli

fmt_data_pathe = fullfile(fmt_data_dir, [expe_name '_experiment.mat']);
load(fmt_data_pathe)
rasters_pathe = fullfile(fmt_data_dir, [expe_name '-rasters.h5']);
long_rasterse = h5read(rasters_pathe, '/rasters');  % size (n_units, time_ms)
unit_namese = h5read(rasters_pathe, '/unit_names');  % size (n_units, 1)
Stimulie=Stimuli; clear Stimuli

fmt_data_pathf = fullfile(fmt_data_dir, [expf_name '_experiment.mat']);
load(fmt_data_pathf)
rasters_pathf = fullfile(fmt_data_dir, [expf_name '-rasters.h5']);
long_rastersf = h5read(rasters_pathf, '/rasters');  % size (n_units, time_ms)
unit_namesf = h5read(rasters_pathf, '/unit_names');  % size (n_units, 1)
Stimulif=Stimuli; clear Stimuli
%
fmt_data_pathg = fullfile(fmt_data_dir, [expg_name '_experiment.mat']);
load(fmt_data_pathg)
rasters_pathg = fullfile(fmt_data_dir, [expg_name '-rasters.h5']);
long_rastersg = h5read(rasters_pathg, '/rasters');  % size (n_units, time_ms)
unit_namesg = h5read(rasters_pathg, '/unit_names');  % size (n_units, 1)
Stimulig=Stimuli; clear Stimuli

fmt_data_pathh = fullfile(fmt_data_dir, [exph_name '_experiment.mat']);
load(fmt_data_pathh)
rasters_pathh = fullfile(fmt_data_dir, [exph_name '-rasters.h5']);
long_rastersh = h5read(rasters_pathh, '/rasters');  % size (n_units, time_ms)
unit_namesh = h5read(rasters_pathh, '/unit_names');  % size (n_units, 1)
Stimulih=Stimuli; clear Stimuli

%
fmt_data_pathi = fullfile(fmt_data_dir, [expi_name '_experiment.mat']);
load(fmt_data_pathi)
rasters_pathi = fullfile(fmt_data_dir, [expi_name '-rasters.h5']);
long_rastersi = h5read(rasters_pathi, '/rasters');  % size (n_units, time_ms)
unit_namesi = h5read(rasters_pathi, '/unit_names');  % size (n_units, 1)
Stimulii=Stimuli; clear Stimuli

fmt_data_pathj = fullfile(fmt_data_dir, [expj_name '_experiment.mat']);
load(fmt_data_pathj)
rasters_pathj = fullfile(fmt_data_dir, [expj_name '-rasters.h5']);
long_rastersj = h5read(rasters_pathj, '/rasters');  % size (n_units, time_ms)
unit_namesj = h5read(rasters_pathj, '/unit_names');  % size (n_units, 1)
Stimulij=Stimuli; clear Stimuli

fmt_data_pathk = fullfile(fmt_data_dir, [expk_name '_experiment.mat']);
load(fmt_data_pathk)
rasters_pathk = fullfile(fmt_data_dir, [expk_name '-rasters.h5']);
long_rastersk = h5read(rasters_pathk, '/rasters');  % size (n_units, time_ms)
unit_namesk = h5read(rasters_pathk, '/unit_names');  % size (n_units, 1)
Stimulik=Stimuli; clear Stimuli

fmt_data_pathl = fullfile(fmt_data_dir, [expl_name '_experiment.mat']);
load(fmt_data_pathl)
rasters_pathl = fullfile(fmt_data_dir, [expl_name '-rasters.h5']);
long_rastersl = h5read(rasters_pathl, '/rasters');  % size (n_units, time_ms)
unit_namesl = h5read(rasters_pathl, '/unit_names');  % size (n_units, 1)
Stimulil=Stimuli; clear Stimuli

%% Make rasters (units x time x presentations)
n_presentationsa = length(Stimulia);
n_unitsa = size(long_rastersa, 1);
window_length = diff(raster_window) + 1;
rastersa = zeros(n_unitsa, window_length, n_presentationsa, 'single');
for i = 1:n_presentationsa
    win_ = round(Stimulia(i).start_time + raster_window(1));
    rastersa(:,:,i) = long_rastersa(:,win_:win_+window_length-1);
end

n_presentationsb = length(Stimulib);
n_unitsb = size(long_rastersb, 1);
window_length = diff(raster_window) + 1;
rastersb = zeros(n_unitsb, window_length, n_presentationsb, 'single');
for i = 1:n_presentationsb
    win_ = round(Stimulib(i).start_time + raster_window(1));
    rastersb(:,:,i) = long_rastersb(:,win_:win_+window_length-1);
end

n_presentationsc = length(Stimulic);
n_unitsc = size(long_rastersc, 1);
window_length = diff(raster_window) + 1;
rastersc = zeros(n_unitsc, window_length, n_presentationsc, 'single');
for i = 1:n_presentationsc
    win_ = round(Stimulic(i).start_time + raster_window(1));
    rastersc(:,:,i) = long_rastersc(:,win_:win_+window_length-1);
end

n_presentationsd = length(Stimulid);
n_unitsd = size(long_rastersd, 1);
window_length = diff(raster_window) + 1;
rastersd = zeros(n_unitsd, window_length, n_presentationsd, 'single');
for i = 1:n_presentationsd
    win_ = round(Stimulid(i).start_time + raster_window(1));
    rastersd(:,:,i) = long_rastersd(:,win_:win_+window_length-1);
end

n_presentationse = length(Stimulie);
n_unitse = size(long_rasterse, 1);
window_length = diff(raster_window) + 1;
rasterse = zeros(n_unitse, window_length, n_presentationse, 'single');
for i = 1:n_presentationse
    win_ = round(Stimulie(i).start_time + raster_window(1));
    rasterse(:,:,i) = long_rasterse(:,win_:win_+window_length-1);
end

n_presentationsf = length(Stimulif);
n_unitsf = size(long_rastersf, 1);
window_length = diff(raster_window) + 1;
rastersf = zeros(n_unitsf, window_length, n_presentationsf, 'single');
for i = 1:n_presentationsf
    win_ = round(Stimulif(i).start_time + raster_window(1));
    rastersf(:,:,i) = long_rastersf(:,win_:win_+window_length-1);
end

n_presentationsg = length(Stimulig);
n_unitsg = size(long_rastersg, 1);
window_length = diff(raster_window) + 1;
rastersg = zeros(n_unitsg, window_length, n_presentationsg, 'single');
for i = 1:n_presentationsg
    win_ = round(Stimulig(i).start_time + raster_window(1));
    rastersg(:,:,i) = long_rastersg(:,win_:win_+window_length-1);
end

n_presentationsh = length(Stimulih);
n_unitsh = size(long_rastersh, 1);
window_length = diff(raster_window) + 1;
rastersh = zeros(n_unitsh, window_length, n_presentationsh, 'single');
for i = 1:n_presentationsh
    win_ = round(Stimulih(i).start_time + raster_window(1));
    rastersh(:,:,i) = long_rastersh(:,win_:win_+window_length-1);
end

n_presentationsi = length(Stimulii);
n_unitsi = size(long_rastersi, 1);
window_length = diff(raster_window) + 1;
rastersi = zeros(n_unitsi, window_length, n_presentationsi, 'single');
for i = 1:n_presentationsi
    win_ = round(Stimulii(i).start_time + raster_window(1));
    rastersi(:,:,i) = long_rastersi(:,win_:win_+window_length-1);
end

n_presentationsj = length(Stimulij);
n_unitsj = size(long_rastersj, 1);
window_length = diff(raster_window) + 1;
rastersj = zeros(n_unitsj, window_length, n_presentationsj, 'single');
for i = 1:n_presentationsj
    win_ = round(Stimulij(i).start_time + raster_window(1));
    rastersj(:,:,i) = long_rastersj(:,win_:win_+window_length-1);
end

n_presentationsk = length(Stimulik);
n_unitsk = size(long_rastersk, 1);
window_length = diff(raster_window) + 1;
rastersk = zeros(n_unitsk, window_length, n_presentationsk, 'single');
for i = 1:n_presentationsk
    win_ = round(Stimulik(i).start_time + raster_window(1));
    rastersk(:,:,i) = long_rastersk(:,win_:win_+window_length-1);
end

n_presentationsl = length(Stimulil);
n_unitsl = size(long_rastersl, 1);
window_length = diff(raster_window) + 1;
rastersl = zeros(n_unitsl, window_length, n_presentationsl, 'single');
for i = 1:n_presentationsl
    win_ = round(Stimulil(i).start_time + raster_window(1));
    rastersl(:,:,i) = long_rastersl(:,win_:win_+window_length-1);
end


stim_ima={Stimulia(:).filename};
stim_imb={Stimulib(:).filename};
stim_imc={Stimulic(:).filename};
stim_imd={Stimulid(:).filename};
stim_ime={Stimulie(:).filename};
stim_imf={Stimulif(:).filename};
stim_img={Stimulig(:).filename};
stim_imh={Stimulih(:).filename};
stim_imi={Stimulii(:).filename};
stim_imj={Stimulij(:).filename};
stim_imk={Stimulik(:).filename};
stim_iml={Stimulil(:).filename};

allimages=[stim_ima stim_imb stim_imc stim_imd stim_ime stim_imf stim_img stim_imh stim_imi stim_imj  stim_imk stim_iml];
allimagesA=[stim_ima stim_imb stim_imc stim_imd stim_ime stim_imf ];
allimagesB=[ stim_img stim_imh stim_imi stim_imj stim_imk stim_iml];
clear stim_ima stim_imb stim_imc stim_imd stim_ime stim_imf stim_img stim_imh stim_imh stim_imi stim_imj stim_imk stim_iml
clear stim_imm stim_imn stim_imo stim_imp stim_imq stim_imr stim_ims stim_imt stim_imu stim_imv

clear stim_xya stim_xyb stim_xyc stim_xyd stim_xye stim_xyf stim_xyg stim_xyh stim_xyh stim_xyi stim_xyj stim_xyk stim_xyl
clear  stim_xym stim_xyn stim_xyo stim_xyp stim_xyq stim_xyr stim_xys stim_xyt stim_xyu stim_xyv

clear stim_xy0 stim_xy1 stim_xy2 stim_xy3 stim_xy4 stim_xy5 stim_xy6 stim_xy7
rastersambkg1=rastersa-permute(repmat(smoothdata(squeeze(nanmean(rastersa,2)),2,'gaussian', [50 50]),[1,1,size(rastersa,2)]),[1,3,2]);;
rastersbmbkg1=rastersb-permute(repmat(smoothdata(squeeze(nanmean(rastersb,2)),2,'gaussian', [50 50]),[1,1,size(rastersb,2)]),[1,3,2]);;
rasterscmbkg1=rastersc-permute(repmat(smoothdata(squeeze(nanmean(rastersc,2)),2,'gaussian', [50 50]),[1,1,size(rastersc,2)]),[1,3,2]);;
rastersdmbkg1=rastersd-permute(repmat(smoothdata(squeeze(nanmean(rastersd,2)),2,'gaussian', [50 50]),[1,1,size(rastersd,2)]),[1,3,2]);;
rastersembkg1=rasterse-permute(repmat(smoothdata(squeeze(nanmean(rasterse,2)),2,'gaussian', [50 50]),[1,1,size(rasterse,2)]),[1,3,2]);;
rastersfmbkg1=rastersf-permute(repmat(smoothdata(squeeze(nanmean(rastersf,2)),2,'gaussian', [50 50]),[1,1,size(rastersf,2)]),[1,3,2]);;
rastersgmbkg1=rastersg-permute(repmat(smoothdata(squeeze(nanmean(rastersg,2)),2,'gaussian', [50 50]),[1,1,size(rastersg,2)]),[1,3,2]);;
rastershmbkg1=rastersh-permute(repmat(smoothdata(squeeze(nanmean(rastersh,2)),2,'gaussian', [50 50]),[1,1,size(rastersh,2)]),[1,3,2]);;
rastersimbkg1=rastersi-permute(repmat(smoothdata(squeeze(nanmean(rastersi,2)),2,'gaussian', [50 50]),[1,1,size(rastersi,2)]),[1,3,2]);;
rastersjmbkg1=rastersj-permute(repmat(smoothdata(squeeze(nanmean(rastersj,2)),2,'gaussian', [50 50]),[1,1,size(rastersj,2)]),[1,3,2]);;
rasterskmbkg1=rastersk-permute(repmat(smoothdata(squeeze(nanmean(rastersk,2)),2,'gaussian', [50 50]),[1,1,size(rastersk,2)]),[1,3,2]);;
rasterslmbkg1=rastersl-permute(repmat(smoothdata(squeeze(nanmean(rastersl,2)),2,'gaussian', [50 50]),[1,1,size(rastersl,2)]),[1,3,2]);;

rastersambkg=rastersambkg1-repmat(nanmean(nanmean(rastersambkg1(:,bkgwindow1,:),3),2),[1,size(rastersa,2),size(rastersa,3)]);
rastersbmbkg=rastersbmbkg1-repmat(nanmean(nanmean(rastersbmbkg1(:,bkgwindow1,:),3),2),[1,size(rastersb,2),size(rastersb,3)]);
rasterscmbkg=rasterscmbkg1-repmat(nanmean(nanmean(rasterscmbkg1(:,bkgwindow1,:),3),2),[1,size(rastersc,2),size(rastersc,3)]);
rastersdmbkg=rastersdmbkg1-repmat(nanmean(nanmean(rastersdmbkg1(:,bkgwindow1,:),3),2),[1,size(rastersd,2),size(rastersd,3)]);
rastersembkg=rastersembkg1-repmat(nanmean(nanmean(rastersembkg1(:,bkgwindow1,:),3),2),[1,size(rasterse,2),size(rasterse,3)]);
rastersfmbkg=rastersfmbkg1-repmat(nanmean(nanmean(rastersfmbkg1(:,bkgwindow1,:),3),2),[1,size(rastersf,2),size(rastersf,3)]);
rastersgmbkg=rastersgmbkg1-repmat(nanmean(nanmean(rastersgmbkg1(:,bkgwindow1,:),3),2),[1,size(rastersg,2),size(rastersg,3)]);
rastershmbkg=rastershmbkg1-repmat(nanmean(nanmean(rastershmbkg1(:,bkgwindow1,:),3),2),[1,size(rastersh,2),size(rastersh,3)]);
rastersimbkg=rastersimbkg1-repmat(nanmean(nanmean(rastersimbkg1(:,bkgwindow1,:),3),2),[1,size(rastersi,2),size(rastersi,3)]);
rastersjmbkg=rastersjmbkg1-repmat(nanmean(nanmean(rastersjmbkg1(:,bkgwindow1,:),3),2),[1,size(rastersj,2),size(rastersj,3)]);
rasterskmbkg=rasterskmbkg1-repmat(nanmean(nanmean(rasterskmbkg1(:,bkgwindow1,:),3),2),[1,size(rastersk,2),size(rastersk,3)]);
rasterslmbkg=rasterslmbkg1-repmat(nanmean(nanmean(rasterslmbkg1(:,bkgwindow1,:),3),2),[1,size(rastersl,2),size(rastersl,3)]);

clear rastersa rastersb rastersc rastersyy rastersww rastersggg rastersiii rastershhh rastersddd rastersfff rastersww rasterseee rastersbbb rastersaaa rastersccc rasterszz rastersuu rastersxx rastersvv rasterspp rastersrr rasterstt rastersss rastersqq rastersoo rastersll rastersnn rastersmm rasterscc rastersjj rasterskk rastersd rasterse rastersf rastersu rastersbb rasterscc rastersdd rastershh rastersee rastersff rastersgg rastersv rastersw rastersx rastersy rastersz rastersaa rastersg rasterso rastersp rastersq rastersr rasterst rasterss rastersj rastersh rastersi rastersj rastersk rastersl rastersm rastersn

rasters = cat(3,rastersambkg,rastersbmbkg,rasterscmbkg,rastersdmbkg,rastersembkg,rastersfmbkg,rastersgmbkg,rastershmbkg,...
    rastersimbkg,rastersjmbkg,rasterskmbkg,rasterslmbkg);
rastersA = cat(3,rastersambkg,rastersbmbkg,rasterscmbkg,rastersdmbkg,rastersembkg,rastersfmbkg);
rastersB = cat(3,rastersgmbkg,rastershmbkg,rastersimbkg,rastersjmbkg,rasterskmbkg,rasterslmbkg);
figure
plot(squeeze(nanmean(nanmean(rasters,1),3)))
imtosave = getframe(gcf);
filename = sprintf('avgpsths.jpg');
imwrite(imtosave.cdata, ['/media/ks161/Neurobio/LivingstoneLab/Code/data-loading-code-peterbranch/margescipts/figimages/',exp_name,'/',filename], 'jpg')

clear filename imtosave
close all
figure
for ich = 1:size(rasters,1)
    subplot(ceil(sqrt(size(rasters,1))),ceil(sqrt(size(rasters,1))), ich)
    %plot(mean(squeeze(diff(lfps(ich,:,:),2)),2))
    plot((smoothdata(squeeze(nanmean(rasters(ich,:,:),3)), 'gaussian', [25 25])))
    axis tight
    title(['s',num2str(ich),'c',num2str(cell2mat(unit_namesa(ich))),'.jpg' ])
    set(gca,'fontsize',6)
end
filename='spikes.png';
saveas(gcf,[sprintf('%s/Code/data-loading-code-peterbranch/margescipts/figimages/', cfg.livingstone_lab),exp_name,'/',filename, 'tizi'],'png')

figure
plot(smoothdata(squeeze(nanmean(nanmean(rasters,1),3)),'gaussian',[25 25]))
imtosave = getframe(gcf);
filename = sprintf('avgpsthspre.jpg');
imwrite(imtosave.cdata, [sprintf('%s/Code/data-loading-code-peterbranch/margescipts/figimages/', cfg.livingstone_lab),exp_name,'/',filename, 'tizi'], 'jpg')
clear filename imtosave
mean_frtrials = squeeze(nanmean(nanmean(rasters,1),2));
figure
plot(mean_frtrials)
imtosave = getframe(gcf);
filename = sprintf('meantrials.jpg');
imwrite(imtosave.cdata, [sprintf('%s/Code/data-loading-code-peterbranch/margescipts/figimages/', cfg.livingstone_lab),exp_name,'/',filename, 'tizi'], 'jpg')


if myflag.commonnoise
    thold = 40;

    % toremove=[14340:15970 17150:19980 21620:23980 24900:size(rasters,3)];
    % toremove=14520:size(rasters,3);
    toremove = find(mean_frtrials > thold);
    rasters(:,:,toremove) = [];
    %     lfps(:,:,toremove) = [];
    stim_xy(toremove,:) = [];
    allimages = allimages(:,find(mean_frtrials <= thold));

end


figure
plot(squeeze(nanmean(nanmean(rasters,1),3)),'r','linew',2)
set(gca,'tickdir','out','linew',2)
box on
imtosave = getframe(gcf);
filename = sprintf('avgpsthsnoiserejected.jpg');
imwrite(imtosave.cdata, [sprintf('%s/Code/data-loading-code-peterbranch/margescipts/figimages/', cfg.livingstone_lab),exp_name,'/',filename, 'tizi'], 'jpg')
clear filename imtosave
close all
close all
no32=find(contains(unit_namesa,'32'),1,'last');
no64=find(contains(unit_namesa,'64'),1,'last');
% no96=find(contains(unit_namesa,'96'),1,'last');
% no128=find(contains(unit_namesa,'128'),1,'last');
figure
plot((squeeze(nanmean(nanmean(rasters(1:no32,:,:),1),3))))
imtosave = getframe(gcf);
filename = sprintf('psth1to32.jpg');
imwrite(imtosave.cdata, [sprintf('%s/Code/data-loading-code-peterbranch/margescipts/figimages/', cfg.livingstone_lab),exp_name,'/',filename, 'tizi'], 'jpg')
close gcf
figure
plot((squeeze(nanmean(nanmean(rasters(1+no32:no64,:,:),1),3))))
imtosave = getframe(gcf);
filename = sprintf('psth33to64.jpg');
imwrite(imtosave.cdata, [sprintf('%s/Code/data-loading-code-peterbranch/margescipts/figimages/', cfg.livingstone_lab),exp_name,'/',filename, 'tizi'], 'jpg')
close gcf

% figure
% plot((squeeze(nanmean(nanmean(rasters(1+no64:no96,:,:),1),3))))
% imtosave = getframe(gcf);
% filename = sprintf('psth65to96.jpg');
% imwrite(imtosave.cdata, ['/media/ks161/Neurobio/LivingstoneLab/Code/data-loading-code-peterbranch/margescipts/figimages/',exp_name,'/',filename], 'jpg')
% close gcf
%
% figure
% plot((squeeze(nanmean(nanmean(rasters(1+no96:no128,:,:),1),3))))
% imtosave = getframe(gcf);
% filename = sprintf('psth96to128.jpg');
% imwrite(imtosave.cdata, ['/media/ks161/Neurobio/LivingstoneLab/Code/data-loading-code-peterbranch/margescipts/figimages/',exp_name,'/',filename], 'jpg')
% close gcf
binwindows=40:5:230;
evk_respbinned = nan(size(rasters,1),rasterlength/binsize,size(rasters,3));
evk_resp = nan(size(rasters,1),size(rasters,3));
for site=1:size(rasters,1)
    evk_resp(site,:) = squeeze(nanmean(rasters(site,evkwindow,:),2))-nanmean(nanmean(rasters(site,bkgwindow ,:),3),2);
    for bin=1:size(binwindows,2)
        %         binwindow=binsize*(bin-1);
        %      evk_respbinned(site,bin,:) = squeeze(nanmean(rasters(site,binwindow:binwindow+10,:),2))-nanmean(nanmean(rasters(site,bkgwindow ,:),3),2);
        evk_respbinned(site,bin,:) = squeeze(nanmean(rasters(site,binwindows(bin):binwindows(bin)+5,:),2));
    end
    evkmstd(site,1)=mean(nanmean(rasters(site, evkwindow,:),3),2)-2*std(nanmean(rasters(site, bkgwindow,:),3));
    bkgpstd(site,1)=mean(nanmean(rasters(site,  bkgwindow,:),3),2)+2*std(nanmean(rasters(site,  bkgwindow,:),3));
    stdratio(site,1)=(mean(nanmean(rasters(site, evkwindow,:),3),2))/std(nanmean(rasters(site, bkgwindow,:),3));
end
goodch=find(evkmstd-bkgpstd>0);
goodstdratio=stdratio(goodch);
goodchsorted=goodch;

figure
plot(squeeze(nanmean(nanmean(rasters(goodch,:,:),1),3)),'r','linew',2)
set(gca,'tickdir','out','linew',2)
box on
imtosave = getframe(gcf);
filename = sprintf('avgpsthsgoodch.jpg');
imwrite(imtosave.cdata, [sprintf('%s/Code/data-loading-code-peterbranch/margescipts/figimages/', cfg.livingstone_lab),exp_name,'/',filename, 'tizi'], 'jpg')
clear filename imtosave

%% sort responses to all the images

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[uniqueImage, ~, imIndex]  = unique(allimages);
[uniqueImageA, ~, imIndexA]  = unique(allimagesA);
[uniqueImageB, ~, imIndexB]  = unique(allimagesB);
imds = imageDatastore(uniqueImage);
imnames = imds.Files;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

natraster = nan(size(rasters,1),size(rasters,2), max(imIndex),1); % initialize natraster
natrasterA = nan(size(rasters,1),size(rasters,2), max(imIndex),1); % initialize natraster
natrasterB = nan(size(rasters,1),size(rasters,2), max(imIndex),1); % initialize natraster
imageResp = nan(size(rasters,1),max(imIndex),1);
% rastersbyIm=nan(size(rasters,1),size(rasters,2),max(imIndex),1);
% for iIm = 1:max(imIndex)
for iIm = 1:max(imIndex)
    imageResp(:,iIm) = nanmean(evk_resp(:,imIndex == iIm),2);
    %     natraster(:,:,iIm)=nanmean(rasters(:,:,imIndex == iIm),3);
end
myflag.UMAP=0;
if myflag.UMAP
    umap_embedder = UMAP;
    % umap_embedder_natim = UMAP;
    natumap = zscore((imageResp(goodch,:)));
    natumap = umap_embedder.fit_transform(natumap);

    [val,sortedorder] = sort(natumap(:,1));
    goodchsorted=goodch(sortedorder,1)
    goodstdratiosorted=goodstdratio(sortedorder,1);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    clear natumap umap_embedder imageResp
    %%%%%%%%%%%%%%%%%%%%kaspers sorting
end
goodchsorted=goodch;
% trialsevenL=rem(1:size(rasters,3),2)==0;
% trialsoddL=rem(1:size(rasters,3),2)==1;




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% montage each site


natrasterEv = nan(size(rasters,1),rasterlength, max(imIndex),1); % initialize natraster
natrasterOd = nan(size(rasters,1),rasterlength, max(imIndex),1); % initialize natraster


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%find categories
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

imageRespbinned = nan(size(rasters,1),40,max(imIndex),1);
for iIm = 1:max(imIndex)
    for bin=1:40
        imageRespbinned(:,bin,iIm)=nanmean(evk_respbinned(:,bin,imIndex == iIm),3);
    end
    natraster(:,:,iIm)=nanmean(rasters(:,:,imIndex == iIm),3);
    natrasterA(:,:,iIm)=nanmean(rastersA(:,:,imIndexA == iIm),3);
    natrasterB(:,:,iIm)=nanmean(rastersB(:,:,imIndexB == iIm),3);
    natrasterEv(:,:,iIm)=nanmean(rasters(:,:,intersect(find(ismember(imIndex,iIm)),2:2:size(rasters,3))),3);
    natrasterOd(:,:,iIm)=nanmean(rasters(:,:,intersect(find(ismember(imIndex,iIm)),1:2:size(rasters,3))),3);
end
imageRespbinnedz=zscore(imageRespbinned(goodchsorted,:,:),[],3);
clear imageRespbinned evk_respbinned
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%55
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
colormap(jet);
color=jet;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
colormap(jet);
color=jet;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



figure
plot(squeeze(nanmean(nanmean(natraster(goodch,:,:),1),3)),'k','linew',2)
set(gca,'tickdir','out','linew',2)
box on
imtosave = getframe(gcf);
filename = sprintf('avgpsthgood.jpg');
imwrite(imtosave.cdata, [sprintf('%s/Code/data-loading-code-peterbranch/margescipts/figimages/', cfg.livingstone_lab),exp_name,'/',filename, 'tizi'], 'jpg')
clear filename imtosave

close all



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


close all
myflag.rolling3=0;
if myflag.rolling3
    rollwindowstart=10:5:495;
    clear complity
    load('/media/ks161/Neurobio/livingstone/marge/margemonkeys/complexities/complexity_talia_20each_SHINE_k4.mat')
    load('/media/ks161/Neurobio/livingstone/marge/margemonkeys/complexities/complexityname_talia_20each_SHINE_k4.mat')
    for imno=1:size(complexityname,1)
        toremove(imno)=max(cell2mat(regexp(complexityname(imno),'/')));
        complexitynametrimmed(imno,1)=extractAfter(complexityname(imno),toremove(imno));
    end
    for imno=1:size(uniqueImage,2)
        complexitybyimage(imno)=complexity(find(strcmp(complexitynametrimmed,uniqueImage(1,imno)),1));
    end
    for rollingwindow=1:size(rollwindowstart,2)
        window=rollwindowstart(rollingwindow):rollwindowstart(rollingwindow)+5;
        mean_resp_image(rollingwindow,:)=squeeze(mean(nanmean(natraster(goodch,window,:),2),1));
    end
    for rollingwindow=1:size(rollwindowstart,2)
        %             figure;hold on
        [val,ind]=sort(mean_resp_image(rollingwindow,:),'descend');
        for valnum=1:size(mean_resp_image,2)
            bestresponses(valnum) = mean_resp_image(rollingwindow,ind(valnum));
            bestcomplexities(valnum)=complexitybyimage(ind(valnum));
        end
        bestcomplexitiesN=bestcomplexities/max(bestcomplexities);
        %             plot(bestcomplexities,bestresponses,'ko')
        corrcoef(rollingwindow)=corr(bestcomplexitiesN',bestresponses');
    end
    figure; hold on;
    plot(rollwindowstart,corrcoef,'g','linew',2)
    xlabel('time ms')
    ylabel('correlation of response vs complexity/SF')
    set(gca,'tickdir','out','linew',2)
    box on
    %    filename = (['complexity slope alltop',exp_name,' 10by5to265.jpg']);
    %     saveas(gcf,['/media/ks161/Neurobio/LivingstoneLab/Code/data-loading-code-peterbranch/margescipts/figimages/',exp_name,'/',filename],'jpg');
end
%     clear rasters metai natraster nat_images
%   close all
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% close all
myflag.rolling3=0;
if myflag.rolling3
    %     rollwindowstart=50:5:180;
    clear complity
    load('/media/ks161/Neurobio/livingstone/marge/margemonkeys/complexities/SpFq_talia_20each_SHINE.mat')
    load('/media/ks161/Neurobio/livingstone/marge/margemonkeys/complexities/SpFqname_talia_20each_SHINE.mat')
    for imno=1:size(SFname,2)
        toremove(imno)=max(cell2mat(regexp(SFname(1,imno),'/')));
        SFnametrimmed(imno,1)=extractAfter(SFname(1,imno),toremove(imno));
    end
    for imno=1:size(uniqueImage,2)
        SFbyimage(imno)=SF(1,find(strcmp(SFnametrimmed,uniqueImage(1,imno)),1));
    end
    for rollingwindow=1:size(rollwindowstart,2)
        window=rollwindowstart(rollingwindow):rollwindowstart(rollingwindow)+5;
        mean_resp_image(rollingwindow,:)=squeeze(mean(nanmean(natraster(goodch,window,:),2),1));
    end
    for rollingwindow=1:size(rollwindowstart,2)
        %             figure;hold on
        [val,ind]=sort(mean_resp_image(rollingwindow,:),'descend');
        for valnum=1:size(mean_resp_image,2)
            bestresponses(valnum) = mean_resp_image(rollingwindow,ind(valnum));
            bestcomplexities(valnum)=SFbyimage(ind(valnum));
        end
        bestcomplexitiesN=bestcomplexities/max(bestcomplexities);
        %             plot(bestcomplexities,bestresponses,'ko')
        corrcoef(rollingwindow)=corr(bestcomplexitiesN',bestresponses');
    end
    plot(rollwindowstart,corrcoef,'color',[.635 0 1],'linew',2)
    plot([0 max(rollwindowstart)],[0 0],'k--','linew',2)
    set(gca,'tickdir','out','linew',2)
    box on
    filename = (['complx SF SHINE corr std1',exp_name,' 10by5to495.jpg']);
    saveas(gcf,[sprintf('%s/Code/data-loading-code-peterbranch/margescipts/figimages/', cfg.livingstone_lab),exp_name,'/',filename, 'tizi'],'png')
end
%     clear rasters metai natraster nat_images
%   close all
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all
myflag.rolling3=0;
if myflag.rolling3
    rollwindowstart=10:5:495;
    clear complity
    load('/media/ks161/Neurobio/livingstone/marge/margemonkeys/complexities/complexity_talia_20each_k4.mat')
    load('/media/ks161/Neurobio/livingstone/marge/margemonkeys/complexities/complexityname_talia_20each_k4.mat')
    for imno=1:size(complexityname,1)
        toremove(imno)=max(cell2mat(regexp(complexityname(imno),'/')));
        complexitynametrimmed(imno,1)=extractAfter(complexityname(imno),toremove(imno));
    end
    for imno=1:size(uniqueImage,2)
        complexitybyimage(imno)=complexity(find(strcmp(complexitynametrimmed,uniqueImage(1,imno)),1));
    end
    for rollingwindow=1:size(rollwindowstart,2)
        window=rollwindowstart(rollingwindow):rollwindowstart(rollingwindow)+5;
        mean_resp_image(rollingwindow,:)=squeeze(mean(nanmean(natraster(goodch,window,:),2),1));
    end
    for rollingwindow=1:size(rollwindowstart,2)
        %             figure;hold on
        [val,ind]=sort(mean_resp_image(rollingwindow,:),'descend');
        for valnum=1:size(mean_resp_image,2)
            bestresponses(valnum) = mean_resp_image(rollingwindow,ind(valnum));
            bestcomplexities(valnum)=complexitybyimage(ind(valnum));
        end
        bestcomplexitiesN=bestcomplexities/max(bestcomplexities);
        %             plot(bestcomplexities,bestresponses,'ko')
        coefficients=polyfit(bestcomplexitiesN,bestresponses,1);
        xFit=linspace(min(bestcomplexitiesN),max(bestcomplexitiesN),1000);
        yFit=polyval(coefficients,xFit);
        %             plot(xFit,yFit,'k','linew',2)
        slope(rollingwindow)=coefficients(1);
    end
    figure; hold on;
    plot(rollwindowstart,slope,'g','linew',2)
    xlabel('time ms')
    ylabel('slope of response vs complexity/SF')
    set(gca,'tickdir','out','linew',2)
    box on
    %    filename = (['complexity slope alltop',exp_name,' 10by5to265.jpg']);
    %     saveas(gcf,['/media/ks161/Neurobio/LivingstoneLab/Code/data-loading-code-peterbranch/margescipts/figimages/',exp_name,'/',filename],'jpg');
end
%     clear rasters metai natraster nat_images
%   close all
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% close all
myflag.rolling3=0;
if myflag.rolling3
    %     rollwindowstart=50:5:180;
    clear complity
    load('/media/ks161/Neurobio/livingstone/marge/margemonkeys/complexities/SpFq_talia_20each.mat')
    load('/media/ks161/Neurobio/livingstone/marge/margemonkeys/complexities/SpFqname_talia_20each.mat')
    for imno=1:size(SFname,2)
        toremove(imno)=max(cell2mat(regexp(SFname(1,imno),'/')));
        SFnametrimmed(imno,1)=extractAfter(SFname(1,imno),toremove(imno));
    end
    for imno=1:size(uniqueImage,2)
        SFbyimage(imno)=SF(1,find(strcmp(SFnametrimmed,uniqueImage(1,imno)),1));
    end
    for rollingwindow=1:size(rollwindowstart,2)
        window=rollwindowstart(rollingwindow):rollwindowstart(rollingwindow)+5;
        mean_resp_image(rollingwindow,:)=squeeze(mean(nanmean(natraster(goodch,window,:),2),1));
    end
    for rollingwindow=1:size(rollwindowstart,2)
        %             figure;hold on
        [val,ind]=sort(mean_resp_image(rollingwindow,:),'descend');
        for valnum=1:size(mean_resp_image,2)
            bestresponses(valnum) = mean_resp_image(rollingwindow,ind(valnum));
            bestcomplexities(valnum)=SFbyimage(ind(valnum));
        end
        bestcomplexitiesN=bestcomplexities/max(bestcomplexities);
        %             plot(bestcomplexities,bestresponses,'ko')
        coefficients=polyfit(bestcomplexitiesN,bestresponses,1);
        xFit=linspace(min(bestcomplexitiesN),max(bestcomplexitiesN),1000);
        yFit=polyval(coefficients,xFit);
        %             plot(xFit,yFit,'k','linew',2)
        slope(rollingwindow)=coefficients(1);
    end
    plot(rollwindowstart,slope,'color',[.635 0 1],'linew',2)
    %    xlabel('time ms')
    %    ylabel('slope of response vs SF')
    set(gca,'tickdir','out','linew',2)
    box on
    filename = (['complx SF slope alltop',exp_name,' 10by5to495.jpg']);
    saveas(gcf,[sprintf('%s/Code/data-loading-code-peterbranch/margescipts/figimages/', cfg.livingstone_lab),exp_name,'/',filename, 'tizi'],'png')
end
%     clear rasters metai natraster nat_images
close all
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
myflag.updown=0;
close all
if myflag.updown

    row_images=20;
    col_images=size(goodch,1);
    image_size=50;
    %          row1_bydepth=zeros(50*size(rasters,1),2000,3);
    up_bybin=zeros(50*size(goodch,1),2500,3);
    down_bybin=zeros(50*size(goodch,1),2500,3);
    nat_images = uniqueImage;;
    for sitee = 1:size(goodch,1)+1
        if sitee<=size(goodch,1)
            site=goodch(sitee,1);
            siteno=goodch(sitee);
            factor=5;
        else
            site=goodch;
            factor=5;
        end
        gintimage3=zeros(50*20,2000,3);
        goodimcount=0;
        goodimindices=[];
        for rollingwindow=1:size(rollingwindowstart,2)-1
            window1=rollingwindowstart(rollingwindow):rollingwindowstart(rollingwindow)+10;
            window2=rollingwindowstart(rollingwindow+1):rollingwindowstart(rollingwindow+1)+10;
            mean_resp_ch1=squeeze(mean(nanmean(natraster(site,window1,:),2),1));
            mean_resp_ch2=squeeze(mean(nanmean(natraster(site,window2,:),2),1));
            mean_bkg_ch=squeeze(nanmean(nanmean(nanmean(natraster(site,bkgwindow,:),1),2),3));;
            std_bkg_ch=std(squeeze(nanmean(nanmean(natraster(site,bkgwindow,:),1),2)));;
            mean_resp_ch1(isnan(mean_resp_ch1))=0;
            mean_resp_ch2(isnan(mean_resp_ch2))=0;
            [val,ind]=sort(mean_resp_ch2-mean_resp_ch1,'descend');
            %                     for valnum=1:20
            %                          if val(valnum)>mean_bkg_ch+factor*std_bkg_ch %%was5.5
            sortedImup = nat_images(ind(1));
            sortedImdown = nat_images(ind(end));
            imup = readimage(imds, find(contains(imnames,sortedImup),1));
            imdown = readimage(imds, find(contains(imnames,sortedImdown),1));
            preppedImageup = prepImage(imup);
            imtoprint3up=double(imresize(preppedImageup,[50,50]));
            preppedImagedown = prepImage(imdown);
            imtoprint3down=double(imresize(preppedImagedown,[50,50]));

            %                             goodimcount=goodimcount+1;
            %                             goodimindices(goodimcount)=ind(valnum);
            %                         else
            %                             imtoprint3=255*ones(48,48,3);
            %                         end
            %                         gintimage3(1+image_size*(valnum-1):48+image_size*(valnum-1),1+image_size*(mod(rollingwindow-1,3*col_images)):image_size*(mod(rollingwindow-1,3*col_images))+48,:)=imtoprint3;
            %                     end

            %                 if sitee<=size(goodch,1)
            %                  row1_bydepth(1+50*(siteno-1):50*(siteno-0),:,:)=gintimage3(1:50,:,:);
            upbybin(1+image_size*(sitee-1):50+image_size*(sitee-1),1+image_size*(mod(rollingwindow-1,3*col_images)):image_size*(mod(rollingwindow-1,3*col_images))+50,:)=imtoprint3up;
            downbybin(1+image_size*(sitee-1):50+image_size*(sitee-1),1+image_size*(mod(rollingwindow-1,3*col_images)):image_size*(mod(rollingwindow-1,3*col_images))+50,:)=imtoprint3down;
        end
        %  if sitee<=size(goodch,1)
        figure
        image(upbybin/255)
        imtosave = upbybin/255;
        filename = ['goingup_std5',exp_name,'.png'];
        imwrite(imtosave, [sprintf('%s/Code/data-loading-code-peterbranch/margescipts/figimages/', cfg.livingstone_lab),exp_name,'/',filename, 'tizi'], 'jpg')
        close all
        figure
        image(downbybin/255)
        imtosave = downbybin/255;
        filename = ['goingdown_std5',exp_name,'.png'];
        imwrite(imtosave, [sprintf('%s/Code/data-loading-code-peterbranch/margescipts/figimages/', cfg.livingstone_lab),exp_name,'/',filename, 'tizi'], 'jpg')
        close all
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



myflag.plotsig=0;
if myflag.plotsig
    figure; hold on
    rollingwindowstart=10:5:495;
    sigresponses=zeros(size(rollingwindowstart,2));
    for rollingwindw=1:size(rollingwindowstart,2)
        windw=rollingwindowstart(rollingwindw):rollingwindowstart(rollingwindw)+5;
        for imno=1:size(natraster,3)
            h=kstest2(squeeze(nanmean(nanmean(rasters(:,windw,imIndex == imno),1),2)),squeeze(nanmean(nanmean(rasters(:,50:60,imIndex == imno),1),2)),'Alpha',0.01,'Tail','larger');
            sigresponses(rollingwindw)=sigresponses(rollingwindw)+h;
        end
    end

    figure; hold on
    plot(rollingwindowstart,sigresponses,'ro','markersize',7,'markerfacecolor','r')
    plot(rollingwindowstart,sigresponses,'r','linew',2)
    set(0,'defaultfigurecolor',[1 1 1]);
    orient tall
    set(gca,'tickdir','out','linew',2)
    box on
    filename=(['imagesunder_p01.jpg']);
    imtosave = getframe(gcf);
    imwrite(imtosave.cdata, [sprintf('%s/Code/data-loading-code-peterbranch/margescipts/figimages/', cfg.livingstone_lab),exp_name,'/',filename, 'tizi'], 'jpg')

end

%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
myflag.skipthis=0;
if myflag.skipthis
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    colormap(jet);
    color=jet;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %  it6=color_it6;
    rollingwindowstart=10:5:495;
    myflag.plote=0;
    if myflag.plote
        figure; hold on

        for rollingwindw=1:size(rollingwindowstart,2)
            windw=rollingwindowstart(rollingwindw):rollingwindowstart(rollingwindw)+5;
            valusAll=squeeze(nanmean(nanmean(natraster(:,windw,:),1),2));
            valusAll(isnan(valusAll))=0;
            [valsAll, indsAll]=sort(valusAll,'ascend');
            Allvals=  smoothdata(valsAll,'gaussian',[100 100]);
            AllvalsN=(Allvals-min(Allvals))/(max(Allvals)-min(Allvals));
            allvN(rollingwindw)=sum(AllvalsN>.3679)*(1/size(natraster,3));
            allvN9(rollingwindw)=sum(AllvalsN>.9)*(1/size(natraster,3));
            allvN95(rollingwindw)=sum(AllvalsN>.95)*(1/size(natraster,3));
            allvN75(rollingwindw)=sum(AllvalsN>.75)*(1/size(natraster,3));
            allvN5(rollingwindw)=sum(AllvalsN>.5)*(1/size(natraster,3));
            plot(rollingwindowstart(rollingwindw),sum(AllvalsN>.3679)*(1/size(natraster,3)),'ro','markersize',7,'markerfacecolor','r')
        end
        plot(rollingwindowstart,allvN,'r','linew',2)
        set(0,'defaultfigurecolor',[1 1 1]);
        orient tall
        set(gca,'tickdir','out','linew',2)
        box on
        filename=(['imagesover_e.jpg']);
        imtosave = getframe(gcf);
        imwrite(imtosave.cdata, [sprintf('%s/Code/data-loading-code-peterbranch/margescipts/figimages/', cfg.livingstone_lab),exp_name,'/',filename, 'tizi'], 'jpg')
        figure; hold on
        plot(rollingwindowstart,allvN9,'ro','markersize',7,'markerfacecolor','r')
        plot(rollingwindowstart,allvN9,'r','linew',2)
        set(0,'defaultfigurecolor',[1 1 1]);
        orient tall
        set(gca,'tickdir','out','linew',2)
        box on
        filename=(['imagesover_9.jpg']);
        imtosave = getframe(gcf);
        imwrite(imtosave.cdata, [sprintf('%s/Code/data-loading-code-peterbranch/margescipts/figimages/', cfg.livingstone_lab),exp_name,'/',filename, 'tizi'], 'jpg')

        figure; hold on
        plot(rollingwindowstart,allvN75,'ro','markersize',7,'markerfacecolor','r')
        plot(rollingwindowstart,allvN75,'r','linew',2)
        set(0,'defaultfigurecolor',[1 1 1]);
        orient tall
        set(gca,'tickdir','out','linew',2)
        box on
        filename=(['imagesover_75.jpg']);
        imtosave = getframe(gcf);
        imwrite(imtosave.cdata, [sprintf('%s/Code/data-loading-code-peterbranch/margescipts/figimages/', cfg.livingstone_lab),exp_name,'/',filename, 'tizi'], 'jpg')

        figure; hold on
        plot(rollingwindowstart,allvN5,'ro','markersize',7,'markerfacecolor','r')
        plot(rollingwindowstart,allvN5,'r','linew',2)
        set(0,'defaultfigurecolor',[1 1 1]);
        orient tall
        set(gca,'tickdir','out','linew',2)
        box on
        filename=(['imagesover_5.jpg']);
        imtosave = getframe(gcf);
        imwrite(imtosave.cdata, [sprintf('%s/Code/data-loading-code-peterbranch/margescipts/figimages/', cfg.livingstone_lab),exp_name,'/',filename, 'tizi'], 'jpg')


        figure; hold on
        plot(rollingwindowstart,allvN95,'ro','markersize',7,'markerfacecolor','r')
        plot(rollingwindowstart,allvN95,'r','linew',2)
        set(0,'defaultfigurecolor',[1 1 1]);
        orient tall
        set(gca,'tickdir','out','linew',2)
        box on
        filename=(['imagesover_95.jpg']);
        imtosave = getframe(gcf);
        imwrite(imtosave.cdata, [sprintf('%s/Code/data-loading-code-peterbranch/margescipts/figimages/', cfg.livingstone_lab),exp_name,'/',filename, 'tizi'], 'jpg')

    end
end
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    rollingwindowstart=10:5:355;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
myflag.plotcorr=0;
if myflag.plotcorr
    clear corrsamediff2
    corrsamediff2=zeros(size(rollingwindowstart,2),size(natraster,1)-1);
    corrsamediff2_100=zeros(size(rollingwindowstart,2),size(natraster,1)-1);
    countbydist=ones(size(rollingwindowstart,2),size(natraster,1)-1);
    for rollingwindw=1:size(rollingwindowstart,2)
        windw=rollingwindowstart(rollingwindw):rollingwindowstart(rollingwindw)+5;
        valusAll=squeeze(nanmean(nanmean(natraster(goodch,windw,:),1),2));
        valusAll(isnan(valusAll))=0;
        [valsAll, indsAll]=sort(valusAll,'descend');

        for site1=1:size(natraster,1)-1
            for site2=site1+1:size(natraster,1)
                dist=site2-site1;
                countbydist(rollingwindw,dist)=countbydist(rollingwindw,dist)+1;
                corrsamediff2_100(rollingwindw,dist)= corrsamediff2(rollingwindw,dist)+squeeze(corr(squeeze(nanmean(natraster(site1,windw,indsAll(1:100)),2)),squeeze(nanmean(natraster(site2,windw,indsAll(1:100)),2))));
                corrsamediff2(rollingwindw,dist)= corrsamediff2(rollingwindw,dist)+squeeze(corr(squeeze(nanmean(natraster(site1,windw,:),2)),squeeze(nanmean(natraster(site2,windw,:),2))));
            end
        end
    end


    figure; hold on
    plot(rollingwindowstart,nanmean(corrsamediff2_100./countbydist,2),'b','linew',2)
    plot(rollingwindowstart,nanmean(corrsamediff2_100./countbydist,2),'bo','markerfacecolor','b','markersize',9)
    set(gca,'tickdir','out','linew',2)
    imtosave=getframe(gcf);
    filename = ['corrtop100.png'];
    imwrite(imtosave.cdata, [sprintf('%s/Code/data-loading-code-peterbranch/margescipts/figimages/', cfg.livingstone_lab),exp_name,'/',filename, 'tizi'], 'jpg')
    % clear corrsamediff2
    close all


    figure; hold on
    plot(rollingwindowstart,nanmean(corrsamediff2./countbydist,2),'b','linew',2)
    plot(rollingwindowstart,nanmean(corrsamediff2./countbydist,2),'bo','markerfacecolor','b','markersize',9)
    set(gca,'tickdir','out','linew',2)
    imtosave=getframe(gcf);
    filename = ['corrtopall.png'];
    imwrite(imtosave.cdata, [sprintf('%s/Code/data-loading-code-peterbranch/margescipts/figimages/', cfg.livingstone_lab),exp_name,'/',filename, 'tizi'], 'jpg')
    % clear corrsamediff2
    close all
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


colormap(jet);
color=jet;



close all
myflag.rolling3=0;
if myflag.rolling3
    clear complity
    load('/media/ks161/Neurobio/livingstone/marge/margemonkeys/complexities/complexity_manyOO_k4.mat')
    load('/media/ks161/Neurobio/livingstone/marge/margemonkeys/complexities/complexityname_manyOO_k4.mat')
    for imno=1:size(complexityname,1)
        toremove(imno)=max(cell2mat(regexp(complexityname(imno),'/')));
        complexitynametrimmed(imno,1)=extractAfter(complexityname(imno),toremove(imno));
    end
    complexity_mny=complexity(find(ismember(complexitynametrimmed,uniqueImage)));
    %      rollingwindowstart=10:5:355;
    for sitee = 1:size(goodch,1)
        site=goodch(sitee,1);
        for rollingwindow=1:size(rollingwindowstart,2)
            window=rollingwindowstart(rollingwindow):rollingwindowstart(rollingwindow)+5;
            mean_resp_ch=squeeze(mean(nanmean(natraster(site,window,:),2),1));
            mean_resp_ch(isnan(mean_resp_ch))=0;
            [val,ind]=sort(mean_resp_ch,'descend');
            for valnum=1:50
                %                 valnum=valnum;
                sortedIm = uniqueImage(ind(valnum));
                complity(rollingwindow,sitee,valnum)=mean(complexity(find(strcmp(complexitynametrimmed,sortedIm))));
            end
        end
    end

    % save  ('/media/ks161/Neurobio/LivingstoneLab/Code/data-loading-code-peterbranch/margescipts/figimages/red_240302_to_14_DNN/complex_red_240302_to_14_tal20.mat',  'complex_red_240302_to_14_tal20')
    %     clear rasters metai natraster nat_images
    figure; hold on
    % for site=1:size(goodch,1)
    plot(rollingwindowstart,squeeze(nanmean(nanmean(complity,3),2)),'g','linew',2)
    plot(rollingwindowstart,squeeze(nanmean(nanmean(complity,3),2)),'go','markerfacecolor','g','markersize',9)
    xticks(0:50:rollingwindowstart(1,end))
    xticklabels(0:50:rollingwindowstart(1,end))
    plot(250,nanmean(complexity_mny),'rx')
    ylabel('image complexity')
    set(gca,'tickdir','out','linew',2)
    box on
    filename = (['complexity 50top',exp_name,' 10by5to495.jpg']);
    saveas(gcf,[sprintf('%s/Code/data-loading-code-peterbranch/margescipts/figimages/', cfg.livingstone_lab),exp_name,'/',filename, 'tizi'],'png')
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all
myflag.SF=0;
if myflag.SF
    clear SFity
    load('/media/ks161/Neurobio/livingstone/marge/margemonkeys/complexities/SpFq_talia_20each.mat')
    load('/media/ks161/Neurobio/livingstone/marge/margemonkeys/complexities/SpFqname_talia_20each.mat')
    for imno=1:size(SFname,2)
        toremove(imno)=max(cell2mat(regexp(SFname(imno),'/')));
        SFnametrimmed(1,imno)=extractAfter(SFname(1,imno),toremove(imno));
    end
    SF_mny=SF(find(ismember(SFnametrimmed,uniqueImage)));
    %      rollingwindowstart=10:5:355;
    for sitee = 1:size(goodch,1)
        site=goodch(sitee,1);
        for rollingwindow=1:size(rollingwindowstart,2)
            window=rollingwindowstart(rollingwindow):rollingwindowstart(rollingwindow)+5;
            mean_resp_ch=squeeze(mean(nanmean(natraster(site,window,:),2),1));
            mean_resp_ch(isnan(mean_resp_ch))=0;
            [val,ind]=sort(mean_resp_ch,'descend');
            for valnum=1:50
                %                 valnum=valnum;
                sortedIm = uniqueImage(ind(valnum));
                SpFr(rollingwindow,sitee,valnum)=mean(SF(find(strcmp(SFnametrimmed,sortedIm))));
            end
        end
    end

    % save  ('/media/ks161/Neurobio/LivingstoneLab/Code/data-loading-code-peterbranch/margescipts/figimages/red_240302_to_14_DNN/complex_red_240302_to_14_tal20.mat',  'complex_red_240302_to_14_tal20')
    %     clear rasters metai natraster nat_images
    figure; hold on
    % for site=1:size(goodch,1)
    plot(rollingwindowstart,squeeze(nanmean(nanmean(SpFr/5,3),2)),'c','linew',2)
    plot(rollingwindowstart,squeeze(nanmean(nanmean(SpFr/4,3),2)),'co','markerfacecolor','c','markersize',9)
    xticks(0:50:rollingwindowstart(1,end))
    xticklabels(0:50:rollingwindowstart(1,end))
    plot(250,nanmean(SF_mny/5),'rx')
    ylabel('image SF')
    set(gca,'tickdir','out','linew',2)
    box on
    filename = (['SpFq 50top',exp_name,' 10by5to495.jpg']);
    saveas(gcf,[sprintf('%s/Code/data-loading-code-peterbranch/margescipts/figimages/', cfg.livingstone_lab),exp_name,'/',filename, 'tizi'],'png')

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    close all

    %     load('/media/ks161/Neurobio/LivingstoneLab/Code/data-loading-code-peterbranch/margescipts/figimages/red_240302_to_14_DNN/SF.mat')
    %      load('/media/ks161/Neurobio/LivingstoneLab/Code/data-loading-code-peterbranch/margescipts/figimages/red_240302_to_14_DNN/SFname.mat')
    %      load('/media/ks161/Neurobio/LivingstoneLab/Code/data-loading-code-peterbranch/margescipts/figimages/red_240302_to_14_DNN/complexity_talia_20each_k4.mat')
    %      load('/media/ks161/Neurobio/LivingstoneLab/Code/data-loading-code-peterbranch/margescipts/figimages/red_240302_to_14_DNN/complexityname_talia_20each_k4.mat')
    figure; plot(SF,complexity,'ko')
    xlabel('SF')
    ylabel('complexity')
    axis([5 25 0 500])
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
colormap(jet);
color=jet;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  it6=color_it6;
%        colormap(it6)
myflag.plotsigmoid=0;
myflag.plotsigmoid=0;
if myflag.plotsigmoid
    for sitee=1:size(goodch,1)
        site=goodch(sitee);
        notsite=goodch(goodch~=site);
        % figure;hold on
        rollingwindowstart=95:20:490;
        for rollingwindw=1:size(rollingwindowstart,2)
            windw=rollingwindowstart(rollingwindw):rollingwindowstart(rollingwindw)+20;
            valusAll=squeeze(nanmean(nanmean(natraster(site,windw,:),1),2));
            valusE=squeeze(nanmean(nanmean(natrasterEv(site,windw,:),1),2));
            valusO=squeeze(nanmean(nanmean(natrasterOd(site,windw,:),1),2));
            valusnotsiteE=squeeze(nanmean(nanmean(natrasterEv(notsite,windw,:),1),2));
            valusnotsiteO=squeeze(nanmean(nanmean(natrasterOd(notsite,windw,:),1),2));
            valusE(isnan(valusE))=0;
            valusAll(isnan(valusAll))=0;
            valusO(isnan(valusO))=0;
            valusnotsiteE(isnan(valusnotsiteE))=0;
            valusnotsiteO(isnan(valusnotsiteO))=0;
            valusAll=(valusAll)/(max(valusAll));
            valusEN=(valusE)/(max(valusE));
            valusON=(valusO)/(max(valusO));
            valusnotsiteEN=(valusnotsiteE)/(max(valusnotsiteE));
            valusnotsiteON=(valusnotsiteO)/(max(valusnotsiteO));
            [valsAll, indsAll]=sort(valusAll,'ascend');
            [valsE, indsE]=sort(valusE,'ascend');
            [valsO, indsO]=sort(valusO,'ascend');
            splitvals=(valusEN(indsO)+valusON(indsE));
            splitvals=splitvals/max(splitvals);
            splitvalscum(site,rollingwindw,:)=splitvals;
            Allvalscum(site,rollingwindw,:)=valsAll;
            notsitevals=valusnotsiteEN(indsO)+valusnotsiteON(indsE);
            notsitevals=notsitevals/max(notsitevals);
            notsitevalscum(site,rollingwindw,:)=notsitevals;
        end
    end
    figure; hold on
    for rollingwindw=1:size(rollingwindowstart,2)
        samevals=  smoothdata(squeeze(nanmean(splitvalscum(:,rollingwindw,:),1)),'gaussian',[100 100]);
        samevalsN=(samevals-min(samevals))/(max(samevals)-min(samevals));
        Allvals=  smoothdata(squeeze(nanmean(Allvalscum(:,rollingwindw,:),1)),'gaussian',[100 100]);
        AllvalsN=(Allvals-min(Allvals))/(max(Allvals)-min(Allvals));
        diffvals= smoothdata(squeeze(nanmean(notsitevalscum(:,rollingwindw,:),1)),'gaussian',[100 100]);
        diffvalsN=(diffvals-min(diffvals))/(max(diffvals)-min(diffvals));
        subplot(2,1,1); hold on
        plot(1:size(splitvals), samevalsN,'color',color(round(rollingwindw*255/size(rollingwindowstart,2)),:),'linew',2)
        set(0,'defaultfigurecolor',[1 1 1]);
        orient tall
        set(gca,'tickdir','out','linew',2)
        box on
        subplot(2,1,2); hold on
        plot(1:size(splitvals), AllvalsN,'color',color(round(rollingwindw*255/size(rollingwindowstart,2)),:),'linew',2)
    end
    set(0,'defaultfigurecolor',[1 1 1]);
    orient tall
    set(gca,'tickdir','out','linew',2)
    box on
    filename=(['ranksallsiteby20.jpg']);
    imtosave = getframe(gcf);
    imwrite(imtosave.cdata, [sprintf('%s/Code/data-loading-code-peterbranch/margescipts/figimages/', cfg.livingstone_lab),exp_name,'/',filename, 'tizi'], 'jpg')

    clear cumsame cumdiff
    for rollingwindw=1:size(rollingwindowstart,2)
        samevals=  smoothdata(squeeze(nanmean(splitvalscum(:,rollingwindw,:),1)),'gaussian',[100 100]);
        samevalsN=(samevals-min(samevals))/(max(samevals)-min(samevals));
        diffvals= smoothdata(squeeze(nanmean(notsitevalscum(:,rollingwindw,:),1)),'gaussian',[100 100]);
        diffvalsN=(diffvals-min(diffvals))/(max(diffvals)-min(diffvals));
        cumsame(rollingwindw)=nanmean(squeeze(samevalsN(500:end,1)));
        cumdiff(rollingwindw)=nanmean(squeeze(diffvalsN(500:end,1)));
    end
    figure; hold  on
    plot(rollingwindowstart,cumsame,'g','linew',2)
    plot(rollingwindowstart,cumdiff,'r','linew',2)
    set(0,'defaultfigurecolor',[1 1 1]);
    orient tall; %axis([ 0 300 0.25 0.6])
    set(gca,'tickdir','out','linew',2)
    box on
    filename=(['last500normresps.jpg']);
    imtosave = getframe(gcf);
    imwrite(imtosave.cdata, [sprintf('%s/Code/data-loading-code-peterbranch/margescipts/figimages/', cfg.livingstone_lab),exp_name,'/',filename, 'tizi'], 'jpg')

    figure; hold  on
    plot(rollingwindowstart,(cumdiff-cumsame)./(cumdiff+cumsame),'k','linew',2)
    orient tall
    set(gca,'tickdir','out','linew',2)
    box on;  %axis([ 0 300 -0.05 0.1])
    filename=(['last2Kindx.png']);
    imtosave = getframe(gcf);
    imwrite(imtosave.cdata, [sprintf('%s/Code/data-loading-code-peterbranch/margescipts/figimages/', cfg.livingstone_lab),exp_name,'/',filename, 'tizi'], 'jpg')

    %%%%%%%%%%corr matrix
    diskfilt = fspecial('disk',3);


    avgsplitvalscum=squeeze(nanmean(splitvalscum,1));
    avgnotsitevalscum=squeeze(nanmean(notsitevalscum,1));

    %    avgsplitvalscum=squeeze(nanmean(splitvalscum(:,:,10000:end),1));
    %     avgnotsitevalscum=squeeze(nanmean(notsitevalscum(:,:,10000:end),1));
    for rollingwindw=1:size(rollingwindowstart,2)
        corrsamediff(rollingwindw)=squeeze(corr((squeeze(avgsplitvalscum(rollingwindw,:)))', (squeeze(avgnotsitevalscum(rollingwindw,:)))'));
    end
    %   figure; plot(rollingwindowstart,corrsamediff)
    %     close all


end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if myflag.skipthis
    rollingwindowstart=80:5:495;
    for rollingwindw=1:size(rollingwindowstart,2)
        count=0;
        windw=rollingwindowstart(rollingwindw):rollingwindowstart(rollingwindw)+5;
        valusAll=squeeze(nanmean(nanmean(natraster(:,windw,:),1),2));
        valusAll(isnan(valusAll))=0;
        [valsAll, indsAll]=sort(valusAll,'descend');
        for sitee1=1:size(goodch,1)-1
            site1=goodch(sitee1);
            for sitee2=sitee1+1:size(goodch,1)
                site2=goodch(sitee2);
                count=count+1;
                corrsamediff2(rollingwindw,count)=squeeze(corr(squeeze(nanmean(natraster(site1,windw,indsAll(1:end)),2)),squeeze(nanmean(natraster(site2,windw,indsAll(1:end)),2))));
            end
        end
    end
    figure; plot(rollingwindowstart,nanmean(corrsamediff2,2),'linew',2)
    %    axis([0 250 -.01 0.065])
    set(gca,'tickdir','out','linew',2)
    imtosave=getframe(gcf);
    filename = ['corrtopall.png'];
    imwrite(imtosave.cdata, [sprintf('%s/Code/data-loading-code-peterbranch/margescipts/figimages/', cfg.livingstone_lab),exp_name,'/',filename, 'tizi'], 'jpg')
    clear corrsamediff2
    close all


end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

myflag.rollinggoodch=0;
if myflag.rollinggoodch
    rollingwindowstart=90:20:470;
    row_images=20;
    col_images=20;
    image_size=50;
    %       row1_bydepth=zeros(50*383,1000,3);
    site=goodch;
    gintimage3=zeros(1000,1000,3);
    goodimcount=0;
    goodimindices=[];
    for rollingwindow=1:size(rollingwindowstart,2)
        window=rollingwindowstart(rollingwindow):rollingwindowstart(rollingwindow)+10;
        mean_resp_ch=squeeze(mean(nanmean(natraster(site,window,:),2),1));
        mean_bkg_ch=squeeze(nanmean(nanmean(natraster(site,bkgwindow,:),2),3));;
        std_bkg_ch=std(squeeze(nanmean(natraster(site,bkgwindow,:),2)));;
        mean_resp_ch(isnan(mean_resp_ch))=0;
        [val,ind]=sort(mean_resp_ch,'descend');
        for valnum=1:20
            if val(valnum)>mean_bkg_ch+.25*std_bkg_ch %%was3
                sortedIm =uniqueImage(ind(valnum));
                im = readimage(imds, find(contains(imnames,sortedIm),1));
                preppedImage = prepImage(im);
                imtoprint3=double(imresize(preppedImage,[48,48]));
                goodimcount=goodimcount+1;
                goodimindices(goodimcount)=ind(valnum);
            else
                imtoprint3=255*ones(48,48,3);
            end
            gintimage3(1+image_size*(valnum-1):48+image_size*(valnum-1),1+image_size*(mod(rollingwindow-1,3*col_images)):image_size*(mod(rollingwindow-1,3*col_images))+48,:)=imtoprint3;
        end
    end
    row1_bydepth(1+50*(site-1):50*site,:,:)=gintimage3(1:50,:,:);
    figure
    image(gintimage3/255)
    imtosave = gintimage3/255;
    filename = 'roll_goodc90by20t0470.jpg';
    imwrite(imtosave.cdata, [sprintf('%s/Code/data-loading-code-peterbranch/margescipts/figimages/', cfg.livingstone_lab),exp_name,'/',filename, 'tizi'], 'jpg')
    close all
    clear imtosave filename

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    rollingwindowstart=90:20:470;
    clear toplot
    for rollingwindow=1:size(rollingwindowstart,2)
        window=rollingwindowstart(rollingwindow):rollingwindowstart(rollingwindow)+10;
        mean_resp_ch=squeeze(mean(nanmean(natraster(site,window,:),2),1));
        mean_resp_ch(mean_resp_ch<=0)=0;
        [val,ind]=sort(mean_resp_ch,'descend');
        sortedIm =uniqueImage(ind(1:25,1));

        ind(val==0)=[];
        if ~isempty(ind)
            toplot(rollingwindow,:)=smoothdata(nanmean(nanmean(natraster(site,:,ind(1:min([25 size(ind,1)]),1)),3),1),'gaussian',[15 15]);
            %  toplot(rollingwindow,:)=smoothdata(nanmean(nanmean(natraster(site,:,ind(1:1:min([10 size(ind,1)]),1)),3),1),'gaussian',[15 15]);
        end
    end
    clear nat_montage natmontage
    close all
    figure; hold on
    for roll=1:size(toplot,1)
        plot(1:size(toplot,2),toplot(roll,:),'color',color(ceil(roll*256/(size(toplot,1))),:),'linew',2)
    end
    box on
    set(gca,'tickdir','out','linew',2)
    %   axis([0 350 -.1 .7])
    set(gca,'tickdir','out','linew',2,'fontsize',19)
    filename = 'times90by20to470_goodch.jpg';
    saveas(gcf,[sprintf('%s/Code/data-loading-code-peterbranch/margescipts/figimages/', cfg.livingstone_lab),exp_name,'/',filename, 'tizi'],'png')
    close all
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

myflag.rolling2=1;

close all
if myflag.rolling2
    rollingwindowstart=60:10:450;
    row_images=20;
    col_images=size(goodch,1);
    image_size=50;
    %          row1_bydepth=zeros(50*size(rasters,1),2000,3);
    row1_bydepthsq=zeros(50*size(goodch,1),2000,3);
    nat_images = uniqueImage;;
    for sitee = 1:size(goodchsorted,1)+1
        if sitee<=size(goodchsorted,1)
            site=goodchsorted(sitee,1);
            siteno=goodch(sitee);
            factor=2;
        else
            site=goodch;
            factor=2;
        end
        gintimage3=zeros(50*20,2000,3);
        goodimcount=0;
        goodimindices=[];
        for rollingwindow=1:size(rollingwindowstart,2)
            window=rollingwindowstart(rollingwindow):rollingwindowstart(rollingwindow)+10;
            mean_resp_ch=squeeze(mean(nanmean(natraster(site,window,:),2),1));
            mean_bkg_ch=squeeze(nanmean(nanmean(nanmean(natraster(site,bkgwindow,:),1),2),3));;
            std_bkg_ch=std(squeeze(nanmean(nanmean(natraster(site,bkgwindow,:),1),2)));;
            mean_resp_ch(isnan(mean_resp_ch))=0;
            [val,ind]=sort(mean_resp_ch,'descend');
            for valnum=1:20
                if val(valnum)>mean_bkg_ch+factor*std_bkg_ch %%was5.5
                    sortedIm = nat_images(ind(valnum));
                    im = readimage(imds, find(contains(imnames,sortedIm),1));
                    preppedImage = prepImage(im);
                    imtoprint3=double(imresize(preppedImage,[48,48]));
                    goodimcount=goodimcount+1;
                    goodimindices(goodimcount)=ind(valnum);
                else
                    imtoprint3=255*ones(48,48,3);
                end
                gintimage3(1+image_size*(valnum-1):48+image_size*(valnum-1),1+image_size*(mod(rollingwindow-1,3*col_images)):image_size*(mod(rollingwindow-1,3*col_images))+48,:)=imtoprint3;
            end
        end
        %                 if sitee<=size(goodch,1)
        %                  row1_bydepth(1+50*(siteno-1):50*(siteno-0),:,:)=gintimage3(1:50,:,:);
        row1_bydepthsq(1+50*(sitee-1):50*(sitee-0),:,:)=gintimage3(1:50,:,:);
        %                 end
        figure
        image(gintimage3/255)
        imtosave = gintimage3/255;
        sitee=sitee
        if sitee<=size(goodch,1)
            filename = sprintf('roll_60by10to450std2_site%02d_.jpg',goodchsorted(sitee));
        else
            filename='goodch60by10to450std2.jpg';
        end
        imwrite(imtosave, [sprintf('%s/Code/data-loading-code-peterbranch/margescipts/figimages/', cfg.livingstone_lab),exp_name,'/',filename, 'tizi'], 'jpg')
        close all
        clear imtosave filename

    end

    %  if sitee<=size(goodchsorted,1)
    figure
    image(row1_bydepthsq/255)
    imtosave = row1_bydepthsq/255;
    filename = ['best_60by10to450_2std',exp_name,'.png'];
    imwrite(imtosave, [sprintf('%s/Code/data-loading-code-peterbranch/margescipts/figimages/', cfg.livingstone_lab),exp_name,'/',filename, 'tizi'], 'jpg')
    close all
    %     figure
    %         image(row1_bydepth/255)
    %                 imtosave = row1_bydepthsq/255;
    %                     filename = ['best_50by10to450_6_std_sorted',exp_name,'.png'];
    %                 imwrite(imtosave, ['/media/ks161/Neurobio/LivingstoneLab/Code/data-loading-code-peterbranch/margescipts/figimages/',exp_name,'/',filename],'png')
    %     close all
    %  end
end
%     clear rasters metai natraster nat_images
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all
if myflag.rolling3
    load('/media/ks161/Neurobio/livingstone/marge/margemonkeys/red/complexity_talia_20each_k4_noblur.mat')
    load('/media/ks161/Neurobio/livingstone/marge/margemonkeys/red/complexityname_talia_20each_k4_noblur.mat')
    for imno=1:size(complexityname,1)
        complexitynametrimmed(imno,1)=extractAfter(complexityname(imno),'/home/msl1/Neurobio/LivingstoneLab/Stimuli/talia_20each/');
    end
    complexity_mny=complexity(find(ismember(complexitynametrimmed,uniqueImage)));
    rollingwindowstart=10:5:545;
    for sitee = 1:size(goodchsorted,1)
        site=goodchsorted(sitee,1);

        for rollingwindow=1:size(rollingwindowstart,2)
            window=rollingwindowstart(rollingwindow):rollingwindowstart(rollingwindow)+5;
            mean_resp_ch=squeeze(mean(nanmean(natraster(site,window,:),2),1));
            mean_resp_ch(isnan(mean_resp_ch))=0;
            [val,ind]=sort(mean_resp_ch,'descend');

            for valnum=1:100
                sortedIm = uniqueImage(ind(valnum));
                complity(rollingwindow,sitee,valnum)=complexity(find(strcmp(complexitynametrimmed,sortedIm)));
            end
        end
    end

    % save  ('/media/ks161/Neurobio/LivingstoneLab/Code/data-loading-code-peterbranch/margescipts/figimages/red_240302_to_14_DNN/complex_red_240302_to_14_tal20.mat',  'complex_red_240302_to_14_tal20')
    %     clear rasters metai natraster nat_images
    figure; hold on
    % for site=1:size(goodch,1)
    plot(rollingwindowstart,squeeze(nanmean(nanmean(complity,3),2)),'g','linew',2)
    plot(rollingwindowstart,squeeze(nanmean(nanmean(complity,3),2)),'ko','markerfacecolor','k','markersize',9)
    plot(250,nanmean(complexity_mny),'kx')
    xticks(0:50:rollingwindowstart(1,end))
    xticklabels(0:50:rollingwindowstart(1,end))
    ylabel('image complexity')
    set(gca,'tickdir','out','linew',2)
    filename = (['complexity k4noblur std1 100top',exp_name,' 1by5to545.jpg']);
    saveas(gcf,[sprintf('%s/Code/data-loading-code-peterbranch/margescipts/figimages/', cfg.livingstone_lab),exp_name,'/',filename, 'tizi'],'png')
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if myflag.rollingrank
    topnumber=100;
    rollingwindowstartrank=1:10:490;
    clear val ind
    for rollingwindow=1:size(rollingwindowstartrank,2)
        for chh=1:size(goodchsorted,1)
            ch1=goodchsorted(chh);
            window=rollingwindowstartrank(rollingwindow):rollingwindowstartrank(rollingwindow)+10;
            mean_resp_ch=squeeze(nanmean(natraster(ch1,window,:),2));
            mean_resp_ch(isnan(mean_resp_ch))=0;
            allchresp(chh,:)=mean_resp_ch;

            mean_resp_chE=squeeze(nanmean(natrasterEv(ch1,window,:),2));
            mean_resp_chE(isnan(mean_resp_chE))=0;
            allchrespE(chh,:)=mean_resp_chE;
            mean_resp_chO=squeeze(nanmean(natrasterOd(ch1,window,:),2));
            mean_resp_chO(isnan(mean_resp_chO))=0;
            allchrespO(chh,:)=mean_resp_chO;
        end
        samecount=0;
        diffcount=0;
        allcount=0;
        clear val1 ind1
        for chh=1:size(goodchsorted,1)
            [val1(chh,:),ind1(chh,:)]=sort(allchresp(chh,:),'descend');
            [val1E(chh,:),ind1E(chh,:)]=sort(allchrespE(chh,:),'descend');
            [val1O(chh,:),ind1O(chh,:)]=sort(allchrespO(chh,:),'descend');
            for ch2=1:size(goodchsorted,1)
                if ch2==chh
                    %                             samecount=samecount+1;
                    %                             samechbest(samecount)=nanmean(val1(chh,1:25),2);
                    samecount=samecount+1;
                    samechbest(samecount)=nanmean(allchrespE(ch2,ind1O(chh,1:topnumber)),2);
                    samecount=samecount+1;
                    samechbest(samecount)=nanmean(allchrespO(ch2,ind1E(chh,1:topnumber)),2);
                else
                    diffcount=diffcount+1;
                    diffch(diffcount)=nanmean(allchrespE(ch2,ind1O(chh,1:topnumber)),2);
                    diffcount=diffcount+1;
                    diffch(diffcount)=nanmean(allchrespO(ch2,ind1E(chh,1:topnumber)),2);
                end
            end
            allcount=allcount+1;
            allch(allcount)=nanmean(allchresp(chh,:),2);
        end
        samechanbest(rollingwindow)=nanmean(samechbest);
        diffchan(rollingwindow)=nanmean(diffch);
        allchan(rollingwindow)=nanmean(allch);
        clear samechbest diffch allch
        samecount=0;
        diffcount=0;
        allcount=0;
    end
    figure; hold on
    plot(rollingwindowstartrank,(samechanbest),'g','linew',2)
    plot(rollingwindowstartrank,(diffchan),'r','linew',2)
    plot(rollingwindowstartrank,(allchan),'k','linew',2)
    xticks(0:50:rollingwindowstartrank(1,end))
    xticklabels(0:50:rollingwindowstartrank(1,end))
    ylabel(' response')
    set(gca,'tickdir','out','linew',2)
    filename = (['ranks',exp_name,' 1by10to490_',num2str(topnumber),'top.jpg']);
    saveas(gcf,[sprintf('%s/Code/data-loading-code-peterbranch/margescipts/figimages/', cfg.livingstone_lab),exp_name,'/',filename, 'tizi'],'png')

    figure; hold on
    plot(rollingwindowstartrank,(samechanbest/max(samechanbest)),'g','linew',2)
    plot(rollingwindowstartrank,(diffchan/max(diffchan)),'r','linew',2)
    plot(rollingwindowstartrank,(allchan/max(allchan)),'k','linew',2)
    xticks(0:50:rollingwindowstartrank(1,end))
    xticklabels(0:50:rollingwindowstartrank(1,end))
    ylabel('normalized response')
    set(gca,'tickdir','out','linew',2)
    filename = (['ranksN 1by10to490_',num2str(topnumber),'top.jpg']);
    saveas(gcf,[sprintf('%s/Code/data-loading-code-peterbranch/margescipts/figimages/', cfg.livingstone_lab),exp_name,'/',filename, 'tizi'],'png')




    figure; hold on
    plot(rollingwindowstartrank,(samechanbest-diffchan)./(samechanbest+diffchan),'b','linew',2)
    xticks(0:50:rollingwindowstartrank(1,end))
    xticklabels(0:50:rollingwindowstartrank(1,end))
    ylabel('same/diff index')
    set(gca,'tickdir','out','linew',2)
    filename = (['ranksIndex 1by10to490_',num2str(topnumber),'top.jpg']);
    saveas(gcf,[sprintf('%s/Code/data-loading-code-peterbranch/margescipts/figimages/', cfg.livingstone_lab),exp_name,'/',filename, 'tizi'],'png')


    close all
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end



close all
if myflag.rolling1
    rollingwindowstart=90:20:480;
    row_images=20;
    col_images=20;
    image_size=50;
    row1_bydepth=zeros(50*size(rasters,1),1000,3);
    row1_bydepthsq=zeros(50*size(goodch,1),1000,3);
    nat_images = uniqueImage;;
    for sitee = 1:size(goodchsorted,1)+1
        if sitee<=size(goodchsorted,1)
            site=goodchsorted(sitee,1);
            siteno=goodch(sitee);
            factor=6;
        else
            site=goodch;
            factor=6;
        end
        gintimage3=zeros(1000,1000,3);
        goodimcount=0;
        goodimindices=[];
        for rollingwindow=1:size(rollingwindowstart,2)
            window=rollingwindowstart(rollingwindow):rollingwindowstart(rollingwindow)+20;
            mean_resp_ch=squeeze(mean(nanmean(natraster(site,window,:),2),1));
            mean_bkg_ch=squeeze(nanmean(nanmean(nanmean(natraster(site,bkgwindow,:),1),2),3));;
            std_bkg_ch=std(squeeze(nanmean(nanmean(natraster(site,bkgwindow,:),1),2)));;
            mean_resp_ch(isnan(mean_resp_ch))=0;
            [val,ind]=sort(mean_resp_ch,'descend');
            for valnum=1:20
                if val(valnum)>mean_bkg_ch+factor*std_bkg_ch %%was5.5
                    sortedIm = nat_images(ind(valnum));
                    im = readimage(imds, find(contains(imnames,sortedIm),1));
                    preppedImage = prepImage(im);
                    imtoprint3=double(imresize(preppedImage,[48,48]));
                    goodimcount=goodimcount+1;
                    goodimindices(goodimcount)=ind(valnum);
                else
                    imtoprint3=255*ones(48,48,3);
                end
                gintimage3(1+image_size*(valnum-1):48+image_size*(valnum-1),1+image_size*(mod(rollingwindow-1,3*col_images)):image_size*(mod(rollingwindow-1,3*col_images))+48,:)=imtoprint3;
            end
        end
        if sitee<=size(goodch,1)
            row1_bydepth(1+50*(siteno-1):50*(siteno-0),:,:)=gintimage3(1:50,:,:);
            row1_bydepthsq(1+50*(sitee-1):50*(sitee-0),:,:)=gintimage3(1:50,:,:);
        end
        figure
        image(gintimage3/255)
        imtosave = gintimage3/255;
        sitee=sitee
        if sitee<=size(goodch,1)
            filename = sprintf('roll_90by20to480_site%02d_.jpg',goodchsorted(sitee));
        else
            filename='goodch90by40to490.jpg';
        end
        imwrite(imtosave, [sprintf('%s/Code/data-loading-code-peterbranch/margescipts/figimages/', cfg.livingstone_lab),exp_name,'/',filename, 'tizi'], 'jpg')
        close all
        clear imtosave filename
        rollingwindowstartcol=90:20:480;
        for rollingwindow=1:size(rollingwindowstartcol,2)
            window=rollingwindowstartcol(rollingwindow):rollingwindowstartcol(rollingwindow)+20;
            mean_resp_ch=squeeze(mean(nanmean(natraster(site,window,:),2),1));
            mean_resp_ch=zscore(mean_resp_ch);
            [val,ind]=sort(mean_resp_ch,'descend');
            if sitee<=size(goodchsorted,1)
                toplot(rollingwindow,:)=smoothdata(nanmean(nanmean(natraster(site,:,ind(1:25,1)),3),1),'gaussian',[15 15]);
                site=site
                sitee=sitee
            end
        end
        site2=site
        sitee2=sitee
        if sitee<=size(goodchsorted,1)
            figure; hold on
            for roll=1:size(toplot,1)
                plot(1:size(rasters,2),toplot(roll,:),'color',color(ceil(roll*255/size(rollingwindowstart,2)),:),'linew',2)
            end
            box on
            set(gca,'tickdir','out','linew',2,'fontsize',19)
            filename = sprintf('times_90by20to480_site%02d.jpg',site);
            saveas(gcf,[sprintf('%s/Code/data-loading-code-peterbranch/margescipts/figimages/', cfg.livingstone_lab),exp_name,'/',filename, 'tizi'],'png')
            close all
        end
    end

    site3=site
    sitee3=sitee
    %  if sitee<=size(goodchsorted,1)
    figure
    image(row1_bydepth/255)
    imtosave = row1_bydepth/255;
    filename = ['best_50by20to400_6_std_unsorted',exp_name,'.png'];
    imwrite(imtosave, [sprintf('%s/Code/data-loading-code-peterbranch/margescipts/figimages/', cfg.livingstone_lab),exp_name,'/',filename, 'tizi'], 'jpg')
    close all
    figure
    image(row1_bydepth/255)
    imtosave = row1_bydepthsq/255;
    filename = ['best_50by20to400_6_std_sorted',exp_name,'.png'];
    imwrite(imtosave, [sprintf('%s/Code/data-loading-code-peterbranch/margescipts/figimages/', cfg.livingstone_lab),exp_name,'/',filename, 'tizi'], 'jpg')
    close all
    %  end
end
%     clear rasters metai natraster nat_images
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if myflag.split
    for sitee = 1:size(goodchsorted,1)+1
        if sitee<=size(goodchsorted,1)
            site=goodchsorted(sitee,1);
        else
            site=goodchsorted;
        end
        rollingwindowstartcol=60:10:450;
        for rollingwindow=1:size(rollingwindowstartcol,2)
            window=rollingwindowstartcol(rollingwindow):rollingwindowstartcol(rollingwindow)+10;
            mean_resp_ch=squeeze(mean(nanmean(natrasterA(site,window,:),2),1));
            mean_resp_ch=zscore(mean_resp_ch);
            [val,ind]=sort(mean_resp_ch,'descend');
            %   if sitee<=size(goodchsorted,1)
            toplotspAB(rollingwindow,:)=smoothdata(nanmean(nanmean(natrasterB(site,:,ind(1:25,1)),3),1),'gaussian',[15 15]);
            %   end
        end

        for rollingwindow=1:size(rollingwindowstartcol,2)
            window=rollingwindowstartcol(rollingwindow):rollingwindowstartcol(rollingwindow)+20;
            mean_resp_ch=squeeze(mean(nanmean(natrasterB(site,window,:),2),1));
            mean_resp_ch=zscore(mean_resp_ch);
            [val,ind]=sort(mean_resp_ch,'descend');
            %   if sitee<=size(goodchsorted,1)
            toplotspBA(rollingwindow,:)=smoothdata(nanmean(nanmean(natrasterA(site,:,ind(1:25,1)),3),1),'gaussian',[15 15]);
            %   end
        end
        for roll=1:size(toplotspAB,1)
            toplotavg(roll,:)=.5*(toplotspAB(roll,:)+toplotspBA(roll,:));
            toplotavgN(roll,:)=toplotavg(roll,:)/max(toplotavg(roll,:));
        end
        figure; hold on
        for roll=1:size(toplotspAB,1)
            plot(1:size(rastersA,2),.5*(toplotspAB(roll,:)+toplotspBA(roll,:)),'color',color(ceil(roll*255/size(rollingwindowstartcol,2)),:),'linew',2)
        end
        box on
        set(gca,'tickdir','out','linew',2,'fontsize',19)
        if sitee<=size(goodchsorted,1)
            filename = sprintf('times_60by10to450_sitee%02d_split.jpg',sitee);
        else
            filename = 'times_60by10to450_goodch_split.jpg';
        end
        saveas(gcf,[sprintf('%s/Code/data-loading-code-peterbranch/margescipts/figimages/', cfg.livingstone_lab),exp_name,'/',filename, 'tizi'],'png')
        close all

        figure; hold on
        for roll=1:size(toplotavgN,1)
            plot(1:size(toplotavgN(roll,:),2),toplotavgN(roll,:),'color',color(ceil(roll*255/size(rollingwindowstartcol,2)),:),'linew',2)
        end
        box on
        set(gca,'tickdir','out','linew',2,'fontsize',19)
        if sitee<=size(goodchsorted,1)
            filename = sprintf('times_60by10to450_sitee%02d_splitN.jpg',sitee);
        else
            filename = 'times_60by10to450_splitgoodchN.jpg';
        end
        saveas(gcf,[sprintf('%s/Code/data-loading-code-peterbranch/margescipts/figimages/', cfg.livingstone_lab),exp_name,'/',filename, 'tizi'],'png')
        close all
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if myflag.times
    for sitee = 1:size(goodchsorted,1)+1
        if sitee<=size(goodchsorted,1)
            site=goodchsorted(sitee,1);
        else
            site=goodchsorted;
        end
        rollingwindowstartcol=60:10:450;
        for rollingwindow=1:size(rollingwindowstartcol,2)
            window=rollingwindowstartcol(rollingwindow):rollingwindowstartcol(rollingwindow)+10;
            mean_resp_ch=squeeze(mean(nanmean(natraster(site,window,:),2),1));
            mean_resp_ch=zscore(mean_resp_ch);
            [val,ind]=sort(mean_resp_ch,'descend');

            toplot(rollingwindow,:)=smoothdata(nanmean(nanmean(natraster(site,:,ind(1:25,1)),3),1),'gaussian',[15 15]);
            toplotN(rollingwindow,:)=(smoothdata(nanmean(nanmean(natraster(site,:,ind(1:25,1)),3),1),'gaussian',[15 15]))/max(smoothdata(nanmean(nanmean(natraster(site,:,ind(1:25,1)),3),1),'gaussian',[15 15]));

        end


        figure; hold on
        for roll=1:size(toplot,1)
            plot(1:size(rasters,2),toplot(roll,:),'color',color(ceil(roll*255/size(rollingwindowstartcol,2)),:),'linew',2)
        end
        box on
        set(gca,'tickdir','out','linew',2,'fontsize',19)
        if sitee<=size(goodchsorted,1)
            filename = sprintf('times_60by10to450_sitee%02d.jpg',sitee);
        else
            filename = 'times_60by10to450_goodch.jpg';
        end
        saveas(gcf,[sprintf('%s/Code/data-loading-code-peterbranch/margescipts/figimages/', cfg.livingstone_lab),exp_name,'/',filename, 'tizi'],'png')
        close all

        figure; hold on
        for roll=1:size(toplot,1)
            plot(1:size(rasters,2),toplotN(roll,:),'color',color(ceil(roll*255/size(rollingwindowstartcol,2)),:),'linew',2)
        end
        box on
        set(gca,'tickdir','out','linew',2,'fontsize',19)
        if sitee<=size(goodchsorted,1)
            filename = sprintf('times_60by10to450_sitee%02dN.jpg',sitee);
        else
            filename = 'times_60by10to450_goodchN.jpg';
        end
        saveas(gcf,[sprintf('%s/Code/data-loading-code-peterbranch/margescipts/figimages/', cfg.livingstone_lab),exp_name,'/',filename, 'tizi'],'png')
        close all
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%
% figure; hold on
%         for colr=1:40
%             plot([0 1],[41-colr,41-colr],'color',color(ceil((41-colr)*256/40),:),'linew',15)
%         end
%          filename='legend.jpg'
%          saveas(gcf,['~/Neurobio/LivingstoneLab/Code/data-loading-code-peterbranch/margescipts/figimages/',exp_name,'/',filename],'jpg')



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% parfor iim = 1:size(imds.Files,1)
%     im = readimage(imds,iim);
%     preppedImage = prepImage(im);
%     smallim = flipud(imresize(preppedImage, [smallimsize smallimsize]));
%     sim_mat(:,:,:,iim) = smallim;
% end

%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% image_dir = '/media/ks161/Neurobio/LivingstoneLab/Stimuli/fewer_occlusion';
% addpath(genpath(image_dir))
% imds = imageDatastore('/media/ks161/Neurobio/LivingstoneLab/Stimuli/fewer_occlusion','IncludeSubfolders',true);
%
net=alexnet;
inputSize = net.Layers(1).InputSize;
%  analyzeNetwork(net)
augimds = augmentedImageDatastore(inputSize(1:2),imds,'ColorPreprocessing','gray2rgb');
%
bylayercorrcoef=nan(20,size(imnames,1),size(imnames,1));
clear imnames
layer='conv1';
features = activations(net,augimds,layer,'OutputAs','rows');
bylayercorrcoef(1,:,:)=corrcoef(features');
clear features
layer='relu1';
features = activations(net,augimds,layer,'OutputAs','rows');
bylayercorrcoef(2,:,:)=corrcoef(features');
clear features
layer='norm1';
features = activations(net,augimds,layer,'OutputAs','rows');
bylayercorrcoef(3,:,:)=corrcoef(features');
clear features
layer='pool1';
features = activations(net,augimds,layer,'OutputAs','rows');
bylayercorrcoef(4,:,:)=corrcoef(features');
clear features
layer='conv2';
features = activations(net,augimds,layer,'OutputAs','rows');
bylayercorrcoef(5,:,:)=corrcoef(features');
clear features
layer='relu2';
features = activations(net,augimds,layer,'OutputAs','rows');
bylayercorrcoef(6,:,:)=corrcoef(features');
clear features
layer='norm2';
features = activations(net,augimds,layer,'OutputAs','rows');
bylayercorrcoef(7,:,:)=corrcoef(features');
clear features
layer='pool2';
features = activations(net,augimds,layer,'OutputAs','rows');
bylayercorrcoef(8,:,:)=corrcoef(features');
clear features
layer='conv3';
features = activations(net,augimds,layer,'OutputAs','rows');
bylayercorrcoef(9,:,:)=corrcoef(features');
clear features
layer='relu3';
features = activations(net,augimds,layer,'OutputAs','rows');
bylayercorrcoef(10,:,:)=corrcoef(features');
clear features
layer='conv4';
features = activations(net,augimds,layer,'OutputAs','rows');
bylayercorrcoef(11,:,:)=corrcoef(features');
clear scorei features
layer='relu4';
features = activations(net,augimds,layer,'OutputAs','rows');
bylayercorrcoef(12,:,:)=corrcoef(features');
clear features
layer='conv5';
features = activations(net,augimds,layer,'OutputAs','rows');
bylayercorrcoef(13,:,:)=corrcoef(features');
clear features
layer='relu5';
features = activations(net,augimds,layer,'OutputAs','rows');
bylayercorrcoef(14,:,:)=corrcoef(features');
clear features
layer='pool5';
features = activations(net,augimds,layer,'OutputAs','rows');
bylayercorrcoef(15,:,:)=corrcoef(features');
clear features
layer='fc6';
features = activations(net,augimds,layer,'OutputAs','rows');
bylayercorrcoef(16,:,:)=corrcoef(features');
clear features
layer='relu6';
features = activations(net,augimds,layer,'OutputAs','rows');
bylayercorrcoef(17,:,:)=corrcoef(features');
clear features
layer='fc7';
features = activations(net,augimds,layer,'OutputAs','rows');
bylayercorrcoef(18,:,:)=corrcoef(features');
clear features
layer='relu7';
features = activations(net,augimds,layer,'OutputAs','rows');
bylayercorrcoef(19,:,:)=corrcoef(features');
clear features
layer='fc8';
features = activations(net,augimds,layer,'OutputAs','rows');
bylayercorrcoef(20,:,:)=corrcoef(features');
clear features

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear gintimage3 imds meta stim_xy net mean_resp_ch imIndex allimages imtosave evk_resp mean_frtrials preppedImage row1_bydepth
clear goodchCCbybinz val uniqueImage im augimds
imageRespbinnedz=single(imageRespbinnedz);
goodchCCbybinz=nan(rasterlength/binsize,size(imageRespbinnedz,3),size(imageRespbinnedz,3),'single');
for bin=1:40
    goodchCCbybinz(bin,:,:)=corrcoef(squeeze(single(imageRespbinnedz(:,bin,:))));
end
%
% for bin=1:rasterlength/binsize
% goodchCCbybinz(bin,:,:)=squeeze(goodchCCbybinz(bin,:,:)).*~(eye(size(squeeze(goodchCCbybinz(bin,:,:)))));
% end

clear imageRespbinnedz

for bin=1:40
    for v=1:size(goodchCCbybinz,2)
        for h=1:size(goodchCCbybinz,2)
            if v==h
                goodchCCbybinz(bin,v,h)=0;
            end
        end
    end
end
% figure
% imagesc(squeeze(range3bybinCCzerodiagz(bin,:,:)))
% filename=['range3_CCzerodiagz_timebin ',num2str(bin),'.png'];
% print(gcf,['/media/ks161/Neurobio/LivingstoneLab/Code/data-loading-code-peterbranch/margescipts/figimages/',exp_name,'/',filename],'-dpng','-r600');
% close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
corrlayerbybinz=nan(20,40);
corrlayerbybinzN=nan(20,40);
for layer=1:20
    for bin=1:40
        corrlayerbybinz(layer,bin)=corr2(squeeze(goodchCCbybinz(bin,:,:)),squeeze(bylayercorrcoef(layer,:,:)));
    end
    corrlayerbybinzN(layer,:)=(corrlayerbybinz(layer,:)-min(corrlayerbybinz(layer,:),[],2))/(max(corrlayerbybinz(layer,:),[],2)-min(corrlayerbybinz(layer,:),[],2));
end
clear goodchCCbybinz
colorjet=colormap(jet);
figure;hold on
for layer=1:20
    plot(1:40,squeeze(corrlayerbybinz(layer,:)),'-','color',colorjet(round(layer*255/21),:),'linew',3)
end
xticks(2:10:40);
xticklabels(binwindows(2:10:40))
filename=['goodch10ms',exp_name,'.png'];
xlabel('timebin')
ylabel('correlation')
set(gca,'tickdir','out','linew',2)
box on
print(gcf,['/media/ks161/Neurobio/LivingstoneLab/Code/data-loading-code-peterbranch/margescipts/figimages/',exp_name,'/',filename],'-dpng','-r600');
close all


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure;hold on
for layer=1:20
    plot(1:40,squeeze(corrlayerbybinzN(layer,:)),'-','color',colorjet(round(layer*255/21),:),'linew',3)
end
filename=['goodch10ms',exp_name,'N.png'];
xticks(2:10:40);
xticklabels(binwindows(2:10:40))
xlabel('timebin')
ylabel('correlation')
set(gca,'tickdir','out','linew',2)
box on
print(gcf,['/media/ks161/Neurobio/LivingstoneLab/Code/data-loading-code-peterbranch/margescipts/figimages/',exp_name,'/',filename],'-dpng','-r600');
close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%handims
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%cartoonfaceims

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% natraster = nan(size(rasters,1),size(rasters,2), max(imIndex),1); % initialize natraster




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


