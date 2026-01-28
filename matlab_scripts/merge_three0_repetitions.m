clear all
%%
ev_path = "/Users/tizianocausin/livingstone_lab_local/tiziano/data/three0_natrasterEv250313.mat";
od_path = "/Users/tizianocausin/livingstone_lab_local/tiziano/data/three0_natrasterOd250313.mat";
%%
load(ev_path)
load(od_path)
%%
natraster = (natrasterOd + natrasterEv)/2;
%%
save("/Users/tizianocausin/livingstone_lab_local/tiziano/data/three0_natraster250313.mat", "natraster", '-v7.3')
%%
imagesc(mean(natraster, 3))
figure
imagesc(mean(natrasterOd, 3))
figure
imagesc(mean(natrasterEv, 3))
%%
plot(mean(natraster, [1,3]))
figure
plot(mean(natrasterOd, [1,3]))
figure
plot(mean(natrasterEv, [1,3]))