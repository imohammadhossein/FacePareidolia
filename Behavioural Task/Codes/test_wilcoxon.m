clc;
clear all;
close all;

ax=imread('test.jpeg');

diam=100;
bubbles=5;
bub_cent=[size(ax,1)./2;size(ax,2)./2];
bub_scatr=50;

x=[1:size(ax,1);1:size(ax,2)];

% ax_modif=zeros(size(ax,1));
% for i=1:255
%     for j=1:255
%         if ax(i,j,1)==64 && ax(i,j,1)==ax(i+1,j+1,1)
%             ax_modif(i,j)=0;
%             ax_modif(i+1,j+1)=0;
%         else
%             ax_modif(i,j)=ax(i,j,1);
%         end
%     end
% end
% ax=ax_modif;

covari=diam.*eye(2,2);
bubbled=zeros(size(ax,1));
bubble_coll=zeros(size(ax,1),size(ax,1));
for bub=1:bubbles
    mu=fix(bub_cent+randn(2,1)*bub_scatr);
    for i=1:size(ax,1)
        for j=1:size(ax,2)
            f(i,j)=(1./(sqrt(2*pi)*sqrt(norm(covari))))*exp(-0.5*[[x(1,i);x(1,j)]-mu]'*inv(covari)*[[x(1,i);x(1,j)]-mu]);
        end
    end
    bubble=(f/max(max(f)))-min(min(f));
    bubble_coll=bubble_coll+bubble;
    bubbled=bubbled+bubble.*double(ax(:,:,1));
end
imshow(uint8(bubbled))
bubble_coll=(bubble_coll./max(max(bubble_coll)))-min(min(bubble_coll));
final_pic=zeros(size(ax,1));
for i=1:size(ax,1)
    for j=1:size(ax,2)
        if ax(i,j,1)==64
            final_pic(i,j)=64;
        else
            final_pic(i,j)=(ax(i,j,1)).*bubble_coll(i,j);
        end
    end
end
figure
imshow(bubble_coll)
figure
imshow(uint8(final_pic))

% bubbled(uint8(bubbled)==0)=64;
% imshow(uint8(bubbled))

