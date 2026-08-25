% 批量生成厚度缺陷模型用于导波声场研究
% 预览 imagesc(thickness);colorbar
clear; clc; close all;

%% 参数设置
n_samples = 2000;                      % 生成样本数量
dx = 1000;                          % 网格间距 (μm)
n = [120, 120];                   % 网格尺寸 [nz, nx] (z方向1000点, x方向500点)
rdefect_max = 50;                   % 最大缺陷半径 (网格点数，约400mm)
rdefect_min = 5;                    % 最小缺陷半径 (网格点数，约30mm)
Thicknessmax = 3;                 % 最大厚度 (mm)
Thicknessmin = 1.5;                 % 最小厚度 (mm)

% 缺陷区域限制 (网格点坐标，Z X坐标系)
defect_region.z_min = 3;          % Z方向最小位置
defect_region.z_max = 117;          % Z方向最大位置  
defect_region.x_min = 3;          % X方向最小位置
defect_region.x_max = 117;          % X方向最大位置

%% 创建存储文件夹
output_folder = 'Defects';
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
    fprintf('创建文件夹: %s\n', output_folder);
end

%% 批量生成
fprintf('开始生成 %d 个厚度缺陷模型...\n', n_samples);

% 创建预览数据存储
preview_data = {};
preview_count = 0;

for i = 1:n_samples
    fprintf('生成第 %d/%d 个模型...', i, n_samples);
    
    % 生成厚度图
    thickness = Rand_thickness_defect_model_1m_1m(n, rdefect_min, rdefect_max, dx, Thicknessmin, Thicknessmax, defect_region);
    
    % 保存数据
    filename = fullfile(output_folder, sprintf('Defect_%d.mat', i));
    save(filename, 'thickness');   
    fprintf(' 完成\n');
    
%     % 每5张保存一个预览数据
%     if mod(i, 5) == 0
%         preview_count = preview_count + 1;
%         preview_data{preview_count} = thickness;
%     end
end

% %% 预览模块
% if preview_count > 0
%     fprintf('\n生成预览图...\n');
%     
%     % 计算子图布局
%     n_rows = ceil(sqrt(preview_count));
%     n_cols = ceil(preview_count / n_rows);
%     
%     figure('Position', [100, 100, 300*n_cols, 250*n_rows]);
%     
%     for j = 1:preview_count
%         subplot(n_rows, n_cols, j);
%         imagesc(preview_data{j}');  % 转置以正确显示Z-X坐标
%         colorbar;
%         caxis([Thicknessmin, Thicknessmax]);
%         title(sprintf('样本 %d', j*5), 'FontSize', 10);
%         xlabel('Z方向 (网格点)');
%         ylabel('X方向 (网格点)');
%         axis equal;
%         axis tight;
%     end
%     
%     % 设置整体标题和颜色映射
%     sgtitle(sprintf('厚度缺陷模型预览 (每5个样本显示1个，共%d个)', preview_count), 'FontSize', 12);
%     colormap(jet);
%     
%     % 保存预览图
%     preview_filename = fullfile(output_folder, 'thickness_preview.png');
%     saveas(gcf, preview_filename);
%     fprintf('预览图保存为: %s\n', preview_filename);
% end

%% 统计信息
fprintf('\n=== 生成完成 ===\n');
fprintf('样本数量: %d\n', n_samples);
fprintf('存储路径: %s\n', output_folder);
fprintf('厚度范围: %.1f - %.1f mm\n', Thicknessmin, Thicknessmax);
fprintf('网格尺寸: %d × %d (Z×X)\n', n(1), n(2));
fprintf('网格间距: %d μm\n', dx);
fprintf('缺陷半径: %d-%d 网格点 (%.0f-%.0fmm)\n', rdefect_min, rdefect_max, rdefect_min*dx/1000, rdefect_max*dx/1000);
fprintf('缺陷区域: Z[%d-%d], X[%d-%d] (网格点)\n', ...
    defect_region.z_min, defect_region.z_max, defect_region.x_min, defect_region.x_max); 