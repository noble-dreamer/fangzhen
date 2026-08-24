% 缺陷模型STL文件生成脚本
% 功能：读取thickness_map mat文件，扩展为500×500，生成STL文件
% 日期: 2024

clear; clc; close all;

%% ========================================================================
%  参数设置和文件夹创建
%% ========================================================================

% 目标网格尺寸和物理尺寸
target_size = [500, 500];
plate_thickness = 3; % 板厚度 [mm]
plate_length = 500; % 板长度 [mm]
plate_width = 500;  % 板宽度 [mm]

% 创建输出文件夹
output_folder = 'STLModel';
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
    fprintf('创建文件夹: %s\n', output_folder);
end

%% ========================================================================
%  自动查找并处理mat文件
%% ========================================================================

% 查找所有thickness_map_*.mat文件
mat_files = dir('Thickness_maps(baseline)/thickness_map_*.mat');

if isempty(mat_files)
    fprintf('错误: 未找到thickness_map_*.mat文件！\n');
    fprintf('请先运行Generate_thickness_examples.m生成缺陷模型\n');
    return;
end

fprintf('找到 %d 个缺陷模型文件\n', length(mat_files));

% 处理每个mat文件
for i = 1:length(mat_files)
    mat_filename = mat_files(i).name;
    mat_filepath = fullfile('Thickness_maps(baseline)', mat_filename);
    fprintf('\n正在处理: %s\n', mat_filename);
    
    % 加载mat文件
    try
        load(mat_filepath);
        fprintf('  成功加载: %s\n', mat_filename);
    catch ME
        fprintf('  错误: 无法加载 %s - %s\n', mat_filename, ME.message);
        continue;
    end
    
    % 检查thickness变量是否存在
    if ~exist('thickness', 'var')
        fprintf('  错误: mat文件中缺少thickness变量\n');
        continue;
    end
    
    fprintf('  原始尺寸: %d×%d\n', size(thickness, 1), size(thickness, 2));
    
    % 扩展thickness数据到500×500
    expanded_thickness = expand_thickness_data(thickness, target_size);
    fprintf('  扩展后尺寸: %d×%d\n', size(expanded_thickness, 1), size(expanded_thickness, 2));
    
    % 创建统一的顶点和面片
    [V, F] = build_closed_plate(expanded_thickness, plate_length, plate_width);
    
    % 生成STL文件名
    [~, base_name, ~] = fileparts(mat_filename);
    stl_filename = fullfile(output_folder, [base_name '.stl']);
    
    % 生成二进制STL文件
    try
        stlwrite_binary(stl_filename, V, F);
        fprintf('  ✓ STL文件已生成: %s\n', stl_filename);
        
    catch ME
        fprintf('  ✗ 生成STL文件失败: %s\n', ME.message);
        continue;
    end
    
    % 生成预览图（只对前3个模型）
    if i <= 3
        generate_preview(expanded_thickness, i, output_folder);
    end
end

fprintf('\n所有STL文件生成完成！\n');

%% ========================================================================
%  辅助函数
%% ========================================================================

function expanded_thickness = expand_thickness_data(thickness, target_size)
    % 将120×120的thickness数据扩展到500×500
    % 在中心放置原数据，周围用最大厚度值填充
    
    [orig_h, orig_w] = size(thickness);
    target_h = target_size(1);
    target_w = target_size(2);
    
    % 创建目标尺寸的矩阵，用最大厚度值填充
    max_thickness = max(thickness(:));
    expanded_thickness = max_thickness * ones(target_h, target_w);
    
    % 计算中心位置
    start_row = round((target_h - orig_h) / 2) + 1;
    end_row = start_row + orig_h - 1;
    start_col = round((target_w - orig_w) / 2) + 1;
    end_col = start_col + orig_w - 1;
    
    % 将原数据放置在中心
    expanded_thickness(start_row:end_row, start_col:end_col) = thickness;
end

function [V,F] = build_closed_plate(thk, Lx, Ly)
    % thk: nrows×ncols 的厚度(单位=mm)，底面Z=0，顶面Z=thk
    % Lx, Ly: 物理尺寸(mm)
    
    [nr, nc] = size(thk);
    x = linspace(0, Lx, nc);
    y = linspace(0, Ly, nr);
    [X,Y] = meshgrid(x,y);
    
    % 顶/底面顶点
    V_top    = [X(:), Y(:), thk(:)];
    V_bottom = [X(:), Y(:), zeros(numel(X),1)];
    
    % 把四个侧边按"条带"方式挤出，保证与顶/底边精确共点
    % 前(Y=0)：沿厚度方向挤出 nr_side 层，建议 6~12 层
    nr_side = 10; s = linspace(0,1,nr_side).';  % s=0 到 1
    % 每个侧面仅沿相应边的厚度做线性插值，保证最外层 s=1 与顶面点完全一致
    % front 边：行=1，所有列
    Zfront = s * (thk(1,:));  Xfront = repmat(x, nr_side, 1); Yfront = zeros(size(Xfront));
    % back 边：行=nr
    Zback  = s * (thk(end,:)); Xback  = repmat(x, nr_side, 1); Yback  = Ly*ones(size(Xback));
    % left 边：列=1
    Zleft  = s * (thk(:,1)).';  Yleft  = repmat(y, nr_side, 1); Xleft  = zeros(size(Yleft));
    % right 边：列=nc
    Zright = s * (thk(:,end)).'; Yright = repmat(y, nr_side, 1); Xright = Lx*ones(size(Yright));
    
    V_front = [Xfront(:), Yfront(:), Zfront(:)];
    V_back  = [Xback(:),  Yback(:),  Zback(:)];
    V_left  = [Xleft(:),  Yleft(:),  Zleft(:)];
    V_right = [Xright(:), Yright(:), Zright(:)];
    
    % 组装全局顶点（注意顺序：先顶/底，再四侧）
    V = [V_top; V_bottom; V_front; V_back; V_left; V_right];
    
    % 面片索引生成函数：把 r×c 条带格子转成三角形
    triStrip = @(nr,nc,offset,flip) ...
        local_quad_to_tris(nr, nc, offset, flip);
    
    % 顶/底 面的三角形（顶面法向朝 +Z，底面朝 -Z）
    F_top = triStrip(nr, nc,            0, false);
    F_bot = triStrip(nr, nc,         size(V_top,1), true); % 翻转以朝 -Z
    
    % 侧面条带三角形
    off = size(V_top,1) + size(V_bottom,1);
    F_front = triStrip(nr_side, nc, off, false);
    off = off + size(V_front,1);
    F_back  = triStrip(nr_side, nc, off, true);   % 反向以朝外
    off = off + size(V_back,1);
    F_left  = triStrip(nr_side, nr, off, false);
    off = off + size(V_left,1);
    F_right = triStrip(nr_side, nr, off, true);
    
    % 全部面片
    F = [F_top; F_bot; F_front; F_back; F_left; F_right];
end

function F = local_quad_to_tris(nr, nc, off, flip)
    % 把 nr×nc 的规则网格（按列主序排）转换成 2*(nr-1)*(nc-1) 个三角形
    % 顶点索引起点为 off+1
    idx = reshape(off + (1:(nr*nc)), nr, nc);
    i1 = idx(1:end-1, 1:end-1); i2 = idx(2:end, 1:end-1);
    i3 = idx(2:end,   2:end  ); i4 = idx(1:end-1, 2:end  );
    F1 = [i1(:), i2(:), i3(:)];
    F2 = [i1(:), i3(:), i4(:)];
    F  = [F1; F2];
    if flip, F = F(:,[1 3 2]); end
end

function stlwrite_binary(filename, V, F)
    % filename: .stl
    fid = fopen(filename,'wb'); 
    fwrite(fid, zeros(1,80), 'uint8');             % header
    fwrite(fid, size(F,1), 'uint32');
    % 计算法向
    P1 = V(F(:,2),:)-V(F(:,1),:);
    P2 = V(F(:,3),:)-V(F(:,1),:);
    N  = cross(P1,P2,2);
    N  = N ./ max(vecnorm(N,2,2), eps);
    for k=1:size(F,1)
        fwrite(fid, N(k,:), 'float32');
        fwrite(fid, V(F(k,1),:), 'float32');
        fwrite(fid, V(F(k,2),:), 'float32');
        fwrite(fid, V(F(k,3),:), 'float32');
        fwrite(fid, 0, 'uint16'); % attribute byte count
    end
    fclose(fid);
end

function generate_preview(thickness_map, model_id, output_folder)
    % 生成预览图
    
    figure('Visible', 'off');
    imagesc(thickness_map);
    colorbar;
    title(sprintf('厚度分布 - 模型 %d', model_id));
    xlabel('X (网格点)');
    ylabel('Y (网格点)');
    
    % 保存预览图
    preview_filename = fullfile(output_folder, sprintf('preview_%d.png', model_id));
    saveas(gcf, preview_filename);
    close(gcf);
end

