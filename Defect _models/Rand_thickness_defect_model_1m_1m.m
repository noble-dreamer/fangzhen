function thickness_map = Rand_thickness_defect_model_1m_1m(n, rdefect_min, rdefect_max, dx, thickness_min, thickness_max, defect_region)
    % 生成随机厚度缺陷模型，用于导波声场研究
    % 输入:
    %   n - 网格尺寸 [nz, nx] (Z X坐标系：第一个为z，第二个为x)
    %   rdefect_min - 最小缺陷半径 (网格点数)
    %   rdefect_max - 最大缺陷半径 (网格点数) 
    %   dx - 网格间距
    %   thickness_min - 最小厚度 (mm)
    %   thickness_max - 最大厚度 (mm)
    %   defect_region - 缺陷区域限制结构体
    % 输出:
    %   thickness_map - 厚度分布图 (单位: mm)
    
    % 使用现有的图像生成缺陷形状
    v = imread('1.jpg');
    theta = randi(360); % 随机旋转角度
    v = imrotate(v, theta, 'bilinear', 'crop');
    
    v = rgb2gray(v);
    v(v < 5) = max(v(:)) + 5;  % 标记均匀板区域
    v = double(v);
    v = imresize(v, n);
    
    % 生成随机多边形掩膜
    [x, y] = Polygon_mask(n, rdefect_min, rdefect_max, dx, defect_region);
    bw = poly2mask(x, y, n(1), n(2));
    
    bw = double(bw);
    v = v .* bw;
    v(v==0) = max(v(:)) + 5;  % 无缺陷区域
    
    % 将图像灰度值映射到厚度范围
    thickness_map = rescale(v, thickness_min, thickness_max);
    
    % 高斯滤波使厚度变化更平滑真实
    thickness_map = imgaussfilt(thickness_map, 2);
    
end

function [x, y] = Polygon_mask(n, rdefect_min, rdefect_max, dx, defect_region)
    % 生成随机多边形掩膜
    N = randi([20, 50]);              % 多边形边数 (增加到20-50，更不规则)
    a = sort(rand(N,1))*2*pi;         % 角度
    
    % 在最小和最大缺陷半径之间随机选择
    rtemp = randi([rdefect_min, rdefect_max]);

%     r = randi([max(rdefect_min, rtemp-5), rtemp], N, 1);
    % 增加半径变化范围，使形状更不规则
    e = rand() * 0.2; % 生成0到0.3之间的随机小数
    r_variation = max(5, round(rtemp * e)); % 变化幅度为平均半径的0-30%
    r = randi([max(rdefect_min, rtemp-r_variation), rtemp], N, 1);
    
    x_poly = cos(a).*r;  % 多边形x坐标
    z_poly = sin(a).*r;  % 多边形z坐标
    
%     % 添加随机噪声使形状更不规则
%     noise_level = max(1, round(rtemp * 0.1)); % 噪声幅度为平均半径的10%
%     x_poly = x_poly + (rand(N,1) - 0.5) * noise_level * 2;
%     z_poly = z_poly + (rand(N,1) - 0.5) * noise_level * 2;
%     
%     x_poly = round(x_poly);                    
%     z_poly = round(z_poly);
    
    rmax = max(r(:));
    
    % 在指定区域内随机选择中心位置 (Z X坐标系：n(1)=nz, n(2)=nx)
    zc_min = max(defect_region.z_min + rmax, 1);
    zc_max = min(defect_region.z_max - rmax, n(1));  % n(1)为z方向
    xc_min = max(defect_region.x_min + rmax, 1);
    xc_max = min(defect_region.x_max - rmax, n(2));  % n(2)为x方向
    
    % 确保有效范围
    if zc_min >= zc_max
        zc = round((defect_region.z_min + defect_region.z_max) / 2);
    else
        zc = randi([zc_min, zc_max]);
    end
    
    if xc_min >= xc_max
        xc = round((defect_region.x_min + defect_region.x_max) / 2);
    else
        xc = randi([xc_min, xc_max]);
    end
    
    % 平移多边形到中心位置
    x_poly = x_poly + xc;
    z_poly = z_poly + zc;
    
    % 确保所有点在网格范围内
    x_poly = max(1, min(n(2), x_poly));  % x方向限制 (n(2)为x方向)
    z_poly = max(1, min(n(1), z_poly));  % z方向限制 (n(1)为z方向)
    
    % 为poly2mask函数准备坐标 (poly2mask需要 x=列坐标, y=行坐标)
    x = x_poly;  % X方向 = 列坐标
    y = z_poly;  % Z方向 = 行坐标
    
end 