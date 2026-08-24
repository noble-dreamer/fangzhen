% 将120×120缺陷模型扩展为1501×1501完整模型
% 用于导波声场研究中避免边界反射
% 坐标系：横向X，纵向Y
clear; clc; close all;

%% 参数设置
defect_folder = 'Defects';           % 缺陷模型存储文件夹
output_folder = 'Models';            % 完整模型输出文件夹
baseline_thickness = 3;              % 基础板厚度 (mm)
model_size = 1501;                   % 完整模型尺寸 1501×1501

% 缺陷放置位置（矩阵索引）
% 缺陷中心：X=751, Y=751（模型中心）
% 缺陷尺寸：120×120
defect_row_range = (751-60):(751+59);          % Y方向 (行): 中心在 751 (691:810)
defect_col_range = (751-60):(751+59);          % X方向 (列): 中心在 751 (691:810)

% 激励源位置（矩阵索引）
% 研究区域坐标系：X=251, Y=51
% 完整模型坐标系：X=751, Y=551
excitation_pos = [551, 751];         % [row, col] = [Y, X]

% 研究区域（矩阵索引，中心501×501区域）
study_row_range = 501:1001;          % Y方向
study_col_range = 501:1001;          % X方向

%% 创建输出文件夹
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
    fprintf('创建文件夹: %s\n', output_folder);
end

%% 确定要处理的缺陷文件范围
% 自动检测 Defect_i.mat 文件的最大编号
max_defect_id = 0;
defect_pattern = fullfile(defect_folder, 'Defect_*.mat');
all_defect_files = dir(defect_pattern);

for k = 1:length(all_defect_files)
    filename = all_defect_files(k).name;
    % 提取文件名中的数字
    tokens = regexp(filename, 'Defect_(\d+)\.mat', 'tokens');
    if ~isempty(tokens)
        defect_id = str2double(tokens{1}{1});
        max_defect_id = max(max_defect_id, defect_id);
    end
end

if max_defect_id == 0
    error('在文件夹 %s 中未找到符合格式 Defect_i.mat 的文件', defect_folder);
end

fprintf('检测到缺陷文件编号范围: 1 - %d\n', max_defect_id);
fprintf('开始扩展模型...\n');

%% 批量处理（按编号顺序）
processed_count = 0;
for i = 1:max_defect_id
    % 构建缺陷文件名
    defect_filename = sprintf('Defect_%d.mat', i);
    defect_file = fullfile(defect_folder, defect_filename);
    
    % 检查文件是否存在
    if ~exist(defect_file, 'file')
        fprintf('跳过不存在的文件: %s\n', defect_filename);
        continue;
    end
    
    % 加载缺陷模型
    try
        data = load(defect_file);
        defect_model = data.thickness;
        processed_count = processed_count + 1;
        
        fprintf('处理文件 %d: %s...', i, defect_filename);
        
        % 创建基础的1501×1501模型（厚度为3mm）
        thickness = ones(model_size, model_size) * baseline_thickness;
        
        % 将缺陷模型替换到指定位置
        thickness(defect_row_range, defect_col_range) = defect_model;
        
        % 保存扩展后的模型（保持相同的编号）
        output_filename = fullfile(output_folder, sprintf('Model_%d.mat', i));
        save(output_filename, 'thickness');
        
        fprintf(' 完成\n');
        
    catch ME
        fprintf(' 失败: %s\n', ME.message);
        continue;
    end
end

%% 生成统计信息
fprintf('\n=== 扩展完成 ===\n');
fprintf('检测到的缺陷文件编号范围: 1 - %d\n', max_defect_id);
fprintf('成功处理文件数量: %d\n', processed_count);
fprintf('完整模型尺寸: %d × %d\n', model_size, model_size);
fprintf('缺陷模型尺寸: 120 × 120\n');
fprintf('缺陷位置: 行[%d:%d], 列[%d:%d]\n', defect_row_range(1), defect_row_range(end), defect_col_range(1), defect_col_range(end));
fprintf('激励源位置: [%d, %d]\n', excitation_pos(1), excitation_pos(2));
fprintf('基础厚度: %.1f mm\n', baseline_thickness);
fprintf('输出文件夹: %s\n', output_folder);

fprintf('\n模型扩展处理完成！\n');
fprintf('输入: Defect_i.mat → 输出: Model_i.mat (保持编号一致)\n');
