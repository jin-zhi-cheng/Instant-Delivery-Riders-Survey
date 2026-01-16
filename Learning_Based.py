
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import shap
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import pandas as pd
import optuna
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import os
import optuna
from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import numpy as np


from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def train_catboost_classification(X_train, y_train, X_test, y_test, n_trials):
    """
    使用 Optuna 进行超参数优化，训练 CatBoost 分类模型，并评估其性能。

    参数：
    - X_train, y_train: 训练集特征和目标变量
    - X_test, y_test: 测试集特征和目标变量
    - n_trials: Optuna 的优化迭代次数

    返回：
    - best_model: 训练好的 CatBoost 模型
    - y_pred: 测试集预测概率
    """
    import optuna
    from catboost import CatBoostClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
    
    # 调整 Optuna 的日志级别，抑制每个 trial 的日志输出
    optuna.logging.set_verbosity(optuna.logging.ERROR)

    def cat_objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
            'depth': trial.suggest_int('depth', 1, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 100.0, log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
            'random_strength': trial.suggest_float('random_strength', 0.0, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.05, 1.0),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 100),
            'loss_function': 'Logloss',  # 修改为分类损失函数
            'verbose': False,
            'random_seed': 42,
            # 'task_type': 'GPU',    # 使用 GPU 加速
            # 'devices': '0',        # 指定 GPU 设备（如果有多个 GPU，可以指定特定的 GPU）
        }

        model = CatBoostClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=100, verbose=False, use_best_model=True)
        preds = model.predict_proba(X_test)[:, 1]  # 获取正类的概率
        auc = roc_auc_score(y_test, preds)  # 使用AUC作为优化目标
        
        trial.report(auc, step=1)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
 
        return -auc  # 因为Optuna默认最小化，所以返回负的AUC

    # 创建 Optuna 的 study 对象
    study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
    study.optimize(cat_objective, n_trials = n_trials,  n_jobs=8)

    # 输出最佳超参数
    print('catboost 最佳超参数:', study.best_params)

    # 使用最佳超参数重新训练模型
    best_params = study.best_params
    best_params['loss_function'] = 'Logloss'
    best_params['verbose'] = False
    best_params['random_seed'] = 42

    best_model = CatBoostClassifier(**best_params)
    best_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=50, verbose=False)

    # 在测试集上进行预测
    y_pred = best_model.predict_proba(X_test)[:, 1]  # 获取正类的概率

    # 评估模型性能
    print("catboost AUC:", roc_auc_score(y_test, y_pred))
    print("catboost Accuracy:", accuracy_score(y_test, y_pred > 0.5))
    print("catboost F1 Score:", f1_score(y_test, y_pred > 0.5))

    return best_model, y_pred

def plot_feature_importance_donut(shap_values, feature_names, save_path=None, num_var=5):
    """
    绘制环形特征重要性图

    参数：
    - shap_values: SHAP 计算的特征贡献值（可以是绝对值求和后的贡献比例）
    - feature_names: 特征名称列表
    - save_path: 保存路径（可选）
    - num_var: 变量个数
    - colors: 前10特征及其他部分的颜色列表
    """

    colors=['#22AFE4', '#52A06C', '#94B1D9', '#225B66', '#E76A2A',
                '#F9C192', '#FEC111', '#D2817E', '#94191C', '#6F3A96', '#B1B2B6']
    # colors=colors_list[:num_var]
    
    # 提取 SHAP 值的数值部分
    shap_values_array = shap_values.values
    
    # 计算每个特征的平均绝对 SHAP 值
    feature_importance = np.abs(shap_values_array).mean(axis=0)
    feature_importance_percentage = feature_importance / feature_importance.sum() * 100

    
    #* 对特征按贡献率排序，取前10特征
    sorted_idx = np.argsort(feature_importance_percentage)[::-1]  # 从大到小排序
    top_10_idx = sorted_idx[:num_var]
    others_idx = sorted_idx[num_var:]

    # 重新整理特征名和重要性
    top_10_names = [feature_names[i] for i in top_10_idx]
    top_10_importance = feature_importance_percentage[top_10_idx]
    others_importance = feature_importance_percentage[others_idx].sum()  # 计算“Others”总贡献率

    # 合并前10和“Others”
    final_names = top_10_names + ['Others']
    final_importance = np.append(top_10_importance, others_importance)

    # 绘制环形图
    plt.figure(figsize=(6, 6))
    wedges, texts, autotexts = plt.pie(
        final_importance,
        # labels=final_names,
        labels=None,  # 不直接在环形图上显示标签
        autopct='%1.1f%%',  # 显示百分比
        startangle=90,
        pctdistance=1.15,  # 百分比文本距离
        colors=colors[:len(final_names)],  # 使用指定颜色
        wedgeprops=dict(width = 0.6, edgecolor = 'white', linewidth = 3),
        textprops={'fontsize': 18}  # 设置字体大小
    )

    # 添加图例
    plt.legend(
        handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], markersize=10)
                 for i in range(len(final_names))],
        labels=final_names,
        loc='center right',  # 图例位置
        bbox_to_anchor=(1.75, 0.5),  # 图例位置稍微偏移
        edgecolor='black',     # 设置边框颜色为黑色
        fancybox=False, 
        fontsize=18
    )

    plt.savefig(save_path, bbox_inches='tight')
    plt.clf()

def explain_shap_catboost_new(model, X_train, X_test, feature_names, save_path=None, depvar=None, num_var=None):
    """
    使用 SHAP 解释 CatBoost 模型。

    参数：
    - model: 训练好的 CatBoost 模型
    - X_train: 训练集特征（用于构建 SHAP 解释器）
    - X_test: 测试集特征（用于计算 SHAP 值）
    - feature_names: 特征名称列表
    - plot_type: SHAP 图类型（"bar" 或 "summary"）

    返回：
    - shap_values: 计算得到的 SHAP 值
    """
    import shap

    plt.rcParams['font.family'] = 'Times New Roman'
    # 如果 X_train 不是 DataFrame，将其转换为 DataFrame 并添加列名
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train, columns=feature_names)
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test, columns=feature_names)

    # 初始化 SHAP 解释器
    # 使用不同的初始化方式来避免版本兼容性问题
    try:
        # 方法1: 不传入背景数据
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_test)
    except Exception as e1:
        try:
            # 方法2: 使用feature_perturbation参数
            explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
            shap_values = explainer(X_test)
        except Exception as e2:
            try:
                # 方法3: 使用较小的背景数据集
                background_sample = X_train.sample(min(100, len(X_train)), random_state=42)
                explainer = shap.TreeExplainer(model, background_sample)
                shap_values = explainer(X_test)
            except Exception as e3:
                # 方法4: 使用Explainer基类
                explainer = shap.Explainer(model, X_train)
                shap_values = explainer(X_test)
                print(f"使用了Explainer基类，原始错误: {e1}")

    plt.rcParams.update({'font.size': 18})

    # 计算每个特征的平均绝对SHAP值和95%置信区间
    import numpy as np
    from scipy import stats

    # 获取SHAP值的绝对值
    # 处理不同版本SHAP返回的对象类型
    if hasattr(shap_values, 'values'):
        abs_shap_values = np.abs(shap_values.values)
    else:
        abs_shap_values = np.abs(shap_values)

    # 计算每个特征的平均绝对SHAP值
    mean_abs_shap = np.mean(abs_shap_values, axis=0)

    # 计算95%置信区间
    confidence_intervals = []
    for i in range(abs_shap_values.shape[1]):
        feature_shap = abs_shap_values[:, i]
        # 使用bootstrap方法计算置信区间
        n_bootstrap = 1000
        bootstrap_means = []
        n_samples = len(feature_shap)

        for _ in range(n_bootstrap):
            # 有放回抽样
            bootstrap_sample = np.random.choice(feature_shap, size=n_samples, replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))

        # 计算95%置信区间
        ci_lower = np.percentile(bootstrap_means, 2.5)
        ci_upper = np.percentile(bootstrap_means, 97.5)
        confidence_intervals.append((ci_lower, ci_upper))

    # 创建自定义的条形图，包含置信区间
    fig, ax = plt.subplots(figsize=(20, 10))

    # 按重要性排序
    sorted_indices = np.argsort(mean_abs_shap)[::-1]
    sorted_means = mean_abs_shap[sorted_indices]
    sorted_features = [feature_names[i] for i in sorted_indices]
    sorted_cis = [confidence_intervals[i] for i in sorted_indices]

    # 计算误差条的长度
    ci_lower_errors = [sorted_means[i] - sorted_cis[i][0] for i in range(len(sorted_means))]
    ci_upper_errors = [sorted_cis[i][1] - sorted_means[i] for i in range(len(sorted_means))]

    # 绘制条形图和误差条
    y_pos = np.arange(len(sorted_features))
    bars = ax.barh(y_pos, sorted_means, color='#99DAEE')

    # 添加置信区间误差条
    ax.errorbar(sorted_means, y_pos,
                xerr=[ci_lower_errors, ci_upper_errors],
                fmt='none', color='black', capsize=6, capthick=1)

    # 设置标签和标题
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_features)
    ax.set_xlabel('Mean |SHAP value| (95% CI)', fontsize=22)
    # ax.set_title('Feature Importance with 95% Confidence Intervals', fontsize=22)

    # 调整字体大小
    ax.tick_params(axis='both', labelsize=22)

    # 添加网格
    ax.grid(True, axis='x', alpha=0.3)

    # 调整布局
    plt.tight_layout()

    # 保存图像
    plt.savefig(save_path + f'Cat_{depvar}_contributor_with_CI.pdf', bbox_inches='tight')
    plt.clf()

    # 同时保存原始的SHAP summary plot（不带置信区间）
    shap.summary_plot(shap_values, features=X_test, feature_names=feature_names, plot_type="bar", show=False)
    plt.savefig(save_path + f'Cat_{depvar}_contributor_original.pdf', bbox_inches='tight')
    plt.clf()

    plot_feature_importance_donut(shap_values, feature_names, save_path+f'Cat_{depvar}_feature_importance_donut.pdf', num_var)

    # 调整图的大小
    plt.figure(figsize=(12, 6))
    shap.plots.beeswarm(shap_values, max_display = 11, show=False)
    # 再次调整字体大小
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel(plt.gca().get_xlabel(), fontsize=18)
    plt.ylabel(plt.gca().get_ylabel(), fontsize=18)
    
    # 获取颜色条对象并调整字体大小
    colorbar = plt.gcf().axes[-1]  # 获取颜色条轴对象（通常是最后一个轴）
    colorbar.tick_params(labelsize=18)  # 调整颜色条刻度字体大小
    colorbar.set_ylabel(colorbar.get_ylabel(), fontsize=18)  # 调整颜色条标签字体大小
    
    plt.savefig(save_path + f'Cat_{depvar}_beeswarm.pdf', bbox_inches='tight')
    plt.clf()

    # 绘制SHAP依赖图
    os.makedirs(save_path + f'Cat_dependence/{depvar}/', exist_ok=True)
    for i, feature in enumerate(feature_names):
        
        # 创建一个图形和主坐标轴
        fig, ax1 = plt.subplots(figsize=(8, 6))

        shap.plots.scatter(shap_values[:, feature], 
                           alpha=1,
                           dot_size=10,
                           show=False,
                           hist=False,
                           ax=ax1,
                           color= '#38a1db', 
                           )
        
        ax1.set_xlabel(f'{feature}', fontsize=20)
        ax1.set_ylabel('SHAP value', fontsize=20)
        
        ax1.tick_params(axis='x', labelsize=20)
        ax1.tick_params(axis='y', labelsize=20)
        
        # 添加网格线
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='lightgray')
        # 添加 SHAP=0 的横线
        plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
        
        # 5.2 创建第二个 y 轴用于绘制部分依赖图
        ax2 = ax1.twinx()

        # make a standard partial dependence plot with a single SHAP value overlaid
        shap.partial_dependence_plot(
            feature,
            model.predict,
            X_test,
            show=False,
            ice=False,  
            hist=False,
            ax=ax2,
        )
        
        # 然后获取 ax2 上的所有线条
        pdp_lines = ax2.get_lines()

        # 如果只想改最后画上去的那一条线，可按索引 [-1]
        if pdp_lines:
            pdp_lines[-1].set_color('#f47983')     # 修改颜色
            pdp_lines[-1].set_linewidth(2)     # 修改线宽
                
        # 5) 设置右侧 y 轴标签
        ax2.set_ylabel('Partial dependence', fontsize=20)
        
        ax2.spines["right"].set_visible(True)  
        
        # 6) 将第二个 y 轴刻度与标签放在右边，并设置字体大小
        ax2.yaxis.set_label_position("right")  
        ax2.yaxis.set_ticks_position("right")
        ax2.tick_params(axis='y', labelsize=20)
        
        # 调整布局并显示图形
        plt.tight_layout()
        plt.savefig(save_path + f'Cat_dependence/{depvar}/Cat_{depvar}_{feature}_mixed_partial_dependence_plot.pdf', bbox_inches='tight')
        plt.clf()
        
    return shap_values        
        
