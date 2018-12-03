###########################################
# Suppress matplotlib user warnings
# Necessary for newer version of matplotlib
import warnings
warnings.filterwarnings("ignore", category = UserWarning, module = "matplotlib")
#
# Display inline matplotlib plots with IPython
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
###########################################

import matplotlib.pyplot as pl
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import f1_score, accuracy_score


def distribution(data, transformed = False):
    """
    Visualization code for displaying skewed distributions of features
    """
    
    # Create figure
    fig = pl.figure(figsize = (12,4));

    # Skewed feature plotting
    for i, feature in enumerate(['capital-gain','capital-loss']):
        ax = fig.add_subplot(1, 2, i+1)
        ax.hist(data[feature], bins = 25, color = 'lightblue')
        ax.set_title("%s 特征分布"%(feature), fontsize = 14)
        ax.set_xlabel("值")
        ax.set_ylabel("记录数")
        ax.set_ylim((0, 2000))
        ax.set_yticks([0, 500, 1000, 1500, 2000])
        ax.set_yticklabels([0, 500, 1000, 1500, ">2000"])

    # Plot aesthetics
    if transformed:
        fig.suptitle("对数变换后的数据特征", \
            fontsize = 16, y = 1.03)
    else:
        fig.suptitle("偏态的数据特征", \
            fontsize = 16, y = 1.03)

    fig.tight_layout()
    fig.show()


def evaluate(results, accuracy, f1):
    """
    Visualization code to display results of various learners.
    
    inputs:
      - learners: a list of supervised learners
      - stats: a list of dictionaries of the statistic results from 'train_predict()'
      - accuracy: The score for the naive predictor
      - f1: The score for the naive predictor
    """
  
    # Create figure
    fig, ax = pl.subplots(2, 3, figsize = (12,8))

    # Constants
    bar_width = 0.3
    colors = ['lightblue','deepskyblue','navy']
    
    # Super loop to plot four panels of data
    for k, learner in enumerate(results.keys()):
        for j, metric in enumerate(['train_time', 'acc_train', 'f_train', 'pred_time', 'acc_val', 'f_val']):
            for i in np.arange(3):
                
                # Creative plot code
                ax[j//3, j%3].bar(i+k*bar_width, results[learner][i][metric], width = bar_width, color = colors[k])
                ax[j//3, j%3].set_xticks([0.45, 1.45, 2.45])
                ax[j//3, j%3].set_xticklabels(["10%", "30%", "100%"])
                ax[j//3, j%3].set_xlabel("训练集")
                ax[j//3, j%3].set_xlim((-0.1, 3.0))
    
    # Add unique y-labels
    ax[0, 0].set_ylabel("时长 (秒)")
    ax[0, 1].set_ylabel("准确率")
    ax[0, 2].set_ylabel("F-得分")
    ax[1, 0].set_ylabel("时长 (秒)")
    ax[1, 1].set_ylabel("准确率")
    ax[1, 2].set_ylabel("F-得分")
    
    # Add titles
    ax[0, 0].set_title("模型训练")
    ax[0, 1].set_title("训练集-正确率")
    ax[0, 2].set_title("训练集-F得分")
    ax[1, 0].set_title("模型预测")
    ax[1, 1].set_title("测试集-正确率")
    ax[1, 2].set_title("测试集-F得分")
    
    # Add horizontal lines for naive predictors
    ax[0, 1].axhline(y = accuracy, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    ax[1, 1].axhline(y = accuracy, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    ax[0, 2].axhline(y = f1, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    ax[1, 2].axhline(y = f1, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    
    # Set y-limits for score panels
    ax[0, 1].set_ylim((0, 1))
    ax[0, 2].set_ylim((0, 1))
    ax[1, 1].set_ylim((0, 1))
    ax[1, 2].set_ylim((0, 1))

    # Create patches for the legend
    patches = []
    for i, learner in enumerate(results.keys()):
        patches.append(mpatches.Patch(color = colors[i], label = learner))
    pl.legend(handles = patches, bbox_to_anchor = (-.80, 2.53), \
              loc = 'upper center', borderaxespad = 0., ncol = 3, fontsize = 'large')
    
    # Aesthetics
    pl.suptitle("模型指标表现", fontsize = 16, y = 1.10)
    pl.subplots_adjust(top=0.85, bottom=0., left=0.10, right=0.95, hspace=0.3,wspace=0.35)
    pl.show()
    

def feature_plot(importances, X_train, y_train):
    
    # Display the five most important features
    indices = np.argsort(importances)[::-1]
    columns = X_train.columns.values[indices[:5]]
    values = importances[indices][:5]

    # Creat the plot
    fig = pl.figure(figsize = (9,5))
    pl.title("归一化后的权重前五的特征", fontsize = 16)
    rects = pl.bar(np.arange(5), values, width = 0.6, align="center", color = 'lightblue', \
                label = "Feature Weight")
    
    # make bar chart higher to fit the text label
    axes = pl.gca()
    axes.set_ylim([0, np.max(values) * 1.1])

    # add text label on each bar
    delta = np.max(values) * 0.02
    
    for rect in rects:
        height = rect.get_height()
        pl.text(rect.get_x() + rect.get_width()/2., 
                height + delta, 
                '%.2f' % height,
                ha='center', 
                va='bottom')
    
    # Detect if xlabels are too long
    rotation = 0 
    for i in columns:
        if len(i) > 20: 
            rotation = 10 # If one is longer than 20 than rotate 10 degrees 
            break
    pl.xticks(np.arange(5), columns, rotation = rotation)
    pl.xlim((-0.5, 4.5))
    pl.ylabel("权重", fontsize = 12)
    pl.xlabel("特征", fontsize = 12)
    
    pl.legend(loc = 'upper center')
    pl.tight_layout()
    pl.show() 
