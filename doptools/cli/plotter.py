# -*- coding: utf-8 -*-
#
#  Copyright 2022-2025 Pavel Sidorov <pavel.o.sidorov@gmail.com> This
#  file is part of DOPTools repository.
#
#  DOPtools is free software; you can redistribute it and/or modify
#  it under the terms of the GNU Lesser General Public License as published by
#  the Free Software Foundation; either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#  GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License
#  along with this program; if not, see <https://www.gnu.org/licenses/>.


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import auc, roc_curve
import argparse
from doptools.optimizer.utils import rmse, r2

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)


def make_regression_plot(predictions, errorbar=False, stats=False, title=""):

    fig, ax = plt.subplots(figsize=(4, 4), dpi=300, facecolor="white")

    cv_res = pd.read_table(predictions, sep=' ')
    prop_name = cv_res.columns[1].split('.')[0]

    a = cv_res[prop_name+".observed"]
    b = cv_res[[c for c in cv_res.columns if c.startswith(prop_name+'.predicted')]]
    if errorbar:
        ax.errorbar(a, b.mean(axis=1), b.std(axis=1), fmt="r.")
    else:
        ax.plot(a, b, 'ro')
    ax.plot([a.min(), a.max()], [a.min(), a.max()], "k--")
    ax.set_xlabel("Observed "+prop_name)
    ax.set_ylabel("Predicted "+prop_name)
    ax.set_title(title)

    if stats:
        textstr = "\n".join([
            "MAE(CV) = %.3f" % (mae(a, b.mean(axis=1)), ),
            "RMSE(CV) = %.3f" % (rmse(a, b.mean(axis=1)), ),
            "R2(CV) = %.3f" % (r2(a, b.mean(axis=1)), )
            ])
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
                verticalalignment="top", horizontalalignment="left",
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0})

    return fig, ax


def prepare_classification_plot(cv_res, pos_class = 1):
    prop_name = cv_res.columns[1].split('.')[0]
    true_val = cv_res[prop_name+".observed"].values
    pos_label = [c for c in cv_res.columns if c.startswith(prop_name+'.predicted_prob.class_'+str(pos_class))]
    if not pos_label:
        raise ValueError("No property label corresponding to the given --pos_class argument")
    pos_probas = cv_res[pos_label]

    roc_repeats = {}
    for col in pos_probas.columns:
        repeat = col.split("repeat")[-1]
        fpr, tpr, _ = roc_curve(true_val, pos_probas[col].values, pos_label=pos_class)
        roc_repeats[repeat] = {'fpr': fpr, 'tpr': tpr, 'auc': auc(fpr, tpr)}

    mean_fpr = np.linspace(0, 1, 100)
    interp_tpr = [np.interp(mean_fpr, roc_repeats[x]['fpr'], roc_repeats[x]['tpr']) for x in roc_repeats]
    mean_tpr = np.mean(interp_tpr, axis=0)
    mean_tpr[0] = 0
    mean_tpr[-1] = 1.0
    std_tpr = np.std(interp_tpr, axis=0)
    roc_mean = {
        'fpr': mean_fpr,
        'tpr': mean_tpr,
        'std_tpr': std_tpr,
        'tpr_upper': np.minimum(mean_tpr + std_tpr, 1),
        'tpr_lower': np.maximum(mean_tpr - std_tpr, 0),
        'auc': auc(mean_fpr, mean_tpr),
        'std_auc': np.std([roc_repeats[x]['auc'] for x in roc_repeats])
    }

    return roc_repeats, roc_mean


def make_classification_plot(predictions, class_number, **params):
    cv_res = pd.read_table(predictions, sep=' ')
    roc_repeats, roc_mean = prepare_classification_plot(cv_res, class_number)
    fig, ax = plt.subplots(figsize=(5, 5), dpi=300, facecolor="w")

    for rpt in roc_repeats:
        ax.plot(roc_repeats[rpt]['fpr'], roc_repeats[rpt]['tpr'],
                label=r"ROC repeat %.0f (AUC = %0.2f)" % (int(rpt), roc_repeats[rpt]['auc']),
                alpha=0.3, lw=1, )

    ax.plot(roc_mean['fpr'], roc_mean['tpr'],
            label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (roc_mean['auc'], roc_mean['std_auc']),
            color="k", lw=2, alpha=0.8, )
    ax.fill_between(roc_mean['fpr'], roc_mean['tpr_lower'], roc_mean['tpr_upper'],
                    color="grey", alpha=0.2, label=r"$\pm$ 1 std. dev.", )

    ax.set(xlabel="False Positive Rate",
           ylabel="True Positive Rate",
           title=f"Mean ROC curve with variability\n(Positive label '{class_number}')", )
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])

    return fig, ax


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Model CV plotter', 
                                     description='Plot out the CV results of the optimizer')
    parser.add_argument('-d', '--datadir', required=True)
    parser.add_argument('-o', '--outfile', type=str, default='')
    parser.add_argument('-t', '--task', type=str, default='R', choices=['R', 'C'])
    parser.add_argument('--pos_class', type=str, help="The label for the positive class")
    parser.add_argument('--errorbar', action='store_true')
    parser.add_argument('--stats', action='store_true')
    parser.add_argument('--title', type=str)
    
    args = parser.parse_args()
    datadir = args.datadir
    outfile = args.outfile
    stats = args.stats
    errorbar = args.errorbar
    title = args.title
    pos_class = args.pos_class

    if args.task == "R":
        fig, ax = make_regression_plot(datadir+'/predictions', errorbar=errorbar, stats=stats, title=title)
    elif args.task == "C":
        if not args.pos_class:
            raise ValueError("Positive class label should be given using --pos_class argument")
        fig, ax = make_classification_plot(datadir+'/predictions', pos_class)
    plt.tight_layout(pad=2)
    if args.outfile:
        if not outfile.endswith('.png'):
            outfile += '.png'
        plt.savefig(outfile)
    else:
        plt.show()


__all__ = ['make_regression_plot', 'make_classification_plot', 'prepare_classification_plot']
