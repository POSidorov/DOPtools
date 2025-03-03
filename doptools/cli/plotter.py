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
from sklearn.metrics import RocCurveDisplay, auc
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


def make_classification_plot(predictions, class_number, **params):
    fig, ax = plt.subplots(figsize=(5, 5), dpi=300, facecolor="w")

    cv_res = pd.read_table(predictions, sep=' ')
    prop_name = cv_res.columns[1].split('.')[0]

    a = cv_res[prop_name+".observed"]
    b = cv_res[[c for c in cv_res.columns if c.startswith(prop_name+'.predicted_prob.class_'+str(class_number))]]
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []
    for column in b.columns:
        repeat = column.split("repeat")[-1]
        viz = RocCurveDisplay.from_predictions(
            a,
            cv_res[column],
            name=f"ROC repeat {repeat}",
            alpha=0.3,
            lw=1,
            ax=ax,
        )
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="k",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=f"Mean ROC curve with variability\n(Positive label '{class_number}')",
    )
    ax.legend(loc="lower right", fontsize=9)
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
        fig, ax = make_classification_plot(datadir+'/predictions', pos_class)
    plt.tight_layout(pad=2)
    if args.outfile:
        if not outfile.endswith('.png'):
            outfile += '.png'
        plt.savefig(outfile)
    else:
        plt.show()


__all__ = ['make_regression_plot', 'make_classification_plot']
