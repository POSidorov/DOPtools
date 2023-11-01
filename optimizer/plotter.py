import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error as mae

import argparse

parser = argparse.ArgumentParser(prog='Model CV plotter', 
                                description='Plot out the CV results of the optimizer')
parser.add_argument('-d', '--datadir', required=True)
parser.add_argument('-o', '--outfile', type=str, default='')
parser.add_argument('--errorbar', action='store_true')
parser.add_argument('--stats', action='store_true')
parser.add_argument('--title', type=str)


def r2(a, b):
    return 1. - np.sum((a-b)**2)/np.sum((a-np.mean(a))**2)

def rmse(a, b):
    return np.sqrt(np.sum((a-b)**2)/len(a))

def make_plot(predictions, **params):
    fig, ax = plt.subplots(figsize=(4,4), dpi=300, facecolor="white")

    cv_res = pd.read_table(predictions, sep=' ')
    prop_name = cv_res.columns[1].split('.')[0]

    a = cv_res[prop_name+".observed"]
    b = cv_res[[c for c in cv_res.columns if c.startswith(prop_name+'.predicted')]]
    if errorbar:
        ax.errorbar(a,b.mean(axis=1),b.std(axis=1), fmt="r.")
    else:
        ax.plot(a,b,'ro')
    ax.plot([a.min(), a.max()], [a.min(), a.max()], "k--")
    ax.set_xlabel("Observed "+prop_name)
    ax.set_ylabel("Predicted "+prop_name)
    ax.set_title(title)

    if stats:
        textstr = "\n".join([
            "MAE(CV) = %.3f"  % (mae(a,b.mean(axis=1)), ),
            "RMSE(CV) = %.3f"  % (rmse(a,b.mean(axis=1)), ),
            "R2(CV) = %.3f"  % (r2(a,b.mean(axis=1)), )
            ])
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
               verticalalignment="top", horizontalalignment="left",
               bbox={"boxstyle":"round", "facecolor":"white", "alpha":0})

    return fig, ax

if __name__ == '__main__':
    args = parser.parse_args()
    datadir = args.datadir
    outfile = args.outfile
    stats = args.stats
    errorbar = args.errorbar
    title = args.title

    fig, ax = make_plot(datadir+'/predictions')
    plt.tight_layout(pad=2)
    if args.outfile:
        if not outfile.endswith('.png'):
            outfile += '.png'
        plt.savefig(outfile)
    else:
        plt.show()
