import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm

def heat_map(df, num_scenarios):
    fig, ax = plt.subplots(figsize=[18, 12])
    plot = sns.heatmap(
        df,
        vmin=0,
        vmax=1,
        cmap=cm.viridis,
        annot=True,
        square=True,
        ax=ax,
        fmt=".1%",
        annot_kws={"fontsize": 11},
    )
    ax.set_title(
        f"{num_scenarios} Scenarios Remaining",
        fontdict={"fontsize": 16, "fontweight": "bold"},
    )
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.tick_params(axis="x", which="major", rotation=90)
    plot.collections[0].colorbar.set_ticks([0, 1])
    plot.collections[0].colorbar.set_ticklabels(["0%", "100%"])
    
def best_matches(df):
    for person in df.index.values:
        person_max = max(df[person].fillna(0))
        person_best_chances = df[person][df[person] == person_max].index.values
        if person_max == 1:
            print(f"{person}'s Perfect Match is {person_best_chances[0]}.")
        elif len(person_best_chances) > 1:
            best_chances_str = ', '.join(person_best_chances)
            print(f"{person}'s best matches are {best_chances_str}, with {round(person_max*100,1)}% each.")
        else:
            print(f"{person}'s best match is {person_best_chances[0]}, with {round(person_max*100,1)}%.")