from collections import defaultdict
import numpy as np
import pandas as pd
import seaborn as sns
from statistics import mean, median, mode, StatisticsError
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
            best_chances_str = ", ".join(person_best_chances)
            print(
                f"{person}'s best matches are {best_chances_str}, with {round(person_max*100,1)}% each."
            )
        else:
            print(
                f"{person}'s best match is {person_best_chances[0]}, with {round(person_max*100,1)}%."
            )


def beam_analysis(remaining_scenarios):
    beam_results = defaultdict(dict)
    for scenario in remaining_scenarios:
        beams = []
        for scenario_b in remaining_scenarios:
            beams.append(len(scenario & scenario_b))
        beam_results[scenario]["beams"] = beams
        beam_results[scenario]["min_beams"] = min(beams)
        beam_results[scenario]["mean_beams"] = mean(beams)
        beam_results[scenario]["median_beams"] = median(beams)
        try:
            beam_results[scenario]["mode_beams"] = mode(beams)
        except StatisticsError:
            beam_results[scenario]["mode_beams"] = None
        beam_results[scenario]["beam_scenarios"] = dict()
        for beam_num in set(beams):
            beam_results[scenario]["beam_scenarios"][beam_num] = beams.count(beam_num)
        beam_results[scenario]["max_scenarios"] = max(
            beam_results[scenario]["beam_scenarios"].values()
        )
        beam_results[scenario]["median_scenarios"] = median(
            beam_results[scenario]["beam_scenarios"].values()
        )
        beam_results[scenario]["mean_scenarios"] = mean(
            beam_results[scenario]["beam_scenarios"].values()
        )
        try:
            beam_results[scenario]["mode_scenarios"] = mode(
                beam_results[scenario]["beam_scenarios"].values()
            )
        except StatisticsError:
            beam_results[scenario]["mode_scenarios"] = None

    beam_results_metadata = _get_beam_metadata(beam_results=beam_results)
    print("--------")
    scenario_metadata = _get_scenario_metadata(beam_results=beam_results)
    return (beam_results, beam_results_metadata, scenario_metadata)


def _get_metadata_maximal(beam_results, metadata_type):
    argmax_min = []
    max_min = 0
    argmax_median = []
    max_median = 0
    argmax_mean = []
    max_mean = 0
    argmax_mode = []
    max_mode = 0
    for scenario, data in beam_results.items():
        if data[f"min_{metadata_type}"] > max_min:
            max_min = data[f"min_{metadata_type}"]
            argmax_min = [scenario]
        elif data[f"min_{metadata_type}"] == max_min:
            argmax_min.append(scenario)
        if data[f"mean_{metadata_type}"] > max_mean:
            max_mean = data[f"mean_{metadata_type}"]
            argmax_mean = [scenario]
        elif data[f"mean_{metadata_type}"] == max_mean:
            argmax_mean.append(scenario)
        if data[f"median_{metadata_type}"] > max_median:
            max_median = data[f"median_{metadata_type}"]
            argmax_median = [scenario]
        elif data[f"median_{metadata_type}"] == max_median:
            argmax_median.append(scenario)
        try:
            if data[f"mode_{metadata_type}"] > max_mode:
                max_mode = data[f"mode_{metadata_type}"]
                argmax_mode = [scenario]
            elif data[f"mode_{metadata_type}"] == max_mode:
                argmax_mode.append(scenario)
        except TypeError:
            pass
    metadata = dict()
    for measure_tuple in [
        ("min", max_min, argmax_min),
        ("median", max_median, argmax_median),
        ("mean", max_mean, argmax_mean),
        ("mode", max_mode, argmax_mode),
    ]:
        print(f"max_{measure_tuple[0]} {metadata_type}={measure_tuple[1]}")
        print(f"{len(measure_tuple[2])} such guesses\n")
        metadata[f"max_{measure_tuple[0]}_{metadata_type}"] = {
            "value": measure_tuple[1],
            "arg": measure_tuple[2],
        }

    return metadata


def _get_metadata_minimal(beam_results, metadata_type):
    argmin_max = []
    min_max = float("inf")
    argmin_median = []
    min_median = float("inf")
    argmin_mean = []
    min_mean = float("inf")
    argmin_mode = []
    min_mode = float("inf")
    for scenario, data in beam_results.items():
        if data[f"max_{metadata_type}"] < min_max:
            min_max = data[f"max_{metadata_type}"]
            argmin_max = [scenario]
        elif data[f"max_{metadata_type}"] == min_max:
            argmin_max.append(scenario)
        if data[f"mean_{metadata_type}"] < min_mean:
            min_mean = data[f"mean_{metadata_type}"]
            argmin_mean = [scenario]
        elif data[f"mean_{metadata_type}"] == min_mean:
            argmin_mean.append(scenario)
        if data[f"median_{metadata_type}"] < min_median:
            min_median = data[f"median_{metadata_type}"]
            argmin_median = [scenario]
        elif data[f"median_{metadata_type}"] == min_median:
            argmin_median.append(scenario)
        try:
            if data[f"mode_{metadata_type}"] < min_mode:
                min_mode = data[f"mode_{metadata_type}"]
                argmin_mode = [scenario]
            elif data[f"mode_{metadata_type}"] == min_mode:
                argmin_mode.append(scenario)
        except TypeError:
            pass
    metadata = dict()
    for measure_tuple in [
        ("max", min_max, argmin_max),
        ("median", min_median, argmin_median),
        ("mean", min_mean, argmin_mean),
        ("mode", min_mode, argmin_mode),
    ]:
        print(f"min_{measure_tuple[0]} {metadata_type}={measure_tuple[1]}")
        print(f"{len(measure_tuple[2])} such guesses\n")
        metadata[f"min_{measure_tuple[0]}_{metadata_type}"] = {
            "value": measure_tuple[1],
            "arg": measure_tuple[2],
        }

    return metadata


def _get_beam_metadata(beam_results):
    return _get_metadata_maximal(beam_results=beam_results, metadata_type="beams")


def _get_scenario_metadata(beam_results):
    return _get_metadata_minimal(beam_results=beam_results, metadata_type="scenarios")
