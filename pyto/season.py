from enum import Enum
from collections import Counter, defaultdict
from itertools import combinations
import pickle
from statistics import mean, median, mode, StatisticsError
import time

import pandas as pd


class Season:
    def __init__(self, contestant_list, skip_initialization=False, verbose=False):
        self.contestant_list = contestant_list
        self.Contestant = Enum("Person", " ".join(self.contestant_list))
        self._couples = [
            frozenset(self._parse_couple(couple))
            for couple in combinations(self.contestant_list, 2)
        ]
        self.verbose = verbose
        if not skip_initialization:
            self.remaining_scenarios = self._generate_matchups()
            self._couple_scenarios = {
                couple: int(
                    len(self.remaining_scenarios) / (len(self.contestant_list) - 1)
                )
                for couple in self._couples
            }

    @property
    def couples(self):
        return [frozenset(self._unparse_couple(couple)) for couple in self._couples]

    def pickle(self, filename):
        with open(filename, "wb") as pfile:
            pickle.dump(self._pickle_repr, pfile, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def unpickle(cls, filename, verbose=False):
        with open(filename, "rb") as pfile:
            pickle_repr = pickle.load(pfile)
        season = cls(
            contestant_list=pickle_repr["contestant_list"],
            skip_initialization=True,
            verbose=verbose,
        )
        season.remaining_scenarios = pickle_repr["remaining_scenarios"]
        season._couple_scenarios = pickle_repr["couple_scenarios"]
        return season

    @property
    def _pickle_repr(self):
        pickle_repr = {
            "contestant_list": self.contestant_list,
            "remaining_scenarios": self.remaining_scenarios,
            "couple_scenarios": self._couple_scenarios,
        }
        return pickle_repr

    @property
    def couple_probabilities(self):
        return {
            couple: self._couple_scenarios.get(self._parse_couple(couple), 0)
            / self.num_remaining_scenarios
            for couple in self.couples
        }

    @property
    def couple_probability_df(self):
        df = pd.DataFrame(
            [
                [
                    self.couple_probabilities.get(frozenset([a, b]))
                    for b in self.contestant_list
                ]
                for a in self.contestant_list
            ]
        )
        df.index = self.contestant_list
        df.columns = self.contestant_list
        return df

    @property
    def couple_scenarios(self):
        return {
            couple: self._couple_scenarios.get(self._parse_couple(couple), 0)
            for couple in self.couples
        }

    @property
    def num_remaining_scenarios(self):
        return len(self.remaining_scenarios)

    def best_matches(self, name):
        couples = [couple for couple in self.couples if name in couple]
        max_probability = max([self.couple_probabilities[couple] for couple in couples])
        max_couples = [
            couple
            for couple in couples
            if self.couple_probabilities[couple] == max_probability
        ]
        return (
            [person for couple in max_couples for person in couple if person != name],
            max_probability,
        )

    def _generate_matchups(self):
        tic = time.time()
        matchups = set(
            frozenset(i)
            for i in self._meta_generate_matchups(
                [person.value for person in self.Contestant]
            )
        )
        toc = time.time()
        if self.verbose:
            print(f"{len(matchups)} matchups generated in {round(toc-tic)} seconds")
        return matchups

    def _meta_generate_matchups(self, people):
        # https://stackoverflow.com/questions/24130745/convert-generator-object-to-list-for-debugging
        if not people:
            return []
        elif len(people) == 2:
            return frozenset([people[0], people[1]])
        else:
            p0 = people[0]
            first_pairs = [frozenset([p0, p1]) for p1 in people[1:]]
            matchups = []
            for pair in first_pairs:
                everyone_else = list(set(people) - set(pair))
                if len(everyone_else) == 2:
                    matchups.append(
                        [pair, frozenset([everyone_else[0], everyone_else[1]])]
                    )
                else:
                    for pairs in self._meta_generate_matchups(everyone_else):
                        matchup = [pair] + pairs
                        matchups.append(matchup)
            return matchups

    def apply_truth_booth(self, couple, outcome):
        scenarios = len(self.remaining_scenarios)
        if self.verbose:
            print(f"beginning with {scenarios} scenarios...")
        t0 = time.time()
        self.remaining_scenarios = self._apply_truth_booth(couple, outcome)
        t1 = time.time()
        if self.verbose:
            print(f"truth booth evaluated in {round(t1-t0)} seconds...")
            print(
                f"{len(self.remaining_scenarios)} scenarios remaining after truth booth..."
            )
        self._couple_scenarios = self._recalculate_couple_scenarios()
        t2 = time.time()
        if self.verbose:
            print(f"couple scenarios retabulated in {round(t2-t1)} seconds")

    def apply_matchup_ceremony(self, matchup, beams):
        scenarios = len(self.remaining_scenarios)
        if self.verbose:
            print(f"beginning with {scenarios} scenarios...")
        t0 = time.time()
        self.remaining_scenarios = self._apply_matchup_ceremony(matchup, beams)
        t1 = time.time()
        if self.verbose:
            print(f"matchup ceremony evaluated in {round(t1-t0)} seconds...")
            print(
                f"{len(self.remaining_scenarios)} scenarios remaining after matchup ceremony..."
            )
        t2 = time.time()
        self._couple_scenarios = self._recalculate_couple_scenarios()
        if self.verbose:
            print(f"couple scenarios retabulated in {round(t2-t1)} seconds")

    def analyze_scenarios(self):
        self.remaining_scenario_analysis = defaultdict(dict)
        for scenario in self.remaining_scenarios:
            beams = []
            for scenario_b in self.remaining_scenarios:
                beams.append(len(scenario & scenario_b))
            self.remaining_scenario_analysis[scenario]["beams"] = beams
            self.remaining_scenario_analysis[scenario]["min_beams"] = min(beams)
            self.remaining_scenario_analysis[scenario]["mean_beams"] = mean(beams)
            self.remaining_scenario_analysis[scenario]["median_beams"] = median(beams)
            try:
                self.remaining_scenario_analysis[scenario]["mode_beams"] = mode(beams)
            except StatisticsError:
                self.remaining_scenario_analysis[scenario]["mode_beams"] = None
        return self.remaining_scenario_analysis
    
    def _apply_truth_booth(self, couple, outcome):
        return self._apply_matchup_ceremony([couple], int(outcome))

    def _apply_matchup_ceremony(self, matchup, beams):
        matchup = frozenset([self._parse_couple(couple) for couple in matchup])
        return {
            scenario
            for scenario in self.remaining_scenarios
            if len(matchup & scenario) == beams
        }

    def _parse_couple(self, couple):
        return frozenset(self.Contestant[name].value for name in couple)

    def _unparse_couple(self, couple):
        return frozenset(self.Contestant(value).name for value in couple)

    def _recalculate_couple_scenarios(self):
        counter = defaultdict(int)
        for matchup in self.remaining_scenarios:
            for couple in matchup:
                counter[couple] += 1
        for couple in self._couples:
            if couple not in counter:
                counter[couple] = 0
        return counter
