import itertools
import numpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import statsmodels.formula.api as sm
import sys
from tabulate import tabulate

from experiments import correlate_backdoors_and_bridges
from jsat_experiments import add_higher_order_features
from old_expts.data_analysis import type_map, all_subsets
from phdtk.db import create_sat_db_csv, init_cp2017_setup
import phdtk.latex_gen as latex_gen

comps = ["sc14-app", "sc13-app", "sc11-app", "sc09-app"]
solvers = ["maplecomsps", "glucose", "minisat"]

FSTR = "{0:.2f}"
BIG_FSTR = "{0:.0f}"


def data_summary(df, benchmarks, out_file, caption_prefix=None):
    """
    Summarizes the average values of base features across each benchmark
    """
    if not caption_prefix:
        caption_prefix = ""
    df = df[df.benchmark.isin(benchmarks)]
    df['SAT'] = df['result']
    df.loc[df.SAT != "SAT", ['SAT']] = np.nan
    g = df.groupby("benchmark")
    res = g.aggregate('count')
    index = res.index
    res = res.append(res.sum(numeric_only=True), ignore_index=True)
    res.index = list(index) + ["Total"]
    # use simp_num_vars for # time instances, since maplecomsps contains some instances that get simplified away
    out = res[["simp_num_vars", "simp_lsr_size", "simp_weak_size", "simp_q", "simp_backbones", "simp_tw_upper"]]
    out.columns = ["Instances", "LSR", "Weak", "Cmty", "Bones", "TW"]

    with open(out_file, 'w') as o:
        latex_gen.insert_table(o, out.to_latex(), tabular=True, precomputed=True, tiny=False,
                               caption=caption_prefix + " The number of instances for which" +
                               " we were able to successfully compute each parameter. " +
                               "``Cmty'' refers to the community parameters; " +
                               "``TW'' denotes the treewidth upper bound; " +
                               "``Bones'' denotes backbone size. ")
    print(out.to_latex())


def average_metric_values(df, benchmarks, out_file, caption_prefix=None):
    """
    Do metrics look better for app as opposed to random/crafted?
    """
    if not caption_prefix:
        caption_prefix = ""
    df = df[df.benchmark.isin(benchmarks)]
    df = df[['benchmark', 'simp_lvr', 'simp_wvr', 'simp_q', 'simp_backbonesvr', 'simp_tw_uppervr']]
    g = df.groupby("benchmark")
    res = g.aggregate('mean')
    res2 = g.aggregate('std')
    res3 = res.combine(res2, lambda x, y: [FSTR.format(i) + " (" + FSTR.format(j) + ")" for i, j in zip(x, y)])
    res3.columns = ['LSR/V', 'Weak/V', 'Q', 'Bones/V', 'TW/V']

    with open(out_file, 'w') as o:
        latex_gen.insert_table(o, res3.to_latex(), tabular=True, precomputed=True, tiny=False,
                               caption=caption_prefix + " Mean (std. dev.) of several parameter values. ",
                               label="tab-meanstd")


def lsr_all_decs_comparison(df, benchmarks, out_file, caption_prefix=None):
    if not caption_prefix:
        caption_prefix = ""
    df = df[df.benchmark.isin(benchmarks)]
    df['simp_lsr_all_decs_overlap_ratio'] = df['simp_lsr_all_decs_intervr'] / df['simp_lsr_all_decs_unionvr']
    df = df[['benchmark', 'simp_lvr', 'simp_all_decsvr', 'simp_lsr_all_decs_overlap_ratio']]

    g = df.groupby("benchmark")
    res = g.aggregate('mean')
    res2 = g.aggregate('std')

    '''
    for col in res:
        res[col] = [np.nan if (not isinstance(val, str) and np.isnan(val)) else
                   (val if isinstance(val, str) else str(float(val)))
                   for val in res[col].tolist()]

    for col in res2:
        res2[col] = [np.nan if (not isinstance(val, str) and np.isnan(val)) else
                   (val if isinstance(val, str) else str(float(val)))
                   for val in res2[col].tolist()]
    '''
    # print(res)
    # print(res2)
    # print(res.combine(res2, lambda x, y: str(x) + " (" + str(y) + ")"))
    res3 = res.combine(res2, lambda x, y: [FSTR.format(i) + " (" + FSTR.format(j) + ")" for i, j in zip(x, y)])
    res3.columns = ["Laser", "All Decisions", "Overlap Ratio"]

    print(res3)
    print(res3.to_latex())

    with open(out_file, 'w') as o:
        latex_gen.insert_table(o, res3.to_latex(), precomputed=True, tiny=False, tabular=True,
                               caption=caption_prefix + " Mean (std. dev.) of Laser produced backdoor sizes " +
                               "versus all decision variables. " +
                               "Overlap Ratio is the size of the set " +
                               "$(Laser \\cap All Decisions) / (Laser \\cup All Decisions)$.",
                               label="lsr_vs_all_decs_table")


def structure_logging_summary(df, benchmarks, out_file, full=False):
    """
    do metrics look better for app as opposed to random/crafted?
    """
    print("Structure logging")
    out_str = ""

    df = df[df.benchmark.isin(benchmarks)]

    # 'struct_gini_normalized_picks', 'struct_ar_gini_normalized_picks', 'struct_nr_gini_normalized_picks',
    # 'struct_gini_normalized_clauses', 'struct_ar_gini_normalized_clauses', 'struct_nr_gini_normalized_clauses',
    df = df[['benchmark', 'name', 'struct_lsr', 'struct_ar_lsr', 'struct_nr_lsr',
             'simp_maplesat_time', 'simp_maplesat_ar_time', 'simp_maplesat_nr_time',
             'simp_maplesat_conflicts', 'simp_maplesat_ar_conflicts', 'simp_maplesat_nr_conflicts',
             'struct_avg_clause_lsr', 'struct_ar_avg_clause_lsr', 'struct_nr_avg_clause_lsr']]
    df = df.dropna()
    if full:
        # ['benchmark',
        # 'struct_gini_normalized_picks',
        # 'struct_ar_gini_normalized_picks',
        # 'struct_nr_gini_normalized_picks'],
        # ['benchmark',
        # 'struct_gini_normalized_clauses',
        # 'struct_ar_gini_normalized_clauses',
        # 'struct_nr_gini_normalized_clauses'],
        feature_lists = [
            ['benchmark', 'struct_lsr', 'struct_ar_lsr', 'struct_nr_lsr'],
            ['benchmark', 'struct_avg_clause_lsr', 'struct_ar_avg_clause_lsr', 'struct_nr_avg_clause_lsr'],
            ['benchmark', 'simp_maplesat_conflicts', 'simp_maplesat_ar_conflicts', 'simp_maplesat_nr_conflicts'],
            ['benchmark', 'simp_maplesat_time', 'simp_maplesat_ar_time', 'simp_maplesat_nr_time']
        ]

        # 'P1: Community-based Spatial Locality of Decisions',
        # 'P2: Community-based Spatial Locality of Learnt Clauses',
        expt_name_list = [
            'LSR Size',
            'Avg. Clause LSR',
            'Num Conflicts',
            'Solving Time (s)']
        best = ["min", "min", "min", "min"]
    else:
        feature_lists = [
            ['benchmark', 'struct_lsr', 'struct_ar_lsr', 'struct_nr_lsr'],
            ['benchmark', 'struct_avg_clause_lsr', 'struct_ar_avg_clause_lsr', 'struct_nr_avg_clause_lsr'],
            ['benchmark', 'simp_maplesat_conflicts', 'simp_maplesat_ar_conflicts', 'simp_maplesat_nr_conflicts'],
            ['benchmark', 'simp_maplesat_time', 'simp_maplesat_ar_time', 'simp_maplesat_nr_time']
        ]

        expt_name_list = [
            'LSR Size',
            'Avg. Clause LSR',
            'Num Conflicts',
            'Solving Time (s)']
        best = ["min", "min", "min", "min"]

    end_row = " \\\\ \\hline"

    out_str += "\\begin{center}\n"
    out_str += "\\begin{tabular}{ |l|c|c|c| }\n"

    # header
    out_str += "\\hline\n"
    out_str += " & ".join(
        ["\\textbf{" + i + "}" for i in ["Property", "Luby", "Always Restart", "Never Restart"]]) + end_row + "\n"

    for l, e, b in zip(feature_lists, expt_name_list, best):
        df2 = df[l]
        g = df2.groupby("benchmark")
        res = g.aggregate('mean')
        res2 = g.aggregate('std')
        res3 = res.combine(res2, lambda x, y: [FSTR.format(i) + " (" + FSTR.format(j) + ")"
                                               if i <= 1000
                                               else BIG_FSTR.format(i) + " (" + BIG_FSTR.format(j) + ")"
                                               for i, j in zip(x, y)])
        out_str += e + "& "
        for index, row in res3.iterrows():
            pre = ""
            post = "\\\\"
            nums = [float(row[fname].split()[0]) for fname in l[1:]]
            nums_and_std = [row[fname] for fname in l[1:]]
            high = -1
            low = 9999999
            high_index = -1
            low_index = -1

            for i in range(len(nums)):
                if str(nums[i]) == "nan":
                    continue
                else:
                    if nums[i] > high:
                        high = nums[i]
                        high_index = i
                    if nums[i] < low:
                        low = nums[i]
                        low_index = i

            for i in range(len(nums)):
                if str(nums[i]) == "nan":
                    continue

                if b == "min" and low_index == i:
                    nums_and_std[low_index] = "\\textbf{" + nums_and_std[low_index] + "}"
                elif b == "max" and high_index == i:
                    nums_and_std[high_index] = "\\textbf{" + nums_and_std[high_index] + "}"

            out_str += pre + " & ".join(str(i) for i in nums_and_std) + post + "\n"
    out_str += "\\hline\n"
    out_str += "\\end{tabular}\n"
    out_str += "\\end{center}\n"
    with open(out_file, 'w') as o:
        latex_gen.insert_table(o, out_str, tabular=True, precomputed=True, tiny=False, label="tab_lens",
                               caption="Comparison of LSR measures and solving time for various restart policies" +
                                       " on the Agile benchmark. LSR sizes are normalized by the number of variables.")

def pocr_structure_logging_summary(df, benchmark_names, identifiers, headers, caption, label, out_file, sat=None):
    # TODO how to filter instances (maybe take all that vsids + lrb finish, and then note in the paper the difference
    # for random

    print("Structure logging")
    out_str = ""

    HOW = "any"

    # 'struct_gini_normalized_picks', 'struct_ar_gini_normalized_picks', 'struct_nr_gini_normalized_picks',
    # 'struct_gini_normalized_clauses', 'struct_ar_gini_normalized_clauses', 'struct_nr_gini_normalized_clauses',
    main_cols = ['benchmark', 'name', 'simp_num_vars', 'simp_backbones', 'simp_backbonesvr', 'result',
             'pocr_lrb_ar_all_decs',
             'pocr_lrb_ar_bb_flips',
             'pocr_lrb_ar_bb_subsumed',
             'pocr_lrb_ar_gini_clauses',
             'pocr_lrb_ar_gini_picks',
             'pocr_lrb_ar_lsr',
             'pocr_lrb_luby_all_decs',
             'pocr_lrb_luby_bb_flips',
             'pocr_lrb_luby_bb_subsumed',
             'pocr_lrb_luby_gini_clauses',
             'pocr_lrb_luby_gini_picks',
             'pocr_lrb_luby_lsr',
             'pocr_lrb_nr_all_decs',
             'pocr_lrb_nr_bb_flips',
             'pocr_lrb_nr_bb_subsumed',
             'pocr_lrb_nr_gini_clauses',
             'pocr_lrb_nr_gini_picks',
             'pocr_lrb_nr_lsr',
             'pocr_random_ar_all_decs',
             'pocr_random_ar_bb_flips',
             'pocr_random_ar_bb_subsumed',
             'pocr_random_ar_gini_clauses',
             'pocr_random_ar_gini_picks',
             'pocr_random_ar_lsr',
             'pocr_random_luby_all_decs',
             'pocr_random_luby_bb_flips',
             'pocr_random_luby_bb_subsumed',
             'pocr_random_luby_gini_clauses',
             'pocr_random_luby_gini_picks',
             'pocr_random_luby_lsr',
             'pocr_random_nr_all_decs',
             'pocr_random_nr_bb_flips',
             'pocr_random_nr_bb_subsumed',
             'pocr_random_nr_gini_clauses',
             'pocr_random_nr_gini_picks',
             'pocr_random_nr_lsr',
             'pocr_vsids_ar_all_decs',
             'pocr_vsids_ar_bb_flips',
             'pocr_vsids_ar_bb_subsumed',
             'pocr_vsids_ar_gini_clauses',
             'pocr_vsids_ar_gini_picks',
             'pocr_vsids_ar_lsr',
             'pocr_vsids_luby_all_decs',
             'pocr_vsids_luby_bb_flips',
             'pocr_vsids_luby_bb_subsumed',
             'pocr_vsids_luby_gini_clauses',
             'pocr_vsids_luby_gini_picks',
             'pocr_vsids_luby_lsr',
             'pocr_vsids_nr_all_decs',
             'pocr_vsids_nr_bb_flips',
             'pocr_vsids_nr_bb_subsumed',
             'pocr_vsids_nr_gini_clauses',
             'pocr_vsids_nr_gini_picks',
             'pocr_vsids_nr_lsr',
             'pocr_lrb_ar_time',
             'pocr_lrb_luby_time',
             'pocr_lrb_nr_time',
             'pocr_random_ar_time',
             'pocr_random_luby_time',
             'pocr_random_nr_time',
             'pocr_vsids_ar_time',
             'pocr_vsids_luby_time',
             'pocr_vsids_nr_time',
             'pocr_lrb_ar_bb_conflicts',
             'pocr_lrb_luby_bb_conflicts',
             'pocr_lrb_nr_bb_conflicts',
             'pocr_random_ar_bb_conflicts',
             'pocr_random_luby_bb_conflicts',
             'pocr_random_nr_bb_conflicts',
             'pocr_vsids_ar_bb_conflicts',
             'pocr_vsids_luby_bb_conflicts',
             'pocr_vsids_nr_bb_conflicts',
             'pocr_lrb_ar_bb_propagations',
             'pocr_lrb_luby_bb_propagations',
             'pocr_lrb_nr_bb_propagations',
             'pocr_random_ar_bb_propagations',
             'pocr_random_luby_bb_propagations',
             'pocr_random_nr_bb_propagations',
             'pocr_vsids_ar_bb_propagations',
             'pocr_vsids_luby_bb_propagations',
             'pocr_vsids_nr_bb_propagations',
             'pocr_lrb_ar_bb_subsumed_raw',
             'pocr_lrb_luby_bb_subsumed_raw',
             'pocr_lrb_nr_bb_subsumed_raw',
             'pocr_random_ar_bb_subsumed_raw',
             'pocr_random_luby_bb_subsumed_raw',
             'pocr_random_nr_bb_subsumed_raw',
             'pocr_vsids_ar_bb_subsumed_raw',
             'pocr_vsids_luby_bb_subsumed_raw',
             'pocr_vsids_nr_bb_subsumed_raw',
             'pocr_lrb_ar_decisions',
             'pocr_lrb_luby_decisions',
             'pocr_lrb_nr_decisions',
             'pocr_random_ar_decisions',
             'pocr_random_luby_decisions',
             'pocr_random_nr_decisions',
             'pocr_vsids_ar_decisions',
             'pocr_vsids_luby_decisions',
             'pocr_vsids_nr_decisions',
             'pocr_lrb_ar_conflicts',
             'pocr_lrb_luby_conflicts',
             'pocr_lrb_nr_conflicts',
             'pocr_random_ar_conflicts',
             'pocr_random_luby_conflicts',
             'pocr_random_nr_conflicts',
             'pocr_vsids_ar_conflicts',
             'pocr_vsids_luby_conflicts',
             'pocr_vsids_nr_conflicts',
             'pocr_lrb_ar_lsr_cmty_spread',
             'pocr_lrb_luby_lsr_cmty_spread',
             'pocr_lrb_nr_lsr_cmty_spread',
             'pocr_random_ar_lsr_cmty_spread',
             'pocr_random_luby_lsr_cmty_spread',
             'pocr_random_nr_lsr_cmty_spread',
             'pocr_vsids_ar_lsr_cmty_spread',
             'pocr_vsids_luby_lsr_cmty_spread',
             'pocr_vsids_nr_lsr_cmty_spread',
             'pocr_lrb_ar_lsr_cmty_largest_ratio',
             'pocr_lrb_luby_lsr_cmty_largest_ratio',
             'pocr_lrb_nr_lsr_cmty_largest_ratio',
             'pocr_random_ar_lsr_cmty_largest_ratio',
             'pocr_random_luby_lsr_cmty_largest_ratio',
             'pocr_random_nr_lsr_cmty_largest_ratio',
             'pocr_vsids_ar_lsr_cmty_largest_ratio',
             'pocr_vsids_luby_lsr_cmty_largest_ratio',
             'pocr_vsids_nr_lsr_cmty_largest_ratio'
             ]

    df = df[main_cols]

    bases = ['pocr_lrb_ar',
             'pocr_lrb_luby',
             'pocr_lrb_nr',
             'pocr_random_ar',
             'pocr_random_luby',
             'pocr_random_nr',
             'pocr_vsids_ar',
             'pocr_vsids_luby',
             'pocr_vsids_nr']


    for c in main_cols:
        if 'lsr' in c and 'lsr_cmty' not in c:
            df[c + "_vr"] = df[c] / df['simp_num_vars']

    for i in bases:
        df[i + "_glr"] = df[i + "_conflicts"] / df[i + "_decisions"]


    df_agile = df[df.benchmark.isin(["Agile"])]
    df_app = df[df.benchmark.isin(["Application"])]

    # todo  warning!!! remove next line
    #df_agile = df_agile.drop(df_agile[df_agile.result != "UNSAT"].index)
    df_agile_sat = df_agile.drop(df_agile[df_agile.result != "SAT"].index)
    df_agile_unsat = df_agile.drop(df_agile[df_agile.result != "UNSAT"].index)

    #glr_cols = [col for col in df_agile.columns if "glr" in col or "name" in col]
    ## print(df_agile[glr_cols])
    #for i in np.mean(df_agile[glr_cols]).items():
    #    print(i)


    #print("SAT")
    #for i in np.mean(df_agile_sat[glr_cols]).items():
    #    print(i)
    #print("UNSAT")
    #for i in np.mean(df_agile_unsat[glr_cols]).items():
    #    print(i)

    #sys.exit()

    # Generic Tables
    all_dfs = []
    for i in benchmark_names:
        d = df[df.benchmark.isin([i])]
        if sat:
            d = d.drop(d[d.result != sat].index)
        all_dfs.append(d)

    avgs = []
    for i in identifiers:
        l = [col for col in df.columns if i in col]
        for curr_df in all_dfs:
            d = curr_df[l]
            d = d.dropna(how=HOW)
            a = np.mean(d)
            avgs.append(a)

    # produce table
    end_row = " \\\\ \\hline"

    out_str = ""
    out_str += "\\begin{table}[t]\n"
    out_str += "\\begin{center}\n"
    out_str += "\\begin{tabular}{ |l|l||c|c||c|c| }\n"

    b_map = {"LRB": "lrb", "VSIDS": "vsids", "Random": "random"}
    r_map = {"Luby": "luby", "Always": "ar", "Never": "nr"}

    out_str += "\\hline\n"
    out_str += "\\multicolumn{2}{ |c|| }{\\textbf{Heuristic}} " + \
        "".join(["& \\multicolumn{" + str(len(identifiers)) + "}{ c"
                 + (" || " if j != len(benchmark_names) - 1 else " | ") + "}{\\textbf{" + i + "}} "
                 for i,j in zip(benchmark_names, range(len(benchmark_names)))]) + end_row + "\n"
    out_str += " & ".join(
        ["\\textbf{" + i + "}" for i in
         ["Branching", "Restart"] + (headers * len(benchmark_names))]) + end_row + "\n"
    for b in ["LRB", "VSIDS", "Random"]:
        for r in ["Luby", "Always", "Never"]:
            row = ""
            if r == "Luby":
                row += "\multirow{3}{4em}{" + b + "} &"
            else:
                row += "& "
            row += r + " & "
            key = "pocr_" + b_map[b] + "_" + r_map[r] + "_"
            for a in avgs:
                print(a)
            row += " & ".join(
                FSTR.format(i) for i in [a[key + ident] for a, ident in zip(avgs, identifiers * len(benchmark_names))])
            out_str += row
            if r == "Never":
                out_str += end_row + "\n"
            else:
                out_str += "\\\\\n"
    out_str += "\\end{tabular}\n"
    out_str += "\\end{center}\n"
    out_str += "\\caption{" + (sat + " ONLY. " if sat else "") + caption + "}\n"
    out_str += "\\label{" + label + "}\n"
    out_str += "\\end{table}\n\n"

    print(out_str)
    out_file.write(out_str)
    return

    # Cmty Tables

    # columns Picks_Comps, Clauses_Comps, double-line, Picks_Agile, Clauses_Agile
    # rows LRB+R, VSIDS+R, Random+R
    cmty_cols = [col for col in df.columns if 'gini' in col]
    df_agile_cmty = df_agile[cmty_cols]
    df_app_cmty = df_app[cmty_cols]
    df_agile_cmty = df_agile_cmty.dropna(how=HOW)
    df_app_cmty = df_app_cmty.dropna(how=HOW)

    picks_cols = [col for col in df_app_cmty.columns if "picks" in col]
    df_app_picks = df_app_cmty[picks_cols]
    pick_app_avg = np.mean(df_app_picks)
    for i in pick_app_avg.items():
        print(i)

    picks_cols = [col for col in df_agile_cmty.columns if "picks" in col]
    df_agile_picks = df_agile_cmty[picks_cols]
    pick_agile_avg = np.mean(df_agile_picks)
    for i in pick_agile_avg.items():
        print(i)

    print("### GINI CLAUSES", len(df_app_picks), len(df_agile_picks))
    clauses_cols = [col for col in df_app_cmty.columns if "gini" in col and "clause" in col]
    df_app_clauses = df_app_cmty[clauses_cols]
    clauses_app_avg = np.mean(df_app_clauses)
    for i in clauses_app_avg.items():
        print(i)

    clauses_cols = [col for col in df_agile_cmty.columns if "gini" in col and "clause" in col]
    df_agile_clauses = df_agile_cmty[clauses_cols]
    clauses_agile_avg = np.mean(df_agile_clauses)
    for i in clauses_agile_avg.items():
        print(i)


    # produce table
    end_row = " \\\\ \\hline"

    out_str = ""
    out_str += "\\begin{table}[t]\n"
    out_str += "\\begin{center}\n"
    out_str += "\\begin{tabular}{ |l|l||c|c||c|c| }\n"

    b_map = {"LRB": "lrb", "VSIDS": "vsids", "Random": "random"}
    r_map = {"Luby": "luby", "Always": "ar", "Never": "nr"}

    out_str += "\\hline\n"
    out_str += "\\multicolumn{2}{ |c|| }{\\textbf{Heuristic}} & \\multicolumn{2}{ c|| }{\\textbf{Application}} & \\multicolumn{2}{ c| }{\\textbf{Agile}}" + end_row + "\n"
    out_str += " & ".join(
        ["\\textbf{" + i + "}" for i in ["Branching", "Restart", "Gini Picks", "Gini Clauses", "Gini Picks", "Gini Clauses"]]) + end_row + "\n"
    for b in ["LRB", "VSIDS", "Random"]:
        for r in ["Luby", "Always", "Never"]:
            row = ""
            if r == "Luby":
                row += "\multirow{3}{4em}{" + b + "} &"
            else:
                row += "& "
            row += r + " & "
            key = "pocr_" + b_map[b] + "_" + r_map[r] + "_"
            row += " & ".join(
                FSTR.format(i) for i in [pick_app_avg[key + "gini_picks"], clauses_app_avg[key + "gini_clauses"],
                                         pick_agile_avg[key + "gini_picks"], clauses_agile_avg[key + "gini_clauses"]])
            out_str += row
            if r == "Never":
                out_str += end_row + "\n"
            else:
                out_str += "\\\\\n"
    out_str += "\\end{tabular}\n"
    out_str += "\\end{center}\n"
    out_str += "\\caption{Measures the spatial locality of the branching heuristics' decisions, with respect to the " + \
               "underlying community structure. Further measures a similar locality notion of the learnt clauses.}\n"
    out_str += "\\label{tab:lens_lsr_cmty}\n"
    out_str += "\\end{table}\n"

    print(out_str)

    # LSR Cmty Tables
    cmty_cols = [col for col in df.columns if 'lsr_cmty' in col]
    df_agile_cmty = df_agile[cmty_cols]
    df_app_cmty = df_app[cmty_cols]
    df_agile_cmty = df_agile_cmty.dropna(how=HOW)
    df_app_cmty = df_app_cmty.dropna(how=HOW)

    spread_cols = [col for col in df_app_cmty.columns if "spread" in col]
    df_app_spread = df_app_cmty[spread_cols]
    spread_app_avg = np.mean(df_app_spread)
    for i in spread_app_avg.items():
        print(i)

    spread_cols = [col for col in df_agile_cmty.columns if "spread" in col]
    df_agile_spread = df_agile_cmty[spread_cols]
    spread_agile_avg = np.mean(df_agile_spread)
    for i in spread_agile_avg.items():
        print(i)

    ratio_cols = [col for col in df_app_cmty.columns if "largest_ratio" in col]
    df_app_ratios = df_app_cmty[ratio_cols]
    ratio_app_avg = np.mean(df_app_ratios)
    for i in ratio_app_avg.items():
        print(i)

    ratio_cols = [col for col in df_agile_cmty.columns if "largest_ratio" in col]
    df_agile_ratios = df_agile_cmty[ratio_cols]
    ratio_agile_avg = np.mean(df_agile_ratios)
    for i in ratio_agile_avg.items():
        print(i)

    # produce table
    end_row = " \\\\ \\hline"

    out_str = ""
    out_str += "\\begin{table}[t]\n"
    out_str += "\\begin{center}\n"
    out_str += "\\begin{tabular}{ |l|l||c|c||c|c| }\n"

    b_map = {"LRB": "lrb", "VSIDS": "vsids", "Random": "random"}
    r_map = {"Luby": "luby", "Always": "ar", "Never": "nr"}

    out_str += "\\hline\n"
    out_str += "\\multicolumn{2}{ |c|| }{\\textbf{Heuristic}} & \\multicolumn{2}{ c|| }{\\textbf{Application}} & \\multicolumn{2}{ c| }{\\textbf{Agile}}" + end_row + "\n"
    out_str += " & ".join(
        ["\\textbf{" + i + "}" for i in
         ["Branching", "Restart", "LSR Spread", "LSR Big Ratio", "LSR Spread", "LSR Big Ratio"]]) + end_row + "\n"
    for b in ["LRB", "VSIDS", "Random"]:
        for r in ["Luby", "Always", "Never"]:
            row = ""
            if r == "Luby":
                row += "\multirow{3}{4em}{" + b + "} &"
            else:
                row += "& "
            row += r + " & "
            key = "pocr_" + b_map[b] + "_" + r_map[r] + "_"

            row += " & ".join(FSTR.format(i) if str(i) != "nan" else "nan" for i in
                              [spread_app_avg[key + "lsr_cmty_spread"],
                               ratio_app_avg[key + "lsr_cmty_largest_ratio"],
                               spread_agile_avg[key + "lsr_cmty_spread"],
                               ratio_agile_avg[key + "lsr_cmty_largest_ratio"]])
            out_str += row
            if r == "Never":
                out_str += end_row + "\n"
            else:
                out_str += "\\\\\n"
    out_str += "\\end{tabular}\n"
    out_str += "\\end{center}\n"
    out_str += "\\caption{Measures the relationship of LSR to cmtys.}\n"


    out_str += "\\label{tab:lens_cmty}\n"
    out_str += "\\end{table}\n"

    print(out_str)

    # BB Tables #####################################################################

    # for each benchmark, report:
    # num_subsumed / num_conflicts (bb_subsumed_raw / bb_conflicts),
    # num_subsumed / num_bb (bb_subsumed),
    # num_flips / num_bb (bb_flips)

    bb_cols = [col for col in df.columns if '_bb_' in col or 'backbones' in col or 'propagations' in col or 'name' in col]

    df_agile_bb = df_agile[bb_cols]
    df_app_bb = df_app[bb_cols]
    df_agile_bb = df_agile_bb.dropna(how=HOW)
    df_app_bb = df_app_bb.dropna(how=HOW)

    l = [i for i in bb_cols if 'conflict' in i]
    print (l)

    for c in [i for i in bb_cols if 'conflict' in i]:
        b = c[:-10]
        df_app_bb[b + "_subsumed_norm_conflicts"] = df_app_bb[b + "_subsumed_raw"] / df_app_bb[b + "_conflicts"]
        df_agile_bb[b + "_subsumed_norm_conflicts"] = df_agile_bb[b + "_subsumed_raw"] / df_agile_bb[b + "_conflicts"]


    df_app_avg = np.mean(df_app_bb)
    df_agile_avg = np.mean(df_agile_bb)
    print("BB RATIO", df_app_avg['simp_backbonesvr'], df_agile_avg['simp_backbonesvr'])

    '''
    print("FLIPS", len(df_app_bb), len(df_agile_bb))
    flips_cols = [col for col in df_app_bb.columns if "flips" in col]

    for i in flips_cols:
        df_agile_bb = df_agile_bb.drop(df_agile_bb[df_agile_bb[i] < 0].index)
        df_app_bb = df_app_bb.drop(df_app_bb[df_app_bb[i] < 0].index)

    flips_cols = [col for col in df_app_bb.columns if "flips" in col]

    df_app_flips = df_app_bb[flips_cols]




    flips_app_avg = np.mean(df_app_flips)



    for i in flips_app_avg.items():
        print(i)

    df_agile_flips = df_agile_bb[flips_cols]
    flips_agile_avg = np.mean(df_agile_flips)
    for i in flips_agile_avg.items():
        print(i)
    
    print("SUBSUMES")
    subsumes_cols = [col for col in df_app_bb.columns if "subsumed" in col]
    df_app_subsumes = df_app_bb[subsumes_cols]
    subsumes_app_avg = np.mean(df_app_subsumes)
    for i in subsumes_app_avg.items():
        print(i)

    subsumes_cols = [col for col in df_agile_bb.columns if "subsumed" in col]
    df_agile_subsumes = df_agile_bb[subsumes_cols]
    subsumes_agile_avg = np.mean(df_agile_subsumes)
    for i in subsumes_agile_avg.items():
        print(i)
    '''
    # produce table
    out_str = ""
    out_str += "\\begin{table}[t]\n"
    out_str += "\\begin{center}\n"
    out_str += "\\begin{tabular}{ |l|l||c|c|c||c|c|c| }\n"


    out_str += "\\hline\n"
    out_str += "\\multicolumn{2}{ |c|| }{\\textbf{Heuristic}} & \\multicolumn{3}{ c|| }{\\textbf{Application (" + \
        BIG_FSTR.format(df_app_avg['simp_backbonesvr']*100) +"\\% backbone)}} & \\multicolumn{3}{ c| }{\\textbf{Agile (" + \
        BIG_FSTR.format(df_agile_avg['simp_backbonesvr']*100) + "\\% backbone)}}" + end_row + "\n"
    out_str += " & ".join(
        ["\\textbf{" + i + "}" for i in
         ["Branching", "Restart", "Subs/L", "Subs/B", "Flips/B", "Subs/L", "Subs/B", "Flips/B"]]) + end_row + "\n"
    for b in ["LRB", "VSIDS", "Random"]:
        for r in ["Luby", "Always", "Never"]:
            row = ""
            if r == "Luby":
                row += "\multirow{3}{4em}{" + b + "} &"
            else:
                row += "& "
            row += r + " & "
            key = "pocr_" + b_map[b] + "_" + r_map[r] + "_"
            row += " & ".join([FSTR.format(df_app_avg[key + "bb_subsumed_norm_conflicts"]),
                               BIG_FSTR.format(df_app_avg[key + "bb_subsumed"]),
                               BIG_FSTR.format(df_app_avg[key + "bb_flips"]),
                               FSTR.format(df_agile_avg[key + "bb_subsumed_norm_conflicts"]),
                               BIG_FSTR.format(df_agile_avg[key + "bb_subsumed"]),
                               BIG_FSTR.format(df_agile_avg[key + "bb_flips"])])
            out_str += row
            if r == "Never":
                out_str += end_row + "\n"
            else:
                out_str += "\\\\\n"
    out_str += "\\end{tabular}\n"
    out_str += "\\end{center}\n"
    out_str += "\\caption{Measures how often the solver learns clauses that would be subsumed by the backbone. " \
               "Further measures how many times the polarity of backbone literals are flipped during solving. " \
               "For each instance, the reported value is normalized by dividing by the size of the backdoor.}\n"
    out_str += "\\label{tab:lens_bb}\n"
    out_str += "\\end{table}\n"

    print(out_str)

    # LSR Tables

    lsr_cols = [col for col in df.columns if 'lsr_vr' in col]
    df_agile_lsr = df_agile[lsr_cols]
    df_app_lsr = df_app[lsr_cols]
    df_agile_lsr = df_agile_lsr.dropna(how=HOW)
    df_app_lsr = df_app_lsr.dropna(how=HOW)

    print("### LSR", len(df_app_lsr), len(df_agile_lsr))
    df_app_lsr = df_app_lsr[lsr_cols]
    lsr_app_avg = np.mean(df_app_lsr)
    for i in lsr_app_avg.items():
        print(i)

    df_agile_lsr = df_agile_lsr[lsr_cols]
    lsr_agile_avg = np.mean(df_agile_lsr)
    for i in lsr_agile_avg.items():
        print(i)
        
    

    # produce table
    end_row = " \\\\ \\hline"

    out_str = ""
    out_str += "\\begin{table}[t]\n"
    out_str += "\\begin{center}\n"
    out_str += "\\begin{tabular}{ |l|l||c|c|}\n"

    out_str += "\\hline"
    out_str += " & ".join(
        ["\\textbf{" + i + "}" for i in
         ["Branching", "Restart", "Application LSR", "Agile LSR"]]) + end_row + "\n"
    for b in ["LRB", "VSIDS", "Random"]:
        for r in ["Luby", "Always", "Never"]:
            row = ""
            if r == "Luby":
                row += "\multirow{3}{4em}{" + b + "} &"
            else:
                row += "& "
            row += r + " & "
            key = "pocr_" + b_map[b] + "_" + r_map[r] + "_"
            row += " & ".join(
                FSTR.format(i) for i in [lsr_app_avg[key + "lsr_vr"], lsr_agile_avg[key + "lsr_vr"]])
            out_str += row
            if r == "Never":
                out_str += end_row + "\n"
            else:
                out_str += "\\\\\n"
    out_str += "\\end{tabular}\n"
    out_str += "\\end{center}\n"
    out_str += "\\caption{LSR-backdoor comparison of the proofs/models found by each heuristic.}\n"
    out_str += "\\label{tab:lens_lsr}\n"
    out_str += "\\end{table}\n"

    print(out_str)

    """
    # Time tables

    time_cols = [col for col in df.columns if 'time' in col or 'conflicts' in col]
    df_agile_time = df_agile[time_cols]
    df_app_time = df_app[time_cols]
    df_agile_time = df_agile_time.dropna(how=HOW)
    df_app_time = df_app_time.dropna(how=HOW)

    print("### TIME", len(df_app_time), len(df_agile_time))
    time_app_avg = np.mean(df_app_time)
    time_agile_avg = np.mean(df_agile_time)

    # produce table
    end_row = " \\\\ \\hline"

    out_str = ""
    out_str += "\\begin{table}[t]\n"
    out_str += "\\begin{center}\n"
    out_str += "\\begin{tabular}{ |l|l||c|c|}\n"

    out_str += "\\hline"
    out_str += " & ".join(
        ["\\textbf{" + i + "}" for i in
         ["Branching", "Restart", "Solving Time (s)", "Number of Conflicts"]]) + end_row + "\n"
    for b in ["LRB", "VSIDS", "Random"]:
        for r in ["Luby", "Always", "Never"]:
            row = ""
            if r == "Luby":
                row += "\multirow{3}{4em}{" + b + "} &"
            else:
                row += "& "
            row += r + " & "
            key = "pocr_" + b_map[b] + "_" + r_map[r] + "_"
            row += " & ".join(
                FSTR.format(i) for i in [time_app_avg[key + "time"], time_agile_avg[key + "conflicts"]])
            out_str += row
            if r == "Never":
                out_str += end_row + "\n"
            else:
                out_str += "\\\\\n"
    out_str += "\\end{tabular}\n"
    out_str += "\\end{center}\n"
    out_str += "\\caption{Average solving time and number of conflicts for each heuristic.}\n"
    out_str += "\\label{tab:lens_time}\n"
    out_str += "\\end{table}\n"

    print(out_str)
    """



def create_extra_ratios(df):
    # df['qcor'] = df['q'] / df['num_cmtys']
    df['simp_qcor'] = df['simp_q'] / df['simp_num_cmtys']
    # df['cvr'] = df['num_clauses'] / df['num_vars']
    df['simp_cvr'] = df['simp_num_clauses'] / df['simp_num_vars']
    # df['tw_upper_vr'] = df['tw_upper'] / df['num_vars']
    df['simp_tw_uppervr'] = df['simp_tw_upper'] / df['simp_num_vars']
    df['simp_backbonesvr'] = df['simp_backbones'] / df['simp_num_vars']
    df['simp_wvr'] = df['simp_weak_size'] / df['simp_num_vars']
    df['simp_lvr'] = df['simp_lsr_size'] / df['simp_num_vars']
    df['simp_unionwvr'] = df['simp_num_vars_in_any_weak'] / df['simp_num_vars']
    df['simp_cmtysvr'] = df['simp_num_cmtys'] / df['simp_num_vars']

    df['simp_all_decsvr'] = df['simp_all_decs'] / df['simp_num_vars']
    df['simp_lsr_all_decs_unionvr'] = df['simp_lsr_all_decs_union'] / df['simp_num_vars']
    df['simp_lsr_all_decs_intervr'] = df['simp_lsr_all_decs_inter'] / df['simp_num_vars']


def regression_helper(df, benchmarks=None, times=None, subsets=None, subset_size_filter=None, rotate=False,
                      filter_under_second=False,
                      scale_features=True, log_time=True, heterogeneous=True, highest_order_features=-1,
                      grab_all=False, ridge=True):
    """
    Performs Ridge regression on subsets of features vs time.

    :param df: the DataFrame
    :param benchmarks: list of string IDs of the considered benchmarks
    :param times: list of solvers to be considered
    :param subsets: if given, only perform regression on these subsets of features
    :param subset_size_filter: if given, tries all subsets of features of this size
    :param rotate: used with subset_size_filter and multiple benchmarks (only dumps the best feature sets of each)
    :param filter_under_second: if True, remove any instances that solved under 1 second
    :param scale_features: currently a mean zero, std. dev. of 1 (time is not scaled by this)
    :param log_time: take log of time before regression
    :param heterogeneous: (Should probably set to True). Whenever we create a higher order feature, don't allow e.g. Q*Q
    :param highest_order_features: only allow subsets of features of given maximal size
    :param grab_all: output results of all regressions, rather than just the best for each benchmark
    :param ridge: if true, use ridge regression, else use linear
    :return:
    """

    if not benchmarks:
        benchmarks = ["app", "random", "crafted", "agile"]
    if not times:
        times = ['time']
    if not subsets:
        data_types = ["simp_num_vars", "simp_num_clauses", "simp_cvr",  # basic
                      "simp_num_cmtys", "simp_q", "simp_qcor",  # cmty
                      "simp_lsr_size", "simp_lvr",  # lsr
                      "simp_tw_upper", "simp_tw_uppervr"  # tw
                      ]

        df = df[data_types + ['benchmark', 'time']]
        df = df.dropna()

        subsets = [list(i) for i in all_subsets(data_types, 1)]

    # filter out subsets of wrong size
    if subset_size_filter:
        subsets = [i for i in subsets if len(i) == subset_size_filter]

    if log_time:
        for t in times:
            df['log_' + t] = df[t].apply(lambda x: 0.01 if x <= 0 else x)
            df['log_' + t] = df['log_' + t].apply(numpy.log)
    else:
        # TODO clean
        for t in times:
            df['log_' + t] = df[t]

    features = []
    r2_values = []
    num_instances = []
    for s in subsets:
        r2_inst = []
        num_insts_inst = []
        for c in benchmarks:
            for t in times:
                print(r2_values)
                if benchmarks == ["all"]:
                    curr_df = df
                else:
                    curr_df = df.loc[df['benchmark'] == c]
                if filter_under_second:
                    curr_df = curr_df.loc[df[t] > 1]
                curr_df = curr_df[s + ['log_' + t]].dropna()
                if len(curr_df) < 4:
                    print("df too small", s)
                    r2_inst.append("N/A")
                    num_insts_inst.append("N/A")
                    continue

                # scale features, but not time
                if scale_features:
                    # use this for 0-1 scaling
                    # curr_df[curr_df.columns.difference(['log_' + t])] -=
                    #        curr_df[curr_df.columns.difference(['log_' + t])].min()
                    # curr_df[curr_df.columns.difference(['log_' + t])]
                    #        /= curr_df[curr_df.columns.difference(['log_' + t])].max()

                    # use this for mean 0 std 1 scaling
                    curr_df[curr_df.columns.difference(['log_' + t])] \
                        -= curr_df[curr_df.columns.difference(['log_' + t])].mean()
                    curr_df[curr_df.columns.difference(['log_' + t])] \
                        /= curr_df[curr_df.columns.difference(['log_' + t])].std()

                fnames = add_higher_order_features(curr_df, s, highest_order_features, heterogenous=heterogeneous)

                model = sm.ols(data=curr_df, formula="log_" + t + " ~ " + "+".join(s + fnames))

                if ridge:
                    # if L1_wt = 0, then it's ridge; if it's 1, then it's lasso
                    res = model.fit_regularized(L1_wt=0)
                else:
                    res = model.fit()
                print(res.summary())
                print(res.pvalues)

                r2 = res.rsquared_adj

                print(r2)
                instances = len(curr_df.index)
                if instances < 1:
                    r2 = "N/A"
                    instances = "N/A"
                r2_inst.append(r2)
                num_insts_inst.append(instances)

        na_test = set(r2_inst)
        if list(na_test) == ["N/A"]:
            print("in here")
            continue
        features.append("$" + "\oplus{}".join([type_map[t] for t in s]) + "$")
        r2_values.append(r2_inst)
        num_instances.append(num_insts_inst)

    print("counts", len(features), len(r2_values), len(num_instances))
    items = list(zip(features, r2_values, num_instances))
    print("ITEMS:", items)
    big_list = []
    if rotate:
        for i in range(len(benchmarks)):
            curr_items = sorted(items, key=lambda x: x[1][i] if x[1][i] != "N/A" else -1, reverse=True)
            big_list.append(curr_items)
    else:
        big_list.append(items)

    out_table_rows = []
    index = 0
    print("Big list:", big_list)
    for l in big_list:
        rows = []
        if not l:
            continue
        if not grab_all:
            f, r2s, insts = l[0]
            out_row = [f]
            count = 0
            for i, j in zip(r2s, insts):
                value = str(FSTR.format(i)) + " (" + str(j) + ")" if i != "N/A" else "N/A"
                if count == index:
                    value = "\\textbf{" + value + "}"
                out_row.append(value)
                count += 1
            out_table_rows.append(out_row)
            index += 1
        for f, r2s, insts in l:
            row = [f] + [str(FSTR.format(i)) + " (" + str(j) + ")" if i != "N/A" else "N/A" for i, j in zip(r2s, insts)]
            rows.append(row)
            if grab_all:
                out_table_rows.append(row)
    return out_table_rows


def regression(df, benchmarks, out_file, caption_prefix=None):
    """
    Tests if subsets of features correlate with solving time.
    """
    heterogeneous_r2 = regression_helper(df, benchmarks=benchmarks, subsets=[
        ["simp_num_vars", "simp_num_clauses", "simp_cvr"],
        ["simp_num_vars", "simp_num_clauses", "simp_num_cmtys", "simp_q"],
        ["simp_num_vars", "simp_num_clauses", "simp_lsr_size", "simp_lvr"],
        ["simp_num_vars", "simp_num_clauses", "simp_num_min_weak", "simp_weak_size"],
        ["simp_num_vars", "simp_num_clauses", "simp_backbones", "simp_backbonesvr"],
        ["simp_num_vars", "simp_num_clauses", "simp_tw_upper", "simp_tw_uppervr"]
    ],
                                         rotate=False, grab_all=True, ridge=False)

    data_types = ["simp_num_vars", "simp_num_clauses", "simp_cvr",  # basic
                  "simp_num_cmtys", "simp_q", "simp_qcor",  # cmty
                  "simp_lsr_size", "simp_lvr",  # lsr
                  "simp_tw_upper", "simp_tw_uppervr"  # tw
                  ]

    df = df[data_types + ['benchmark', 'time']]
    df = df.dropna()


    # NOTE: ordered based on significance values
    best_combined_r2 = regression_helper(df, benchmarks=benchmarks, subsets=[
        ["simp_q", "simp_cvr", "simp_lvr", "simp_qcor", "simp_num_clauses"],
        ["simp_tw_uppervr", "simp_q", "simp_num_cmtys", "simp_tw_upper", "simp_lvr"],
        ["simp_qcor", "simp_lvr", "simp_num_clauses", "simp_lsr_size", "simp_q"],
        ["simp_num_cmtys", "simp_tw_uppervr", "simp_cvr", "simp_tw_upper", "simp_q"]
    ],
    # ["simp_num_vars", "simp_num_clauses", "simp_tw_upper", "simp_lsr_size", "simp_q", "simp_num_cmtys"],
    # ["simp_num_vars", "simp_num_clauses", "simp_tw_upper", "simp_tw_uppervr"]],
                                 rotate=False, grab_all=True, ridge=False)

    # best_combined_r2 = regression_helper(df, benchmarks=benchmarks, subset_size_filter=5, rotate=True)
    rows = heterogeneous_r2 + [["\\hline"]] + best_combined_r2

    with open(out_file, 'w') as o:
        latex_gen.insert_table(o, rows, tiny=False, headers=["Feature Set"] + benchmarks,
                               caption=caption_prefix + " Adjusted R$^2$ values for the given features, "
                               + "compared to log of MapleCOMSPS' solving time. "
                               + "The number in parentheses indicates the number of instances "
                               + "that were considered in each case. The lower section considers "
                               + "heterogeneous sets of features across different parameter types.",
                               label="tab-regressions", tabular=True)


def q5(out_file):
    data = []
    for c in ["/home/ezulkosk/backdoors_benchmarks/" + i + "/" for i in comps] + \
            ["/home/ezulkosk/backdoors_benchmarks/agile/", "/home/ezulkosk/backdoors_benchmarks/crafted/",
             "/home/ezulkosk/backdoors_benchmarks/random/"]:
        sense, spec = correlate_backdoors_and_bridges(c)
        if isinstance(c, list):
            c = "application"
        else:
            c = c.strip("/").split("/")[-1]
        data.append((c, sense, spec))
        print(tabulate(data, headers=["Benchmark", "Sensitivity", "Specificity"], tablefmt="latex"))

    with open(out_file, 'w') as o:
        latex_gen.insert_table(o, data, headers=["Benchmark", "Sensitivity", "Specificity"], caption="Bridge/BD Expt.")


def collect_data(df, benchmarks, tables_dir, prefix_label, tex):
    if not prefix_label:
        caption_prefix = ""
    else:
        caption_prefix = prefix_label.replace("_", " ").upper() + ":"

    data_summary(df, benchmarks, tables_dir + prefix_label + "datasummary.tex", caption_prefix=caption_prefix)
    average_metric_values(df, benchmarks, tables_dir + prefix_label + "averagemetrics.tex",
                          caption_prefix=caption_prefix)
    # structure_logging_summary(d, ["Agile"], tables_dir + l + "structure_logging.tex",
    #                           full=True, caption_prefix=caption_prefix)
    lsr_all_decs_comparison(df,
                            benchmarks,
                            tables_dir + prefix_label + "lsr_all_decs_comparison.tex",
                            caption_prefix=caption_prefix)

    regression(df, benchmarks, tables_dir + prefix_label + "regression.tex", caption_prefix=caption_prefix)
    tex.write("\\begin{table}[t]\n")
    for i in ["datasummary.tex", "averagemetrics.tex", "lsr_all_decs_comparison.tex",
              "regression.tex"]:  # "regression.tex", "same_insts_regression.tex", "bridgebd.tex"]:
        filename = tables_dir + prefix_label + i
        tex.write("\\input{" + filename + "}")
        tex.write("\n")
    tex.write("\\end{table}\n")


def collect_refined_app_data(df, benchmarks, tables_dir, tex, split_sat=True, split_buckets=True):
    # to add a refinement, create a tuple (label, constraint)
    refinements = []
    if split_sat:
        sat_refinement = ("sat_", lambda x: x.result != 'SAT')
        unsat_refinement = ("unsat_", lambda x: x.result != 'UNSAT')
        refinements.append((sat_refinement, unsat_refinement))
    if split_buckets:
        hardware_refinement = ("hardware_", lambda x: x.bucket != 'hardware')
        software_refinement = ("software_", lambda x: x.bucket != 'software')
        crypto_refinement = ("crypto_", lambda x: x.bucket != 'crypto')
        refinements.append((hardware_refinement, software_refinement, crypto_refinement))

    all_refinements = itertools.product(*refinements)
    for refine_list in all_refinements:

        curr_df = df.copy(deep=True)
        full_label = ""
        for (label, refinement) in refine_list:
            full_label += label
            print(curr_df['bucket'])
            curr_df = curr_df.drop(curr_df[refinement].index)
        collect_data(curr_df, benchmarks, tables_dir, full_label, tex)


def main():
    base_dir, data_dir, case_studies, fields = init_cp2017_setup()
    benchmarks = ["Application", "Crafted", "Random", "Agile"]

    pd.set_option('display.max_rows', 1500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    tex_dir = "/home/ezulkosk/cp2017_benchmarks/analysis/"
    tables_dir = tex_dir + "tables/"
    tex_file = tex_dir + "/db.tex"
    tex = latex_gen.gen_tex_article(tex_file)
    # tex.write('\\tiny\n')

    if True:
        create_sat_db_csv(data_dir, fields)
    df = pd.read_csv(data_dir + 'db.csv')

    # TODO replacing sat lsr with sat min
    # print(df['simp_sat_min_lsr'])
    df['simp_lsr_size2'] = df['simp_sat_min_lsr']
    df['simp_lsr_size2'][df['simp_sat_min_lsr'].isnull()] = df['simp_lsr_size']
    df['simp_lsr_size'] = df['simp_lsr_size2']

    # add extra ratios
    create_extra_ratios(df)

    df['time'] = df['maplecomsps'] - df['maplecomsps_pp']


    df['benchmark'] = df['benchmark'].map({'agile': 'Agile',
                                           'crafted': 'Crafted',
                                           'random': 'Random',
                                           'app': 'Application'
                                           })

    df.drop(df[df.simp_num_vars < 1].index, inplace=True)

    # drop any duplicate entries
    df.drop_duplicates(inplace=True)

    collect_refined_app_data(df, benchmarks, tables_dir, tex)
    collect_data(df, benchmarks, tables_dir, "", tex)



    b = ["Agile"]
    pocr_identifiers = ["gini_picks", "gini_clauses"]
    pocr_headers = ["Gini Picks"]
    pocr_captions = "Gini Picks"
    pocr_label = "tab:picks"
    sat = "UNSAT"

    pocr_expt_params = [(b, ["gini_picks", "gini_clauses"], ["Gini Picks", "Gini Clauses"],
                         "Cmty Gini Expt.", "tab:gini"),

                        (b, ["lsr_cmty_largest_ratio", "lsr_cmty_spread"], ["Largest Ratio", "Spread"], "LSR Cmty Expt.",
                        "tab:lsr_cmty")
                        ]


    for p in pocr_expt_params:
        for s in [None, "SAT", "UNSAT"]:
            pocr_structure_logging_summary(df, *p, tex, sat=s)


    latex_gen.end_tex_article(tex)
    latex_gen.open_tex_pdf(tex_file)


if __name__ == '__main__':
    main()
