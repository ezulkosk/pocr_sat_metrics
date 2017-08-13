import operator
import os
import subprocess

import sys

from old_expts.data_analysis import join_csvs, coerce
from phdtk.benchmark import Benchmark
from phdtk.db import create_csv, init_sat_setup, create_sat_db_csv, init_qfbv_setup, init_gtn_setup, init_cp2017_setup, \
    init_scaling_setup
from phdtk.resources import basename
from phdtk.sat_reader import SatInstance

__author__ = 'ezulkosk'


CCMIN = "ccmin2"

IGNORE_FOUND = True  # if a resource file already exists, don't recompute

def csv_line(*args):
    return ",".join([str(i) for i in args]) + "\n"


#########################  CMTY PP  ###################################################################


def process_cmty_pp_output_instance(file_map):
    f = file_map['out_file']
    name = file_map['basename']
    print(f)
    stream = open(f)
    found = False
    time = None
    cpu_time, solver_calls, pp_time, solve_time = "", "", "", ""
    with open(f) as stream:
        for line in stream:
            if line.startswith("WARNING:"):
                continue
            if "CPU time" in line:
                cpu_time = line.strip().split()[3]
            elif "Num Calls" in line:
                solver_calls = line.strip().split()[2]
            elif "Cmty Pre Time" in line:
                pp_time = line.strip().split()[3]
            elif "Solve time" in line:
                solve_time = line.strip().split()[2]
            elif "Solved by simplification" in line:
                solve_time = 0

        return name, cpu_time, pp_time, solve_time, solver_calls

def process_cmty_pp_output(base_dir, data_dir, case_study, simp = False):
    if simp:
        print("Simp solver output not supported")
        return
    bench = Benchmark()
    print("Solver output:", case_study)
    result_map = {}
    configs = ["cmty_pp_basic_0", "cmty_pp_basic_2", "cmty_pp_c2pairs_0"]
    allow_empty = False
    for s in configs:
        print(base_dir + case_study + "/" + s)
        f = bench.add_files(base_dir + case_study + "/" + s, "out_file", ".result", allow_empty_files=allow_empty,
                            ignore_found_file=data_dir + case_study + ".result")
        if not f:
            print("Solver output -- " + s + " -- " + case_study + " failed")
            continue
        bench.compile_experiment()
        res = bench.run_experiment(process_cmty_pp_output_instance)

        with open(data_dir + case_study + "." + s + "_cpu_time", 'a') as cpu_file, \
            open(data_dir + case_study + "." + s + "_pp_time", 'a') as pp_file, \
            open(data_dir + case_study + "." + s + "_solve_time", 'a') as solve_file, \
            open(data_dir + case_study + "." + s + "_solver_calls", 'a') as calls_file:
            for (name, cpu_time, pp_time, solve_time, solver_calls) in res:
                cpu_file.write(csv_line(name, cpu_time))
                pp_file.write(csv_line(name, pp_time))
                solve_file.write(csv_line(name, solve_time))
                calls_file.write(csv_line(name, solver_calls))

#########################    GTN    ###################################################################

def process_gtn_ls_output(base_dir, data_dir, case_study, simp = False):
    if simp:
        print("Simp solver output not supported")
        return
    bench = Benchmark()
    print("LS output:", case_study)
    allow_empty = False
    seeds = ["100", "200", "300", "400", "500"]

    files = bench.add_files(base_dir + case_study + "/learning/" , "out_file", ".ls_out", allow_empty_files=allow_empty,
                        ignore_found_file=data_dir + case_study + ".learning")
    if not files:
        print("Learning output -- " + case_study + " failed")
        return

    m = {}
    for f in files:
        with open(f) as stream:
            for line in stream:
                if "NumUniq" in line:
                    gtn_size = f.strip().split("/")[-1].split(".")[0].split("t")[1]
                    arr = line.strip().split()
                    ls_size = arr[2]
                    m[gtn_size] = min(m.get(gtn_size, 99999999), int(ls_size))

    out_file = open(data_dir + case_study + ".learning", 'a')
    for k, v in m.items():
        name = "gt" + str(k)
        out_file.write(csv_line(name, v))
    out_file.close()



#######################################################################################################


def process_gtn_solver_output_instance(file_map):
    f = file_map['out_file']
    name = file_map['basename']
    print(f)
    stream = open(f)
    time = ""
    restarts = ""
    for line in stream:
        if line.startswith("WARNING:"):
            continue
        if "cpu time" in line.lower():
            index = 3
            time = line.split()[index]
        if "restarts" in line.lower():
            index = 2
            restarts = line.split()[index]
    return name, time, restarts

def process_gtn_solver_output(base_dir, data_dir, case_study, simp = False):
    if simp:
        print("Simp solver output not supported")
        return
    bench = Benchmark()
    print("Solver output:", case_study)
    allow_empty = False
    solvers = ["maplesat", "maplesat_nr", "maplesat_ar", "rwr", "rnr", "rar"]
    for s in solvers:
        print(base_dir + case_study + "/" + s)
        f = bench.add_files(base_dir + case_study + "/" + s, "out_file", ".result", allow_empty_files=allow_empty,
                            ignore_found_file=data_dir + case_study + "." + s)
        if not f:
            print("Solver output -- " + s + " -- " + case_study + " failed")
            continue
        bench.compile_experiment()
        res = bench.run_experiment(process_gtn_solver_output_instance)

        out_file = open(data_dir + case_study + "." + s, 'a')
        restarts_file = open(data_dir + case_study + "." + s + "_restarts", 'a')
        for (name, time, restarts) in res:
            out_file.write(csv_line(name, time))
            restarts_file.write(csv_line(name, restarts))
        out_file.close()
        restarts_file.close()



#######################################################################################################

def process_gtn_proof_instance(file_map):
    clause_file = file_map['out_file']
    name = file_map['basename']
    num_used_lemmas = 0
    largest_used_lemma = 0
    num_units = 0
    num_binary = 0
    avg_clause_total = 0
    print(name)
    with open(clause_file) as stream:
        for line in stream:
            arr = line.strip().split()
            if arr[0] == 'd':
                continue
            else:
                num_used_lemmas += 1
                avg_clause_total += len(arr) - 1
                if len(arr) - 1 > largest_used_lemma:
                    largest_used_lemma = len(arr) - 1
                if len(arr) - 1 == 2:
                    num_binary += 1
                if len(arr) - 1 == 1:
                    num_units += 1
    return name, num_used_lemmas, largest_used_lemma, num_units, num_binary, avg_clause_total / num_used_lemmas

def process_gtn_proof(base_dir, data_dir, case_study, simp=False):
    if simp:
        print("Simp solver output not supported")
        return
    bench = Benchmark()
    print("Solver output:", case_study)
    allow_empty = False
    solvers = ["maplesat", "maplesat_nr", "maplesat_ar", "rwr", "rnr"]
    for s in solvers:
        print(s)
        print(base_dir + case_study + "/lemmas")
        f = bench.add_files(base_dir + case_study + "/lemmas", "out_file", "." + s,
                            allow_empty_files=allow_empty,
                            ignore_found_file=data_dir + case_study + "." + s + "_proof_units")
        if not f:
            print("Clauses proof log -- " + s + " -- " + case_study + " failed")
            continue
        bench.compile_experiment()
        res = bench.run_experiment(process_gtn_proof_instance)

        case_study = "gtn"
        with open(data_dir + case_study + "." + s + "_proof_avg_clause", 'a') as avg_clause_file, \
                open(data_dir + case_study + "." + s + "_proof_units", 'a') as units_file, \
                open(data_dir + case_study + "." + s + "_proof_binary", 'a') as binary_file, \
                open(data_dir + case_study + "." + s + "_proof_largest", 'a') as largest_file, \
                open(data_dir + case_study + "." + s + "_proof_num_clauses", 'a') as num_clauses_file:
            for (name, num_used, largest, units, binary, avg_clause) in res:
                avg_clause_file.write(csv_line(name, avg_clause))
                units_file.write(csv_line(name, units))
                binary_file.write(csv_line(name, binary))
                largest_file.write(csv_line(name, largest))
                num_clauses_file.write(csv_line(name, num_used))
        case_study = ""

#######################################################################################################


def process_gtn_clauses_instance(file_map):
    clause_file = file_map['out_file']
    name = file_map['basename']
    num_clauses = 0
    total_size = 0
    num_units = 0
    num_binary = 0
    with open(clause_file) as stream:
        for line in stream:
            arr = line.strip().split()
            num_clauses += 1
            total_size += len(arr)
            if len(arr) == 1:
                num_units += 1
            elif len(arr) == 2:
                num_binary += 1
    return name, total_size / num_clauses, num_units, num_binary


def process_gtn_clauses(base_dir, data_dir, case_study, simp = False):
    if simp:
        print("Simp solver output not supported")
        return
    bench = Benchmark()
    print("Solver output:", case_study)
    allow_empty = False
    solvers = ["maplesat", "maplesat_nr", "rwr", "rnr"]
    for s in solvers:
        print(base_dir + case_study + "/" + s)
        f = bench.add_files(base_dir + case_study + "/" + s + "_clauses_log", "out_file", ".clauses_log", allow_empty_files=allow_empty,
                            ignore_found_file=data_dir + case_study + "." + s + "_units")
        if not f:
            print("Clauses log -- " + s + " -- " + case_study + " failed")
            continue
        bench.compile_experiment()
        res = bench.run_experiment(process_gtn_clauses_instance)

        with open(data_dir + case_study + "." + s + "_avg_clause", 'a') as avg_clause_file, \
            open(data_dir + case_study + "." + s + "_units", 'a') as units_file, \
            open(data_dir + case_study + "." + s + "_binary", 'a') as binary_file:
            for (name, avg_clause, units, binary) in res:
                avg_clause_file.write(csv_line(name, avg_clause))
                units_file.write(csv_line(name, units))
                binary_file.write(csv_line(name, binary))

#########################  END GTN  ###################################################################

def process_simple_size_instance(file_map):
    b = file_map['data']
    name = file_map['basename']
    s = SatInstance()
    data = s.read_one_int_per_line_file(b)
    return name, len(data)


#######################################################################################################


def process_simple_output_instance(file_map):
    w = file_map['data']
    name = file_map['basename']
    print(w, name)
    stream = open(w)
    line = stream.readline()
    line = line.strip()
    return name, line


def process_simple_output(base_dir, data_dir, case_study, data_type):
    """
    Produces CSV for various simple tests (.data_type), where the output files are assumed to have one line of data.
    """
    bench = Benchmark()
    f = bench.add_files(base_dir + case_study + "/" + data_type + "/", "data", "." + data_type, allow_empty_files=True)
    if not f:
        print(case_study + " " + data_type + " failed")
        return None
    bench.compile_experiment()
    res = bench.run_experiment(process_simple_output_instance)
    data_file = open(data_dir + case_study + "." + data_type, 'w')
    for (name, data) in res:
        data_file.write(csv_line(name, data))
    data_file.close()


#######################################################################################################



def process_cdcl_structure_output_instance_agile(file_map):
    w = file_map['data']
    name = file_map['basename']
    l, al, nl = "","",""  # lsr size
    cl, acl, ncl = "","",""   # avg clause lsr
    p, ap, np = "","",""   # picks
    c, ac, nc = "","",""  # clauses picks
    b, ab, nb = "","",""   # backbone flips
    bs, abs_, nbs = "","","" # backbone subsumed clauses

    nVars = None

    mode = None
    with open(w) as stream:
        for line in stream:
            if "Never restart" in line:
                mode = "nr"
            elif "Always restart" in line:
                mode = "ar"
            elif "Standard restarts" in line:
                mode = "sr"
            elif "LSR Backdoors" in line:
                arr = line.strip().split()
                nVars = int(arr[4].strip("]"))
                temp = arr[6]
                temp = temp.strip("%")
                temp = float(temp) / 100
                temp = str(temp)
                if mode == "sr":
                    l = temp
                elif mode == "ar":
                    al = temp
                elif mode == "nr":
                    nl = temp
                else:
                    print("Mode not set properly")
                    sys.exit(1)
            elif "AvgClauseLSR" in line:
                # normalize with nVars
                temp = line.strip().split()[1]
                if "nan" in temp:
                    temp = ""
                else:
                    temp = str(float(temp) / nVars)
                if mode == "sr":
                    cl = temp
                elif mode == "ar":
                    acl = temp
                elif mode == "nr":
                    ncl = temp
                else:
                    print("Mode not set properly")
            elif "GiniNormalizedPicks" in line:
                # normalize with nVars
                temp = str(line.strip().split()[1])
                if mode == "sr":
                    p = temp
                elif mode == "ar":
                    ap = temp
                elif mode == "nr":
                    np = temp
                else:
                    print("Mode not set properly")
            elif "GiniNormalizedClauses" in line:
                # normalize with nVars
                temp = str(line.strip().split()[1])
                if float(temp) <= 0:
                    temp = ""
                if mode == "sr":
                    c = temp
                elif mode == "ar":
                    ac = temp
                elif mode == "nr":
                    nc = temp
                else:
                    print("Mode not set properly")
            elif "NormalizedBackboneFlips" in line:
                # normalize with nVars
                temp = str(line.strip().split()[1])
                if mode == "sr":
                    b = temp
                elif mode == "ar":
                    ab = temp
                elif mode == "nr":
                    nb = temp
                else:
                    print("Mode not set properly")
            elif "NormalizedBackboneSubsumedClauses" in line:
                # normalize with nVars
                temp = str(line.strip().split()[1])
                if mode == "sr":
                    bs = temp
                elif mode == "ar":
                    abs_ = temp
                elif mode == "nr":
                    nbs = temp
                else:
                    print("Mode not set properly")

    return (name,
         l, al, nl,    # lsr size
         cl, acl, ncl, # avg clause lsr
         p, ap, np,    # picks
         c, ac, nc,    # clauses picks
         b, ab, nb,    # backbone flips
         bs, abs_, nbs # backbone subsumed clauses
         )

def process_cdcl_structure_output_instance(file_map):
    w = file_map['data']
    name = file_map['basename']
    l, al, nl = "","",""  # lsr size
    cl, acl, ncl = "","",""   # avg clause lsr
    p, ap, np = "","",""   # picks
    c, ac, nc = "","",""  # clauses picks
    b, ab, nb = "","",""   # backbone flips
    bs, abs_, nbs = "","","" # backbone subsumed clauses

    nVars = None

    mode = None
    with open(w) as stream:
        for line in stream:
            if "Never restart" in line:
                mode = "nr"
            elif "Always restart" in line:
                mode = "ar"
            elif "Standard restarts" in line:
                mode = "sr"
            elif "LSR Backdoors" in line:
                arr = line.strip().split()
                nVars = int(arr[4].strip("]"))
                temp = arr[6]
                temp = temp.strip("%")
                temp = float(temp) / 100
                temp = str(temp)
                if mode == "sr":
                    l = temp
                elif mode == "ar":
                    al = temp
                elif mode == "nr":
                    nl = temp
                else:
                    print("Mode not set properly")
                    sys.exit(1)
            elif "AvgClauseLSR" in line:
                # normalize with nVars
                temp = line.strip().split()[1]
                if "nan" in temp:
                    temp = ""
                else:
                    temp = str(float(temp) / nVars)
                if mode == "sr":
                    cl = temp
                elif mode == "ar":
                    acl = temp
                elif mode == "nr":
                    ncl = temp
                else:
                    print("Mode not set properly")
            elif "GiniNormalizedPicks" in line:
                # normalize with nVars
                temp = str(line.strip().split()[1])
                if mode == "sr":
                    p = temp
                elif mode == "ar":
                    ap = temp
                elif mode == "nr":
                    np = temp
                else:
                    print("Mode not set properly")
            elif "GiniNormalizedClauses" in line:
                # normalize with nVars
                temp = str(line.strip().split()[1])
                if float(temp) <= 0:
                    temp = ""
                if mode == "sr":
                    c = temp
                elif mode == "ar":
                    ac = temp
                elif mode == "nr":
                    nc = temp
                else:
                    print("Mode not set properly")
            elif "NormalizedBackboneFlips" in line:
                # normalize with nVars
                temp = str(line.strip().split()[1])
                if mode == "sr":
                    b = temp
                elif mode == "ar":
                    ab = temp
                elif mode == "nr":
                    nb = temp
                else:
                    print("Mode not set properly")
            elif "NormalizedBackboneSubsumedClauses" in line:
                # normalize with nVars
                temp = str(line.strip().split()[1])
                if mode == "sr":
                    bs = temp
                elif mode == "ar":
                    abs_ = temp
                elif mode == "nr":
                    nbs = temp
                else:
                    print("Mode not set properly")

    return (name,
         l, al, nl,    # lsr size
         cl, acl, ncl, # avg clause lsr
         p, ap, np,    # picks
         c, ac, nc,    # clauses picks
         b, ab, nb,    # backbone flips
         bs, abs_, nbs # backbone subsumed clauses
         )

def process_cdcl_structure_output(base_dir, data_dir, case_study, simp=False):
    """
    For now, I'm only grabbing data when I have it all
    """
    print(case_study + " " + ("simp " if simp else "") + "structure_logging" + " running")
    base = base_dir + case_study
    if simp:
        base += "/simp/"
    print("dir", base + "/" + "structure_logging" + "/")
    bench = Benchmark()
    f = bench.add_files(base + "/" + "structure_logging" + "/", "data", ".out", allow_empty_files=False)
    if not f:
        print(case_study + " " + "structure_logging" + " failed")
        return None
    bench.compile_experiment()
    res = bench.run_experiment(process_cdcl_structure_output_instance)

    lsr = open(data_dir + case_study + ".struct_lsr", 'w')
    ar_lsr = open(data_dir + case_study + ".struct_ar_lsr", 'w')
    nr_lsr = open(data_dir + case_study + ".struct_nr_lsr", 'w')

    avg_clause_lsr = open(data_dir + case_study + ".struct_avg_clause_lsr", 'w')
    ar_avg_clause_lsr = open(data_dir + case_study + ".struct_ar_avg_clause_lsr", 'w')
    nr_avg_clause_lsr = open(data_dir + case_study + ".struct_nr_avg_clause_lsr", 'w')

    gini_normalized_picks = open(data_dir + case_study + ".struct_gini_normalized_picks", 'w')
    ar_gini_normalized_picks = open(data_dir + case_study + ".struct_ar_gini_normalized_picks", 'w')
    nr_gini_normalized_picks = open(data_dir + case_study + ".struct_nr_gini_normalized_picks", 'w')

    gini_normalized_clauses = open(data_dir + case_study + ".struct_gini_normalized_clauses", 'w')
    ar_gini_normalized_clauses = open(data_dir + case_study + ".struct_ar_gini_normalized_clauses", 'w')
    nr_gini_normalized_clauses = open(data_dir + case_study + ".struct_nr_gini_normalized_clauses", 'w')

    normalized_backbone_flips = open(data_dir + case_study + ".struct_normalized_backbone_flips", 'w')
    ar_normalized_backbone_flips = open(data_dir + case_study + ".struct_ar_normalized_backbone_flips", 'w')
    nr_normalized_backbone_flips = open(data_dir + case_study + ".struct_nr_normalized_backbone_flips", 'w')

    normalized_backbone_subsumed_clauses = open(data_dir + case_study + ".struct_normalized_backbone_subsumed_clauses", 'w')
    ar_normalized_backbone_subsumed_clauses = open(data_dir + case_study + ".struct_ar_normalized_backbone_subsumed_clauses", 'w')
    nr_normalized_backbone_subsumed_clauses = open(data_dir + case_study + ".struct_nr_normalized_backbone_subsumed_clauses", 'w')

    for (name,
         l, al, nl,    # lsr size
         cl, acl, ncl, # avg clause lsr
         p, ap, np,    # picks
         c, ac, nc,    # clauses picks
         b, ab, nb,    # backbone flips
         bs, abs_, nbs # backbone subsumed clauses
         ) in res:
        lsr.write(csv_line(name, l))
        ar_lsr.write(csv_line(name, al))
        nr_lsr.write(csv_line(name, nl))

        avg_clause_lsr.write(csv_line(name, cl))
        ar_avg_clause_lsr.write(csv_line(name, acl))
        nr_avg_clause_lsr.write(csv_line(name, ncl))

        gini_normalized_picks.write(csv_line(name, p))
        ar_gini_normalized_picks.write(csv_line(name, ap))
        nr_gini_normalized_picks.write(csv_line(name, np))

        gini_normalized_clauses.write(csv_line(name, c))
        ar_gini_normalized_clauses.write(csv_line(name, ac))
        nr_gini_normalized_clauses.write(csv_line(name, nc))

        normalized_backbone_flips.write(csv_line(name, b))
        ar_normalized_backbone_flips.write(csv_line(name, ab))
        nr_normalized_backbone_flips.write(csv_line(name, nb))

        normalized_backbone_subsumed_clauses.write(csv_line(name, bs))
        ar_normalized_backbone_subsumed_clauses.write(csv_line(name, abs_))
        nr_normalized_backbone_subsumed_clauses.write(csv_line(name, nbs))

    lsr.close()
    ar_lsr.close()
    nr_lsr.close()
    avg_clause_lsr.close()
    ar_avg_clause_lsr.close()
    nr_avg_clause_lsr.close()
    gini_normalized_picks.close()
    ar_gini_normalized_picks.close()
    nr_gini_normalized_picks.close()
    gini_normalized_clauses.close()
    ar_gini_normalized_clauses.close()
    nr_gini_normalized_clauses.close()
    normalized_backbone_flips.close()
    ar_normalized_backbone_flips.close()
    nr_normalized_backbone_flips.close()
    normalized_backbone_subsumed_clauses.close()
    ar_normalized_backbone_subsumed_clauses.close()
    nr_normalized_backbone_subsumed_clauses.close()


def process_cdcl_structure_output_updated_instance(file_map):
    fl = file_map['lsr']
    fla = file_map['ar_lsr']
    fln = file_map['nr_lsr']
    fc = file_map['cmty']
    fca = file_map['ar_cmty']
    fcn = file_map['nr_cmty']
    name = file_map['basename']
    l, al, nl = "","",""  # lsr size
    # cl, acl, ncl = "","",""   # avg clause lsr
    p, ap, np = "","",""   # picks
    c, ac, nc = "","",""  # clauses picks

    nVars = None

    mode = None
    with open(fl) as stream:
        count = 0
        for line in stream:
            count += 1
        l = str(count)
    with open(fla) as stream:
        count = 0
        for line in stream:
            count += 1
        al = str(count)

    with open(fln) as stream:
        count = 0
        for line in stream:
            count += 1
        nl = str(count)

    with open(fc) as stream:
        for line in stream:
            if "GiniNormalizedPicks" in line:
                p = str(line.strip().split()[1])
            elif "GiniNormalizedClauses" in line:
                c = str(line.strip().split()[1])

    with open(fca) as stream:
        for line in stream:
            if "GiniNormalizedPicks" in line:
                ap = str(line.strip().split()[1])
            elif "GiniNormalizedClauses" in line:
                ac = str(line.strip().split()[1])

    with open(fcn) as stream:
        for line in stream:
            if "GiniNormalizedPicks" in line:
                np = str(line.strip().split()[1])
            elif "GiniNormalizedClauses" in line:
                nc = str(line.strip().split()[1])

    return (name,
         l, al, nl,    # lsr size
         #cl, acl, ncl, # avg clause lsr
         p, ap, np,    # picks
         c, ac, nc,    # clauses picks
         )

def process_cdcl_structure_output_updated(base_dir, data_dir, case_study, simp=False):
    """
    For now, I'm only grabbing data when I have it all
    """
    print(case_study + " " + ("simp " if simp else "") + "structure_logging" + " running")
    base = base_dir + case_study
    if simp:
        base += "/simp/"
    print("dir", base + "/" + "structure_logging" + "/")
    bench = Benchmark()
    f = bench.add_files(base + "/" + CCMIN + "_confl_side_lsr" + "/", "lsr", ".lsr", allow_empty_files=False)
    f2 = bench.add_files(base + "/" + CCMIN + "_ar_confl_side_lsr" + "/", "ar_lsr", ".lsr", allow_empty_files=False)
    f3 = bench.add_files(base + "/" + CCMIN + "_nr_confl_side_lsr" + "/", "nr_lsr", ".lsr", allow_empty_files=False)

    f4 = bench.add_files(base + "/" + CCMIN + "_cmty_loc" + "/", "cmty", ".cmty_loc", allow_empty_files=False)
    f5 = bench.add_files(base + "/" + CCMIN + "_ar_cmty_loc" + "/", "ar_cmty", ".cmty_loc", allow_empty_files=False)
    f6 = bench.add_files(base + "/" + CCMIN + "_nr_cmty_loc" + "/", "nr_cmty", ".cmty_loc", allow_empty_files=False)

    if not (f and f2 and f3 and f4 and f5 and f6):
        print(case_study + " " + "structure_logging" + " failed")
        print(f)
        print(f2)
        print(f3)
        print(f4)
        print(f5)
        print(f6)
        return None
    bench.compile_experiment()
    res = bench.run_experiment(process_cdcl_structure_output_updated_instance)

    lsr = open(data_dir + case_study + ".struct_lsr", 'w')
    ar_lsr = open(data_dir + case_study + ".struct_ar_lsr", 'w')
    nr_lsr = open(data_dir + case_study + ".struct_nr_lsr", 'w')

    #avg_clause_lsr = open(data_dir + case_study + ".struct_avg_clause_lsr", 'w')
    #ar_avg_clause_lsr = open(data_dir + case_study + ".struct_ar_avg_clause_lsr", 'w')
    #nr_avg_clause_lsr = open(data_dir + case_study + ".struct_nr_avg_clause_lsr", 'w')

    gini_normalized_picks = open(data_dir + case_study + ".struct_gini_normalized_picks", 'w')
    ar_gini_normalized_picks = open(data_dir + case_study + ".struct_ar_gini_normalized_picks", 'w')
    nr_gini_normalized_picks = open(data_dir + case_study + ".struct_nr_gini_normalized_picks", 'w')

    gini_normalized_clauses = open(data_dir + case_study + ".struct_gini_normalized_clauses", 'w')
    ar_gini_normalized_clauses = open(data_dir + case_study + ".struct_ar_gini_normalized_clauses", 'w')
    nr_gini_normalized_clauses = open(data_dir + case_study + ".struct_nr_gini_normalized_clauses", 'w')

    for (name,
         l, al, nl,    # lsr size
         #cl, acl, ncl, # avg clause lsr
         p, ap, np,    # picks
         c, ac, nc    # clauses picks
         ) in res:
        lsr.write(csv_line(name, l))
        ar_lsr.write(csv_line(name, al))
        nr_lsr.write(csv_line(name, nl))

        #avg_clause_lsr.write(csv_line(name, cl))
        #ar_avg_clause_lsr.write(csv_line(name, acl))
        #nr_avg_clause_lsr.write(csv_line(name, ncl))

        gini_normalized_picks.write(csv_line(name, p))
        ar_gini_normalized_picks.write(csv_line(name, ap))
        nr_gini_normalized_picks.write(csv_line(name, np))

        gini_normalized_clauses.write(csv_line(name, c))
        ar_gini_normalized_clauses.write(csv_line(name, ac))
        nr_gini_normalized_clauses.write(csv_line(name, nc))

    lsr.close()
    ar_lsr.close()
    nr_lsr.close()
    gini_normalized_picks.close()
    ar_gini_normalized_picks.close()
    nr_gini_normalized_picks.close()
    gini_normalized_clauses.close()
    ar_gini_normalized_clauses.close()
    nr_gini_normalized_clauses.close()




#######################################################################################################

def process_pocr_structure_output(base_dir, data_dir, case_study, simp=False):
    pocr_dir = base_dir + case_study + ("/simp/" if simp else "") + "/pocr_structure_logging/"
    print(pocr_dir)
    if not simp:
        print("Need simp for pocr, skipping")
        return

    expts = ["lsr", "bb_metrics", "cmty_loc", "all_decs"]  # TODO avg_lsr? would need to add a section below

    heuristics = ["maplesat_lrb_ar",
                  "maplesat_lrb_luby",
                  "maplesat_lrb_nr",
                  "maplesat_random_ar",
                  "maplesat_random_luby",
                  "maplesat_random_nr",
                  "maplesat_vsids_ar",
                  "maplesat_vsids_luby",
                  "maplesat_vsids_nr"]

    # lsr
    for h in heuristics:
        h_name = h[len("maplesat_"):]
        print(h_name)
        bench = Benchmark()
        f = bench.add_files(pocr_dir + h + "/lsr/", "data", ".lsr", allow_empty_files=False)
        if not f:
            print(case_study + " " + h + " lsr pocr failed")
            continue
        bench.compile_experiment()
        res = bench.run_experiment(process_simple_size_instance)
        print(res)
        with open(data_dir + case_study + ".pocr_" + h_name + "_lsr", 'w') as out:
            for n, d in res:
                out.write(csv_line(n, d))

    # all_decs
    for h in heuristics:
        h_name = h[len("maplesat_"):]
        print(h_name)
        bench = Benchmark()
        f = bench.add_files(pocr_dir + h + "/all_decs/", "data", ".all_decs", allow_empty_files=False)
        if not f:
            print(case_study + " " + h + " all_decs pocr failed")
            continue
        bench.compile_experiment()
        res = bench.run_experiment(process_simple_size_instance)
        print(res)
        with open(data_dir + case_study + ".pocr_" + h_name + "_all_decs", 'w') as out:
            for n, d in res:
                out.write(csv_line(n, d))

    # cmty_loc
    def cmty_h(file_map):
        print(file_map)

        fc = file_map['data']
        name = file_map['basename']
        with open(fc) as stream:
            for line in stream:
                if "GiniNormalizedPicks" in line:
                    p = str(line.strip().split()[1])
                elif "GiniNormalizedClauses" in line:
                    c = str(line.strip().split()[1])
        return name, p, c

    for h in heuristics:
        h_name = h[len("maplesat_"):]
        print(h_name)
        bench = Benchmark()
        f = bench.add_files(pocr_dir + h + "/cmty_loc/", "data", ".cmty_loc_maplesat", allow_empty_files=False)
        if not f:
            print(case_study + " " + h + " cmty_loc pocr failed")
            continue
        bench.compile_experiment()
        res = bench.run_experiment(cmty_h)
        print(res)
        with open(data_dir + case_study + ".pocr_" + h_name + "_gini_picks", 'w') as out:
            for n, p, c in res:
                out.write(csv_line(n, p))
        with open(data_dir + case_study + ".pocr_" + h_name + "_gini_clauses", 'w') as out:
            for n, p, c in res:
                out.write(csv_line(n, c))


    # bb_metrics
    def bb_h(file_map):
        flips, subsumed, conflicts, subsumed_raw, props = "", "", "", "", ""
        print(file_map)
        fc = file_map['data']
        name = file_map['basename']
        with open(fc) as stream:
            for line in stream:
                if "NormalizedBackboneFlips" in line:
                    flips = str(line.strip().split()[1])
                elif "NormalizedBackboneSubsumedClauses" in line:
                    subsumed = str(line.strip().split()[1])
                elif "BackboneSubsumedClauses" in line:
                    subsumed_raw = str(line.strip().split()[1])
                elif "Conflicts" in line:
                    conflicts = str(line.strip().split()[1])
                elif "Propagations" in line:
                    props = str(line.strip().split()[1])
        return name, flips, subsumed, conflicts, subsumed_raw, props

    for h in heuristics:
        h_name = h[len("maplesat_"):]
        print(h_name)
        bench = Benchmark()
        f = bench.add_files(pocr_dir + h + "/bb_metrics/", "data", ".bb_metrics_maplesat", allow_empty_files=False)
        if not f:
            print(case_study + " " + h + " bb_metrics pocr failed")
            continue
        bench.compile_experiment()
        res = bench.run_experiment(bb_h)
        print(res)
        with open(data_dir + case_study + ".pocr_" + h_name + "_bb_flips", 'w') as out:
            for n, f, s, c, r, p in res:
                out.write(csv_line(n, f))
        with open(data_dir + case_study + ".pocr_" + h_name + "_bb_subsumed", 'w') as out:
            for n, f, s, c, r, p in res:
                out.write(csv_line(n, s))
        with open(data_dir + case_study + ".pocr_" + h_name + "_bb_conflicts", 'w') as out:
            for n, f, s, c, r, p in res:
                out.write(csv_line(n, c))
        with open(data_dir + case_study + ".pocr_" + h_name + "_bb_subsumed_raw", 'w') as out:
            for n, f, s, c, r, p in res:
                out.write(csv_line(n, r))
        with open(data_dir + case_study + ".pocr_" + h_name + "_bb_propagations", 'w') as out:
            for n, f, s, c, r, p in res:
                out.write(csv_line(n, p))


    # times/conflicts
    def time_h(file_map):
        print(file_map)
        t, c, r, d = "", "", "", ""
        fc = file_map['data']
        name = file_map['basename']
        with open(fc) as stream:
            for line in stream:
                if line.startswith("CPU time"):
                    t = str(line.strip().split()[3])
                elif line.startswith("INDETERMINATE"):
                    r = "TIMEOUT"
                elif line.startswith("UNSAT"):
                    r = "UNSAT"
                elif line.startswith("SAT"):
                    r = "SAT"
                elif line.startswith("conflicts"):
                    c = line.strip().split()[2]
                elif line.startswith("decisions"):
                    d = line.strip().split()[2]
        return name, t, c, r, d

    for h in heuristics:
        h_name = h[len("maplesat_"):]
        print(h_name)
        bench = Benchmark()
        f = bench.add_files(pocr_dir + h + "/time/", "data", ".out", allow_empty_files=False)
        if not f:
            print(case_study + " " + h + " time pocr failed")
            continue
        bench.compile_experiment()
        res = bench.run_experiment(time_h)
        # print(res)
        with open(data_dir + case_study + ".pocr_" + h_name + "_time", 'w') as out:
            for n, t, c, r, d in res:
                out.write(csv_line(n, t))
        with open(data_dir + case_study + ".pocr_" + h_name + "_conflicts", 'w') as out:
            for n, t, c, r, d in res:
                out.write(csv_line(n, c))
        with open(data_dir + case_study + ".pocr_" + h_name + "_result", 'w') as out:
            for n, t, c, r, d in res:
                out.write(csv_line(n, r))
        with open(data_dir + case_study + ".pocr_" + h_name + "_decisions", 'w') as out:
            for n, t, c, r, d in res:
                out.write(csv_line(n, d))

        # lsr cmty spread metrics
        for h in heuristics:
            h_name = h[len("maplesat_"):]
            print(h_name)
            bench = Benchmark()
            f = bench.add_files(pocr_dir + h + "/lsr/", "lsr", ".lsr", allow_empty_files=False)
            f2 = bench.add_files(base_dir + case_study + ("/simp/" if simp else "") + "/cmty/", "cmty", ".cmty", allow_empty_files=False)
            if not f or not f2:
                print(case_study + " " + h + " time pocr failed")
                continue
            bench.compile_experiment()
            res = bench.run_experiment(process_cmtys_backdoor_overlaps_instance)

            overlap_file = open(data_dir + case_study + ".pocr_" + h_name + "_lsr_cmty_largest_ratio", 'w')
            lsr_spread_file = open(data_dir + case_study + ".pocr_" + h_name + "_lsr_cmty_spread", 'w')

            for (name, largest_ratio, cmty_spread) in res:
                overlap_file.write(csv_line(name, largest_ratio))
                lsr_spread_file.write(csv_line(name, cmty_spread))
            lsr_spread_file.close()
            overlap_file.close()


#######################################################################################################

def process_tw_instance(file_map):
    tw_file = file_map['tw']
    name = file_map['basename']

    with open(tw_file) as stream:
        line = stream.readline()
        arr = line.strip().split()
        low, up = int(arr[1]), int(arr[2])

    return name, low, up


def process_tw(base_dir, data_dir, case_study):
    """
    Produces CSVs for the treewidth size (.weak_size),
    the number of minimal weak backdoors (.num_min_weak),
    the number of variables in at least one backdoor (.num_vars_in_any_weak).
    """
    bench = Benchmark()
    vig_f = bench.add_files(base_dir + case_study + "/vig_tw/", "tw", ".vig_tw")
    vig_upper_file = open(data_dir + case_study + ".vig_upper_tw", 'w')
    vig_lower_file = open(data_dir + case_study + ".vig_lower_tw", 'w')
    bi_upper_file = open(data_dir + case_study + ".bipartite_upper_tw", 'w')
    bi_lower_file = open(data_dir + case_study + ".bipartite_lower_tw", 'w')

    if vig_f:
        print(case_study + " tw failed")
        bench.compile_experiment()
        res = bench.run_experiment(process_tw_instance)
        for (name, v_low, v_up) in res:
            vig_upper_file.write(csv_line(name, v_up))
            vig_lower_file.write(csv_line(name, v_low))

    bench = Benchmark()
    bi_f = bench.add_files(base_dir + case_study + "/bipartite_tw/", "tw", ".bipartite_tw")
    if bi_f:
        bench.compile_experiment()
        res = bench.run_experiment(process_tw_instance)
        for (name, b_low, b_up) in res:
            bi_upper_file.write(csv_line(name, b_up))
            bi_lower_file.write(csv_line(name, b_low))

    vig_upper_file.close()
    vig_lower_file.close()
    bi_upper_file.close()
    bi_lower_file.close()

#######################################################################################################

def process_mateescu_tw_instance(file_map):
    tw_file = file_map['tw']
    name = file_map['basename']
    up = ""

    with open(tw_file) as stream:
        for line in stream:
            arr = line.strip().split()
            if len(arr) > 0 and arr[0] == "Min":
                up = int(arr[4])
                break
    return name, up


def process_mateescu_tw(base_dir, data_dir, case_study, simp=False):
    bench = Benchmark()
    base = base_dir + case_study
    if simp:
        base += "/simp"
    tw_in = bench.add_files(base + "/tw_upper/", "tw", ".tw_upper")
    print("processing tw", case_study)
    if simp:
        tw_upper_file = open(data_dir + case_study + ".simp_tw_upper", 'w')
    else:
        tw_upper_file = open(data_dir + case_study + ".tw_upper", 'w')

    if tw_in:
        bench.compile_experiment()
        res = bench.run_experiment(process_mateescu_tw_instance)
        for (name, tw_up) in res:
            tw_upper_file.write(csv_line(name, tw_up))
    tw_upper_file.close()



#######################################################################################################


def process_cmtys_instance(file_map):
    c_file = file_map['cmty']
    q_file = file_map['q']
    name = file_map['basename']
    s = SatInstance()
    cmty_map = s.read_cmtys(c_file)
    q = s.read_q(q_file)
    cmtys = set([])
    for v in cmty_map.values():
        cmtys.add(v)
    return name, q, len(cmtys)


def process_cmtys(base_dir, data_dir, case_study, simp=False):
    """
    Produces CSVs for the weak backdoor size (.weak_size),
    the number of minimal weak backdoors (.num_min_weak),
    the number of variables in at least one backdoor (.num_vars_in_any_weak).
    """
    bench = Benchmark()
    base = base_dir + case_study
    if simp:
        base += "/simp"
        ignore_file = data_dir + case_study + ".simp_q"
    else:
        ignore_file = data_dir + case_study + ".q"

    f = bench.add_files(base + "/cmty/", "q", ".q", ignore_found_file=ignore_file)
    f2 = bench.add_files(base + "/cmty/", "cmty", ".cmty")
    if not f or not f2:
        print(case_study +  " cmtys failed, simp: ", simp)
        return None
    bench.compile_experiment()
    res = bench.run_experiment(process_cmtys_instance)

    if simp:
        q_file = open(data_dir + case_study + ".simp_q", 'a')
        cmty_file = open(data_dir + case_study + ".simp_num_cmtys", 'a')
    else:
        q_file = open(data_dir + case_study + ".q", 'a')
        cmty_file = open(data_dir + case_study + ".num_cmtys", 'a')

    for (name, q, c) in res:
        q_file.write(csv_line(name, q))
        cmty_file.write(csv_line(name, c))

    q_file.close()
    cmty_file.close()

#######################################################################################################


def process_cmtys_instance(file_map):
    c_file = file_map['cmty']
    q_file = file_map['q']
    name = file_map['basename']
    s = SatInstance()
    cmty_map = s.read_cmtys(c_file)
    q = s.read_q(q_file)
    cmtys = set([])
    for v in cmty_map.values():
        cmtys.add(v)
    return name, q, len(cmtys)


def process_cmtys(base_dir, data_dir, case_study, simp=False):
    """
    Produces CSVs for the weak backdoor size (.weak_size),
    the number of minimal weak backdoors (.num_min_weak),
    the number of variables in at least one backdoor (.num_vars_in_any_weak).
    """
    bench = Benchmark()
    base = base_dir + case_study
    if simp:
        base += "/simp"
        ignore_file = data_dir + case_study + ".simp_q"
    else:
        ignore_file = data_dir + case_study + ".q"

    f = bench.add_files(base + "/cmty/", "q", ".q", ignore_found_file=ignore_file)
    f2 = bench.add_files(base + "/cmty/", "cmty", ".cmty")
    if not f or not f2:
        print(case_study +  " cmtys failed, simp: ", simp)
        return None
    bench.compile_experiment()
    res = bench.run_experiment(process_cmtys_instance)

    if simp:
        q_file = open(data_dir + case_study + ".simp_q", 'a')
        cmty_file = open(data_dir + case_study + ".simp_num_cmtys", 'a')
    else:
        q_file = open(data_dir + case_study + ".q", 'a')
        cmty_file = open(data_dir + case_study + ".num_cmtys", 'a')

    for (name, q, c) in res:
        q_file.write(csv_line(name, q))
        cmty_file.write(csv_line(name, c))

    q_file.close()
    cmty_file.close()
#######################################################################################################


def process_fractal_dim_alpha_instance(file_map):
    fractal_dim_vig_file = file_map['fractal_dim_vig']
    fractal_dim_cvig_file = file_map['fractal_dim_cvig']
    alpha_var_file = file_map['alpha_var']
    name = file_map['basename']
    with open(fractal_dim_vig_file) as stream:
        l = stream.readline()
        assert l.startswith('dimension')
        v = l.split()[2]
    with open(fractal_dim_cvig_file) as stream:
        l = stream.readline()
        assert l.startswith('dimension')
        c = l.split()[2]
    with open(alpha_var_file) as stream:
        stream.readline()
        l = stream.readline()
        assert l.startswith('alpha')
        a = l.split()[2]
    return name, v, c, a


def process_fractal_dim_alpha(base_dir, data_dir, case_study, simp=False):
    """
    Produces CSVs for the weak backdoor size (.weak_size),
    the number of minimal weak backdoors (.num_min_weak),
    the number of variables in at least one backdoor (.num_vars_in_any_weak).
    """
    bench = Benchmark()
    base = base_dir + case_study
    if simp:
        base += "/simp"
    else:
        print("Simp required for fractal dim alpha")
        return None

    f = bench.add_files(base + "/fractal_dim_vig/", "fractal_dim_vig", ".fractal_dim_vig", ignore_found_file=None)
    f2 = bench.add_files(base + "/fractal_dim_cvig/", "fractal_dim_cvig", ".fractal_dim_cvig", ignore_found_file=None)
    f3 = bench.add_files(base + "/alpha_var/", "alpha_var", ".alpha_var", ignore_found_file=None)
    if not f or not f2 or not f3:
        print(case_study +  " fractal_dim_alpha failed, simp: ", simp)
        return None
    bench.compile_experiment()
    res = bench.run_experiment(process_fractal_dim_alpha_instance)

    if simp:
        vig_dim_file = open(data_dir + case_study + ".simp_fractal_dim_vig", 'w')
        cvig_dim_file = open(data_dir + case_study + ".simp_fractal_dim_cvig", 'w')
        alpha_var_file = open(data_dir + case_study + ".simp_alpha_var", 'w')
    else:
        sys.exit("todo")

    for (name, v, c, a) in res:
        vig_dim_file.write(csv_line(name, v))
        cvig_dim_file.write(csv_line(name, c))
        alpha_var_file.write(csv_line(name, a))

    vig_dim_file.close()
    cvig_dim_file.close()
    alpha_var_file.close()



#######################################################################################################

def process_cmtys_backdoor_overlaps_instance(file_map):
    # returns name, ratio of num bds in the "best cmty" over total bds, and num cmtys that the bd is spread across
    c_file = file_map['cmty']
    name = file_map['basename']
    s = SatInstance()
    cmty_map = s.read_cmtys(c_file)
    # cmtys are zero based
    # bd is zero based (at least for lsr)
    bd = s.read_one_int_per_line_file(file_map['lsr'])

    # bucketize the bd
    buckets = {}
    for v in bd:
        try:
            buckets[cmty_map[v]] = buckets.get(cmty_map[v], 0) + 1
        except ValueError:
            pass
    denom = len(bd)
    # print(max(buckets.values()) / denom, len(buckets.items()))
    return name, max(buckets.values()) / denom, len(buckets.items())


def process_cmtys_backdoor_overlaps(base_dir, data_dir, case_study, simp=False):
    bench = Benchmark()
    if simp:
        simp_str = "/simp/"
    else:
        simp_str = ""
    f = bench.add_files(base_dir + case_study + simp_str + "/cmty/", "q", ".q")
    f2 = bench.add_files(base_dir + case_study + simp_str + "/cmty/", "cmty", ".cmty")
    # get a backdoor
    f3 = bench.add_files(base_dir + case_study + simp_str + "/" + CCMIN + "_confl_side_lsr/", "lsr", ".lsr")
    if not f or not f2 or not f3:
        print(case_study + str(simp) + " backdoors cmtys failed")
        print(base_dir + case_study + ("/simp/" if simp else "") + "/" + CCMIN + "_confl_side_lsr/")
        return None
    bench.compile_experiment()
    res = bench.run_experiment(process_cmtys_backdoor_overlaps_instance)

    overlap_file = open(data_dir + case_study + ".lsr_cmty_largest_ratio", 'w')
    lsr_spread_file = open(data_dir + case_study + ".lsr_cmty_spread", 'w')

    for (name, largest_ratio, cmty_spread) in res:
        overlap_file.write(csv_line(name, largest_ratio))
        lsr_spread_file.write(csv_line(name, cmty_spread))
    lsr_spread_file.close()
    overlap_file.close()


#######################################################################################################


def process_weak_backdoors_instance(file_map):
    w = file_map['weak']
    name = file_map['basename']
    stream = open(w)
    seen_vars = set()
    num_min_weak_backdoors = 0
    weak_backdoor_size = 0
    for line in stream:
        # each line contains a weak backdoor
        arr = line.strip().split()
        weak_backdoor_size = len(arr)
        num_min_weak_backdoors += 1
        for v in arr:
            seen_vars.add(v)
    num_vars_in_any_weak = len(seen_vars)
    return name, weak_backdoor_size, num_min_weak_backdoors, num_vars_in_any_weak


def process_weak_backdoors(base_dir, data_dir, case_study, simp = False):
    """
    Produces CSVs for the weak backdoor size (.weak_size),
    the number of minimal weak backdoors (.num_min_weak),
    the number of variables in at least one backdoor (.num_vars_in_any_weak).
    """
    if simp:
        base = base_dir + case_study + "/simp/weak/"
        suffix = ".simp_"
    else:
        base = base_dir + case_study + "/weak/"
        suffix = "."
    bench = Benchmark()
    f = bench.add_files(base, "weak", ".weak", ignore_found_file=data_dir + case_study + suffix + "weak_size")
    if not f:
        print(case_study + " weak backdoors failed")
        return None
    bench.compile_experiment()
    res = bench.run_experiment(process_weak_backdoors_instance)

    weak_bds_file = open(data_dir + case_study + suffix + "weak_size", 'a')
    num_min_weak_bds_file = open(data_dir + case_study + suffix + "num_min_weak", 'a')
    num_vars_in_any_weak_file = open(data_dir + case_study + suffix + "num_vars_in_any_weak", 'a')

    for (name, weak_backdoor_size, num_min_weak_backdoors, num_vars_in_any_weak) in res:
        weak_bds_file.write(csv_line(name, weak_backdoor_size))
        num_min_weak_bds_file.write(csv_line(name, num_min_weak_backdoors))
        num_vars_in_any_weak_file.write(csv_line(name, num_vars_in_any_weak))

    weak_bds_file.close()
    num_min_weak_bds_file.close()
    num_vars_in_any_weak_file.close()


#######################################################################################################

def process_num_decisions_per_clause_instance(file_map):
    numdecs_file = file_map['numdecs']
    name = file_map['basename']
    print(name)
    total_sum, count = 0, 0
    with open(numdecs_file) as stream:
        for line in stream:
            num = int(line.strip())
            count += 1
            total_sum += num
    return name, total_sum / count


def process_num_decisions_per_clause(base_dir, data_dir, case_study):
    """
    Produces CSV for the avg num decisions needed for each clause (.numdecs),
    """

    bench = Benchmark()
    f = bench.add_files(base_dir + case_study + "/numdecs/", "numdecs", ".numdecs", allow_empty_files=False)
    if not f:
        print("Numdecs:", case_study, "failed")
        return None
    bench.compile_experiment()
    res = bench.run_experiment(process_num_decisions_per_clause_instance)
    num_decs_file = open(data_dir + case_study + ".numdecs", 'w')
    for (name, avg_num_decs) in res:
        num_decs_file.write(csv_line(name, avg_num_decs))
    num_decs_file.close()


#######################################################################################################

def process_backbones_instance(file_map):
    numdecs_file = file_map['data']
    name = file_map['basename']
    print(name)
    backbones = ""
    finished = False
    with open(numdecs_file) as stream:
        for line in stream:
            arr = line.strip().split()
            if "unsatisfiable" in line:
                return name, ""
            if len(arr) >= 2 and arr[0] == 'i' and arr[1] == 'complete':
                finished = True
            if len(arr) >= 3 and arr[1] == "backbone" and arr[2] == "size:":
                backbones = arr[3]
    if not finished:
        return name, ""
    else:
        return name, backbones

def process_backbones(base_dir, data_dir, case_study, simp = False):
    """
    Produces CSV for the backbone size (.backbone_size),
    """
    if simp:
        base = base_dir + case_study + "/simp/"
        suffix = ".simp_"
    else:
        base = base_dir + case_study
        suffix = "."

    bench = Benchmark()
    f = bench.add_files(base + "/backbones/", "data", ".sharcnet", allow_empty_files=True,
                        ignore_found_file=data_dir + case_study + suffix + "backbones")
    if not f:
        print("Backbones:", case_study, "failed")
        return None
    bench.compile_experiment()
    res = bench.run_experiment(process_backbones_instance)
    backbones_file = open(data_dir + case_study + suffix + "backbones", 'a')
    for (name, backbone_size) in res:
        backbones_file.write(csv_line(name, backbone_size))
    backbones_file.close()


#######################################################################################################

def process_minlsr_backdoors_instance(file_map):
    minlsr_file = file_map['data']
    name = file_map['basename']
    # print(minlsr_file)
    with open(minlsr_file) as stream:
        for line in stream:
            if "Best solution" in line:
                return name, int(line.strip().split()[3])
    return None


def process_minlsr_backdoors(base_dir, data_dir, case_study):
    """
    Produces CSV for the lsr size (.lsr_backdoor_size),
    """
    bench = Benchmark()
    f = bench.add_files(base_dir + case_study + "/minlaser/", "data", ".minlaser", allow_empty_files=False)
    if not f:
        print("Min LSR " + case_study + " failed")
        return None
    bench.compile_experiment()
    res = bench.run_experiment(process_minlsr_backdoors_instance)
    lsr_file = open(data_dir + case_study + ".minlsr_size", 'w')
    for (name, lsr_size) in res:
        lsr_file.write(csv_line(name, lsr_size))
    lsr_file.close()


#######################################################################################################

def process_lit_contradictions_instance(file_map):
    lits_file = file_map['data']
    name = file_map['basename']
    print(name)
    num_lines = sum(1 for _ in open(lits_file))
    print(num_lines)
    cur_max_line = int(num_lines / 20)
    incr = cur_max_line
    cur_total = 0
    cur_nonzero_total = 0
    cur_line = 0
    cur_zeros = 0
    cur_non_zeroes = 0
    big_total = 0
    with open(lits_file) as stream:
        try:
            for line in stream:
                if cur_line == cur_max_line:
                    print(cur_max_line, cur_total / incr, cur_nonzero_total / cur_non_zeroes, cur_zeros)
                    cur_total = 0
                    cur_zeros = 0
                    cur_non_zeroes = 0
                    cur_nonzero_total = 0
                    cur_max_line += incr
                    if cur_max_line + incr >= num_lines:
                        cur_max_line = num_lines - 1

                arr = [int(i) for i in line.strip().split()]
                decisions, contras = arr[0], arr[1]
                cur_total += contras / decisions
                big_total += contras / decisions
                if contras != 0:
                    cur_nonzero_total += contras / decisions
                    cur_non_zeroes += 1
                else:
                    cur_zeros += 1
                cur_line += 1
        except Exception as e:
            print(e)
            return None
    return name, big_total / num_lines


def process_lit_contradictions(base_dir, data_dir, case_study):
    """
    Produces CSV for the lsr size (.lsr_backdoor_size),
    """
    bench = Benchmark()
    f = bench.add_files(base_dir + case_study + "/litlaser/", "data", ".lit_contradictions", allow_empty_files=False)
    if not f:
        print("Lit contradictions " + case_study + " failed")
        return None
    bench.compile_experiment()
    res = bench.run_experiment(process_lit_contradictions_instance)
    lsr_file = open(data_dir + case_study + ".avg_lit_contradictions", 'w')
    for (name, avg_lit) in res:
        lsr_file.write(csv_line(name, avg_lit))
    lsr_file.close()

###########################################################################################


def process_lsr_backdoors(base_dir, data_dir, case_study, simp=False):
    """
    Produces CSV for the lsr size (.lsr_backdoor_size),
    """
    if simp:
        base = base_dir + case_study + "/simp/"
        suffix = ".simp_lsr_size"
    else:
        suffix = ".lsr_size"
        base = base_dir + case_study

    bench = Benchmark()
    print("dir", base + "/" + CCMIN + "_confl_side_lsr/")
    f = bench.add_files(base + "/" + CCMIN + "_confl_side_lsr/", "data", ".lsr", allow_empty_files=False,
                        ignore_found_file=None) #data_dir + case_study + ".lsr_size")
    if not f:
        print("LSR " + case_study + " failed")
        return None
    bench.compile_experiment()
    res = bench.run_experiment(process_simple_size_instance)
    lsr_file = open(data_dir + case_study + suffix, 'a')
    for (name, lsr_size) in res:
        lsr_file.write(csv_line(name, lsr_size))
    lsr_file.close()

###########################################################################################


def process_lsr_sat_min(base_dir, data_dir, case_study, simp=False):
    """
    Produces CSV for the lsr size (.lsr_backdoor_size),
    """
    if simp:
        base = base_dir + case_study + "/simp/"
        suffix = ".simp_sat_min_lsr"
    else:
        suffix = ".sat_min_lsr"
        base = base_dir + case_study

    bench = Benchmark()
    print("dir", base + "/" + CCMIN + "_sat_min/")
    f = bench.add_files(base + "/" + CCMIN + "_sat_min/", "data", ".sat_min", allow_empty_files=False,
                        ignore_found_file=None) #data_dir + case_study + ".lsr_size")
    if not f:
        print("LSR sat_min" + case_study + " failed")
        return None
    bench.compile_experiment()
    res = bench.run_experiment(process_simple_size_instance)
    lsr_file = open(data_dir + case_study + suffix, 'a')
    for (name, lsr_size) in res:
        lsr_file.write(csv_line(name, lsr_size))
    lsr_file.close()

###########################################################################################


def process_all_decs(base_dir, data_dir, case_study, simp=False):
    """
    Produces CSV for the lsr size (.lsr_backdoor_size),
    """
    if simp:
        base = base_dir + case_study + "/simp/"
        suffix = ".simp_all_decs"
    else:
        suffix = ".all_decs"
        base = base_dir + case_study

    bench = Benchmark()
    print("running all_decs " + case_study)
    f = bench.add_files(base + "/" + CCMIN + "_all_decs/", "data", ".all_decs", allow_empty_files=False,
                        ignore_found_file=None) #data_dir + case_study + ".lsr_size")
    if not f:
        print("All decs " + case_study + " failed")
        return None
    bench.compile_experiment()
    res = bench.run_experiment(process_simple_size_instance)
    all_decs_file = open(data_dir + case_study + suffix, 'a')
    for (name, all_decs_size) in res:
        all_decs_file.write(csv_line(name, all_decs_size))
    all_decs_file.close()


###########################################################################################

def process_lsr_all_decs_overlap_instance(file_map):
    name = file_map['basename']
    lsr_file = file_map['lsr']
    all_decs_file = file_map['all_decs']
    lsr = SatInstance.read_one_int_per_line_file(lsr_file)
    all_decs = SatInstance.read_one_int_per_line_file(all_decs_file)
    return name, len(set(lsr).union(all_decs)), len(set(lsr).intersection(all_decs))


def process_lsr_all_decs_overlap(base_dir, data_dir, case_study, simp=False):
    """
    Produces CSV for the lsr size (.lsr_backdoor_size),
    """
    if simp:
        base = base_dir + case_study + "/simp/"
        union_file_suffix = ".simp_lsr_all_decs_union"
        inter_file_suffix = ".simp_lsr_all_decs_inter"
    else:
        union_file_suffix = ".lsr_all_decs_union"
        inter_file_suffix = ".lsr_all_decs_inter"
        base = base_dir + case_study

    bench = Benchmark()
    print("dir overlap ", base + "/" + CCMIN + "_all_decs/")
    f = bench.add_files(base + "/" + CCMIN + "_all_decs/", "all_decs", ".all_decs", allow_empty_files=False,
                        ignore_found_file=None) #data_dir + case_study + ".lsr_size")
    f2 = bench.add_files(base + "/" + CCMIN + "_confl_side_lsr/", "lsr", ".lsr", allow_empty_files=False,
                         ignore_found_file=None)  # data_dir + case_study + ".lsr_size")
    if not f or not f2:
        print("All decs lsr overlap " + case_study + " failed")
        return None
    bench.compile_experiment()
    res = bench.run_experiment(process_lsr_all_decs_overlap_instance)

    union_file = open(data_dir + case_study + union_file_suffix, 'a')
    inter_file = open(data_dir + case_study + inter_file_suffix, 'a')
    for (name, union, inter) in res:
        union_file.write(csv_line(name, union))
        inter_file.write(csv_line(name, inter))
    union_file.close()
    inter_file.close()


#######################################################################################################


def process_sat_header_instance(file_map):
    c = file_map['cnf']
    name = file_map['basename']
    s = SatInstance(c)
    s.read_cnf()
    # workaround for mathcheck header error
    if name.startswith("hamilton5."):
        s.number_of_clauses = str(int(name.split(".")[1]) - 1)

    return name, s.number_of_variables, s.number_of_clauses


def process_sat_header(base_dir, data_dir, case_study, simp=False):
    """
    Produces CSV for the vars and clauses data (.num_vars and .num_clauses),
    """
    bench = Benchmark()
    print(case_study)

    base = base_dir + case_study
    if simp:
        base += "/simp"
        suffix = "simp_"
        ignore_file = data_dir + case_study + ".simp_num_vars"
    else:
        ignore_file = data_dir + case_study + ".num_vars"
        suffix = ""

    f = bench.add_files(base + "/cnf/", "cnf", ".cnf", allow_empty_files=False, ignore_found_file=ignore_file)

    if not f:
        print("Sat headers -- " + case_study + " failed, simp:", simp)
        return None
    bench.compile_experiment()
    res = bench.run_experiment(process_sat_header_instance)


    vars_file = open(data_dir + case_study + "." + suffix + "num_vars", 'a')
    clauses_file = open(data_dir + case_study + "." + suffix + "num_clauses", 'a')
    for (name, total_vars, total_clauses) in res:
        vars_file.write(csv_line(name, total_vars))
        clauses_file.write(csv_line(name, total_clauses))
    print(data_dir + case_study + ".num_vars")
    vars_file.close()
    clauses_file.close()


#######################################################################################################

def process_lsr_heuristics_instance(file_map):
    f = file_map['heuristic']
    name = file_map['basename']
    timeout = 5000
    stream = open(f)
    found = False
    time = None
    for line in stream:
        if "CPU time" in line:
            found = True
            time = float(line.split()[3]) if float(line.split()[3]) < timeout else timeout
            break
    # this might happen in case of memlimit, etc.
    if not found:
        return name, timeout
    return name, time


def process_lsr_heuristics(base_dir, data_dir, case_study):
    bench = Benchmark()
    print(case_study)
    heuristics = [("lsr_deletion3_increase1_bump1", "lsr_311")]
    for (h, s) in heuristics:
        f = bench.add_files(base_dir + case_study + "/" + h, "heuristic", ".sharcnet", allow_empty_files=False)
        if not f:
            print("LSR heuristic -- " + case_study + " failed")
            continue
        bench.compile_experiment()
        res = bench.run_experiment(process_lsr_heuristics_instance)

        out_file = open(data_dir + case_study + "." + s, 'a')
        print(f, res)
        for (name, time) in res:
            out_file.write(csv_line(name, time))
        out_file.close()

#######################################################################################################

def process_solver_output_instance(file_map):
    # TODO for maplecomsps should actually grab the c CPU line, the one we grab rounds down
    f = file_map['out_file']
    name = file_map['basename']
    result = ""
    timeout = 5000
    scaling_timeout = 1200
    agile_timeout = 60

    simplification_time = 0
    # print(f)
    stream = open(f)
    found = False
    time = None
    try:
        for line in stream:
            if line.startswith("WARNING:"):
                continue
            if "Simplification time" in line:
                arr = line.split()
                simplification_time = arr[4]
            if "cpu time" in line.lower():
                found = True
                if line.split()[0] == "CPU" and ("cls_partition" in f or "cmty_partition" in f):
                    index = 2
                    time = float(line.split()[index]) if float(line.split()[index]) < timeout else timeout
                elif line.split()[0] == "CPU":
                    index = 3
                    time = float(line.split()[index]) if float(line.split()[index]) < timeout else timeout
                elif line.split()[0] == "cpu":
                    index = 2
                    t = line.split()[index]
                    if t.endswith("s"):
                        time = float(t.rstrip("s"))
                    elif t.endswith("h"):
                        time = float(t.rstrip("h")) * 3600
                    elif t == "0":
                        time = 0
                    time = time if time < timeout else timeout
                elif line.split()[0] == "c":
                    time = float(line.split()[4]) if float(line.split()[4]) < timeout else timeout
            if line.startswith("s SAT") or line.startswith("SAT"):
                result = "SAT"
            elif line.startswith("s UNSAT") or line.startswith("UNSAT"):
                result = "UNSAT"
        # this might happen in case of memlimit, maplecomsps, etc.
        if not found:
            if "agile" in f and "maplecomsps" in f:
                return name, agile_timeout, "TIMEOUT", 0
            elif "scaling" in f and "maplecomsps" in f:
                return name, scaling_timeout, "TIMEOUT", 0
            elif "agile" not in f and "maplecomsps" in f:
                return name, timeout, "TIMEOUT", 0
        if ("agile" in f and float(time) >= agile_timeout) or ("agile" not in f and float(time >= timeout)):
            simplification_time = 0
        return name, time, result, simplification_time
    except UnicodeDecodeError:
        return None

def process_solver_output(base_dir, data_dir, case_study, simp = False):
    if simp:
        base = base_dir + case_study + "/simp/"
        suffix = ".simp_"
    else:
        base = base_dir + case_study
        suffix = "."

    bench = Benchmark()
    #print("Solver output:", case_study)
    result_map = {}
    solvers = ["maplecomsps"]
    #print("WARNING -- changed process_solver_output")

    for s in solvers:
        '''
        if s == "maplecomsps":
            allow_empty = True
            if simp:
                infile_suffix = s
            else:
                infile_suffix = "sharcnet"
        else:
            allow_empty = False
            infile_suffix = ".sharcnet"
        '''
        infile_suffix = s
        if s == "maplecomsps":
            allow_empty = True
        else:
            allow_empty = False

        f = bench.add_files(base + "/" + s, "out_file", "." + infile_suffix, allow_empty_files=allow_empty,
                            ignore_found_file=data_dir + case_study + suffix + s)
        #print(base_dir + case_study + "/" + s)
        #f = bench.add_files(base_dir + case_study + "/" + s, "out_file", "." + s, allow_empty_files=allow_empty,
        #                    ignore_found_file=data_dir + case_study + "." + s)
        if not f:
            print("Solver output -- " + s + " -- " + case_study + " failed")
            continue
        bench.compile_experiment()
        res = bench.run_experiment(process_solver_output_instance)

        out_file = open(data_dir + case_study + suffix + s, 'a')
        out_file_pp = open(data_dir + case_study + suffix + s + "_pp", 'a')

        for (name, time, result, simp_time) in res:
            out_file.write(csv_line(name, time))
            out_file_pp.write(csv_line(name, simp_time))

            if result == "SAT" or result == "UNSAT":
                result_map[name] = result
        out_file.close()
        out_file_pp.close()
    prev_results = {}
    try:
        with open(data_dir + case_study + ".result", 'r') as stream:
            for line in stream:
                arr = line.strip().split(",")
                name, res = arr[0], arr[1]
                prev_results[name] = res
    except:
        pass
    for name, result in result_map.items():
        prev_results[name] = result
    out_file = open(data_dir + case_study + ".result", 'w')
    for name, result in prev_results.items():
        out_file.write(csv_line(name, result))
    out_file.close()


########################################################################################

def process_maplesat_restart_output_instance(file_map):
    f = file_map['maplesat']
    fa = file_map['maplesat_ar']
    fn = file_map['maplesat_nr']
    name = file_map['basename']
    result = ""
    timeout = 5000
    agile_timeout = 60
    if "agile" in f:
        timeout = agile_timeout

    time = ""
    conflicts = ""
    time_ar = ""
    conflicts_ar = ""
    time_nr = ""
    conflicts_nr = ""

    with open(f) as stream:
        for line in stream:
            if "cpu time" in line.lower():
                time = float(line.split()[3])
                time = time if time < timeout else timeout
                time = str(time)
            if "conflicts" in line:
                conflicts = line.split()[2]
    with open(fa) as stream:
        for line in stream:
            if "cpu time" in line.lower():
                time_ar = float(line.split()[3])
                time_ar = time_ar if time_ar < timeout else timeout
                time_ar = str(time_ar)
            if "conflicts" in line:
                conflicts_ar = line.split()[2]
    with open(fn) as stream:
        for line in stream:
            if "cpu time" in line.lower():
                time_nr = float(line.split()[3])
                time_nr = time_nr if time_nr < timeout else timeout
                time_nr = str(time_nr)
            if "conflicts" in line:
                conflicts_nr = line.split()[2]
    return name, time, conflicts, time_ar, conflicts_ar, time_nr, conflicts_nr

def process_maplesat_restart_output(base_dir, data_dir, case_study, simp = False):
    if simp:
        base = base_dir + case_study + "/simp/"
        suffix = ".simp_"
    else:
        base = base_dir + case_study
        suffix = "."

    bench = Benchmark()
    print("Solver output:", case_study)
    result_map = {}
    solvers = ["maplesat", "maplesat_ar", "maplesat_nr"]
    infile_suffix = ".result"
    allow_empty = False

    for s in solvers:
        f = bench.add_files(base + "/" + s, s, infile_suffix, allow_empty_files=allow_empty,
                            ignore_found_file=None)
        #print(base_dir + case_study + "/" + s)
        #f = bench.add_files(base_dir + case_study + "/" + s, "out_file", "." + s, allow_empty_files=allow_empty,
        #                    ignore_found_file=data_dir + case_study + "." + s)
        if not f:
            print("Maplesat restart output -- " + s + " -- " + case_study + " failed")
            return
    bench.compile_experiment()
    res = bench.run_experiment(process_maplesat_restart_output_instance)
    if not res:
        print("restart expt failed")
        return
    with open(data_dir + case_study + suffix + "maplesat_time", 'w') as tf, \
            open(data_dir + case_study + suffix + "maplesat_conflicts", 'w') as cf, \
            open(data_dir + case_study + suffix + "maplesat_ar_time", 'w') as taf, \
            open(data_dir + case_study + suffix + "maplesat_ar_conflicts", 'w') as caf, \
            open(data_dir + case_study + suffix + "maplesat_nr_time", 'w') as tnf, \
            open(data_dir + case_study + suffix + "maplesat_nr_conflicts", 'w') as cnf:
        for name, time, conflicts, time_ar, conflicts_ar, time_nr, conflicts_nr in res:
            tf.write(csv_line(name, time))
            cf.write(csv_line(name, conflicts))
            taf.write(csv_line(name, time_ar))
            caf.write(csv_line(name, conflicts_ar))
            tnf.write(csv_line(name, time_nr))
            cnf.write(csv_line(name, conflicts_nr))


#######################################################################################################


def process_bv_spread_file_instance(file_map):
    f = file_map['out_file']
    name = file_map['basename']
    overall = None
    avg_non_singleton = 0  # avg spread of bit-vectors of size >= 2, NOT NORMALIZED FOR SIZE!!!
    avg_non_singleton_size = 0
    count = 0
    result = ""
    timeout = 5000
    agile_timeout = 60
    with open(f) as stream:
        for line in stream:
            arr = line.strip().split()
            n, bits, cmtys = arr[0], int(arr[1]), int(arr[2])
            if arr[0] == "Overall":
                overall = cmtys/bits
            elif bits > 1:
                count += 1
                avg_non_singleton += cmtys
                avg_non_singleton_size += bits

    return name, overall, (avg_non_singleton / count if count != 0 else 0), (avg_non_singleton_size / count if count != 0 else 0)


def process_bv_spread_file(base_dir, data_dir, case_study, simp = False):
    if simp:
        print("Simp solver output not supported")
        return
    bench = Benchmark()
    result_map = {}
    f = bench.add_files(base_dir + case_study + "/" + "spread/", "out_file", ".spread", allow_empty_files=False,
                        ignore_found_file=data_dir + case_study + ".spread")
    if not f:
        print("Spread output -- " + case_study + " failed")
        return
    bench.compile_experiment()
    res = bench.run_experiment(process_bv_spread_file_instance)

    out_file = open(data_dir + case_study + ".cmtys_with_at_least_one_bit", 'a')
    out_file2 = open(data_dir + case_study + ".avg_num_cmtys_per_non_singleton_bv", 'a')
    out_file3 = open(data_dir + case_study + ".avg_size_of_non_singleton_bv", 'a')
    for (name, cmty_with_at_least_one_bit, avg_non_singleton, avg_size_of_non_singleton_bv) in res:
        out_file.write(csv_line(name, cmty_with_at_least_one_bit))
        out_file2.write(csv_line(name, avg_non_singleton))
        out_file3.write(csv_line(name, avg_size_of_non_singleton_bv))
    out_file.close()
    out_file2.close()


#######################################################################################################


def process_bv_bits_bridge_instance(file_map):
    f = file_map['bits_bridge_file']
    name = file_map['basename']
    bits_bridge, bridge_bits, tseitin_bridge, bridge_tseitin = None, None, None, None
    with open(f) as stream:
        for line in stream:
            arr = line.strip().split()
            if arr[0] == "Ratios":
                bits_bridge = float(arr[1])
                bridge_bits = float(arr[2])
                tseitin_bridge = float(arr[3])
                bridge_tseitin = float(arr[4])
    if bits_bridge:
        return name, bits_bridge, bridge_bits, tseitin_bridge, bridge_tseitin
    else:
        return None


def process_bv_bits_bridge(base_dir, data_dir, case_study, simp = False):
    if simp:
        print("Simp solver output not supported")
        return
    bench = Benchmark()
    f = bench.add_files(base_dir + case_study + "/" + "bits_bridge_data/", "bits_bridge_file", ".bits_bridge_data", allow_empty_files=False,
                        ignore_found_file=data_dir + case_study + ".bits_bridge_data")
    if not f:
        print("Spread output -- " + case_study + " failed")
        return
    bench.compile_experiment()
    res = bench.run_experiment(process_bv_bits_bridge_instance)

    bits_that_are_bridge_file = open(data_dir + case_study + ".bits_that_are_bridge", 'a')
    bridge_that_are_bits_file = open(data_dir + case_study + ".bridge_that_are_bits", 'a')
    tseitin_that_are_bridge_file = open(data_dir + case_study + ".tseitin_that_are_bridge", 'a')
    bridge_that_are_tseitin_file = open(data_dir + case_study + ".bridge_that_are_tseitin", 'a')
    for (name, bits_bridge, bridge_bits, tseitin_bridge, bridge_tseitin) in res:
        bits_that_are_bridge_file.write(csv_line(name, bits_bridge))
        bridge_that_are_bits_file.write(csv_line(name, bridge_bits))
        tseitin_that_are_bridge_file.write(csv_line(name, tseitin_bridge))
        bridge_that_are_tseitin_file.write(csv_line(name, bridge_tseitin))
    bits_that_are_bridge_file.close()
    bridge_that_are_bits_file.close()
    tseitin_that_are_bridge_file.close()
    bridge_that_are_tseitin_file.close()


#######################################################################################################


def process_bv_louvain_orderings_instance(file_map):
    leaf_file = file_map['leaf_file']
    root_file = file_map['root_file']
    standard_file = file_map['standard_file']
    name = file_map['basename']
    leaf_time, root_time, standard_time = 0, 0, 0
    with open(leaf_file) as stream:
        for line in stream:
            arr = line.strip().split()
            if arr[0] == 'time':
                leaf_time = float(arr[2])
    with open(root_file) as stream:
        for line in stream:
            arr = line.strip().split()
            if arr[0] == 'time':
                root_time = float(arr[2])
    with open(standard_file) as stream:
        for line in stream:
            arr = line.strip().split()
            if arr[0] == 'time':
                standard_time = float(arr[2])
    return name, leaf_time, root_time, standard_time


def process_bv_louvain_orderings(base_dir, data_dir, case_study, simp = False):
    #TODO this ignores any sat instance that doesnt have data on all 3 orderings
    if simp:
        print("Simp solver output not supported")
        return
    bench = Benchmark()
    result_map = {}
    f = bench.add_files(base_dir + case_study + "/" + "louvain_orderings/", "root_file", ".root_out", allow_empty_files=False,
                        ignore_found_file=data_dir + case_study + ".root_louvain_time")
    f2 = bench.add_files(base_dir + case_study + "/" + "louvain_orderings/", "leaf_file", ".leaf_out",
                        allow_empty_files=False,
                        ignore_found_file=data_dir + case_study + ".leaf_louvain_time")
    f3 = bench.add_files(base_dir + case_study + "/" + "louvain_orderings/", "standard_file", ".standard_out",
                        allow_empty_files=False,
                        ignore_found_file=data_dir + case_study + ".standard_louvain_time")
    if not f or not f2 or not f3:
        print("louvain ordering output -- " + case_study + " failed")
        return
    bench.compile_experiment()
    res = bench.run_experiment(process_bv_louvain_orderings_instance)

    out_leaf_time = open(data_dir + case_study + ".leaf_louvain_time", 'a')
    out_root_time = open(data_dir + case_study + ".root_louvain_time", 'a')
    out_standard_time = open(data_dir + case_study + ".standard_louvain_time", 'a')
    for (name, leaf_time, root_time, standard_time) in res:
        out_leaf_time.write(csv_line(name, leaf_time))
        out_root_time.write(csv_line(name, root_time))
        out_standard_time.write(csv_line(name, standard_time))
    out_leaf_time.close()
    out_root_time.close()
    out_standard_time.close()


#######################################################################################################

def lll_lsr_instance(file_map):
    name = file_map['basename']
    cnf = file_map['cnf']
    s = SatInstance(cnf)
    bd = s.read_one_int_per_line_file(file_map['lsr'])
    clause_stream = s.read_cnf()
    n_vars = s.number_of_variables
    var_counts = [0] * n_vars
    longest_clause = -1
    for c in clause_stream:
        if len(c) > longest_clause:
            longest_clause = len(c)
        for l in c:
            v = abs(l)
            var_counts[v - 1] += 1
    tuples = []
    for i in range(n_vars):
        tuples.append((i, var_counts[i]))
    tuples.sort(key=operator.itemgetter(1), reverse=True)

    top_bd = [i[0] for i in tuples[:len(bd)]]
    top_bd_overlap = len(set(top_bd).intersection(bd))

    # print("----------------------------------------------------")
    # print("Name:", name)
    # print("LSR Size:", len(bd))

    # originally proposed experiment -- take all variables satisfying the lll_constant
    if longest_clause > 30:
        # print("Original Expt. Failed -- longest clause size is", longest_clause)
        allowed_vars = []
        overlap = 0
    else:
        lll_constant = (2 ** longest_clause) - 3
        allowed_vars = [i[0] for i in tuples if i[1] >= lll_constant]

        # print("LLL Constant Size: 2^" + str(longest_clause) + " (" + str(lll_constant) + ")")
        # print("Allowed Vars Size:", len(allowed_vars))
        overlap = len(set(allowed_vars).intersection(bd))
        # print("Overlap:", overlap)
    # print("Top BD Overlap:", top_bd_overlap)
    return name, len(bd), len(allowed_vars), overlap, top_bd_overlap


def lll_lsr(base_dir, data_dir, case_study):
    bench = Benchmark()

    f = bench.add_files(base_dir + '/' + case_study + "/lsr/", "lsr", ".lsr", allow_empty_files=False)
    f2 = bench.add_files(base_dir + '/' + case_study + "/cnf/", "cnf", ".cnf", allow_empty_files=False)
    if not f or not f2:
        print("LSR-LLL " + case_study + " failed")
        return None
    bench.compile_experiment()
    res = bench.run_experiment(lll_lsr_instance)
    print("RESULTS:", case_study)
    rows = []
    with open(data_dir + case_study + ".lsr_lll_same_card_overlap", 'w') as same_card, \
            open(data_dir + case_study + ".lsr_lll_2kminusc_overlap", 'w') as overlap_card, \
            open(data_dir + case_study + ".lsr_lll_2kminusc_allowed", 'w') as allowed_card:
        for (name, bd, allowed, overlap, same_card_overlap) in res:
            print(name, bd, allowed, overlap, same_card_overlap)
            rows.append([name, bd, allowed, overlap, same_card_overlap])
            same_card.write(csv_line(name, same_card_overlap))
            overlap_card.write(csv_line(name, overlap))
            allowed_card.write(csv_line(name, allowed_card))
    return rows


#######################################################################################################

def process_buckets(base_dir, curr_data_dir, c):
    raw_buckets = []
    md5_to_bucket = {}
    md5_to_sub_benchmark = {}
    if c != "sc09-app":
        raw_buckets = open(base_dir + c + "/" + c + ".raw_buckets")

    buckets = open(curr_data_dir + c + ".bucket", 'w')
    print("buckets", curr_data_dir + c + ".bucket")
    if c != "sc11-app" and c != "sc09-app":
        # map md5 to bucket
        for line in raw_buckets:
            md5 = line.split()[1]
            category = line.split()[3]
            if "crypto" in category:
                bucket = "crypto"
            elif "hardware" in category:
                bucket = "hardware"
            else:
                bucket = "software"
            md5_to_bucket[md5] = bucket

    # weka sub-benchmark test
    if c == "sc13-app":
        with open(base_dir + c + "/" + c + ".raw_buckets") as stream:
            for line in stream:
                md5 = line.split()[1]
                sub_benchmark = line.split()[5]
                md5_to_sub_benchmark[md5] = sub_benchmark
        bench = Benchmark()
        files = bench.add_files(base_dir + c + "/simp/cnf/", "cnf", ".cnf", allow_empty_files=False)
        with open(curr_data_dir + c + ".weka_bucket", 'w') as out:
            for f in files:
                name = basename(f, suffix=False)
                arr = name.split("___")
                md5 = arr[1]
                sub_benchmark = md5_to_sub_benchmark[md5]
                out.write(name + "," + sub_benchmark + "\n")
                print(name, sub_benchmark)

    # map cnf to md5 to bucket
    hard, soft, crypto, unknown = 0, 0, 0, 0
    if c == "sc14-app":
        sums = open(base_dir + c + "/" + c + ".sums")
        for line in sums:
            line = line.strip().split()
            md5 = line[0]
            name = line[1][4:-4]
            bucket = md5_to_bucket[md5]
            buckets.write(name + "," + bucket + "\n")
            if bucket == "hardware":
                hard += 1
            elif bucket == "software":
                soft += 1
            elif bucket == "crypto":
                crypto += 1
            else:
                unknown += 1
    elif c == "sc13-app":
        bench = Benchmark()
        files = bench.add_files(base_dir + c + "/simp/cnf/", "cnf", ".cnf", allow_empty_files=False)

        for f in files:
            name = basename(f, suffix=False)
            arr = name.split("___")
            md5 = arr[1]
            bucket = md5_to_bucket[md5]
            buckets.write(name + "," + bucket + "\n")
            if bucket == "hardware":
                hard += 1
            elif bucket == "software":
                soft += 1
            elif bucket == "crypto":
                crypto += 1
            else:
                unknown += 1
    elif c == "sc11-app" or c == "sc09-app":
        software_strings = ["UTI", "AProVE", "openstacks", "transport", "partial", "blocks",
                            "aaai10-planning", "UCG", "ACG", "UR", "rpoc", "dated", "ndhf", "q_query", "korf",
                            "maxor", "maxxor", "countbits", "total-", "rbcl_", "smtlib-qfbv", "minxor", "minor",
                            "grid-strips", "k2fix", "traffic"
                            ]
        hardware_strings = ["hwmcc", "manol-pipe", "velev", "9dlx", "SAT_dat", "9vliw", "pipe_"]
        crypto_strings = ["aes", "vmpc", "gss", "gus", "post-"]
        unknown_names = []
        bench = Benchmark()
        files = bench.add_files(base_dir + c + "/simp/cnf/", "cnf", ".cnf", allow_empty_files=False)
        for f in files:
            name = basename(f, suffix=False)
            bucket = ""
            if name.startswith("E0") or name.startswith("E1"):
                bucket = "software"
            for p in software_strings:
                if p in name:
                    bucket = "software"
            for p in hardware_strings:
                if p in name:
                    bucket = "hardware"
            for p in crypto_strings:
                if p in name:
                    bucket = "crypto"

            buckets.write(name + "," + bucket + "\n")
            if bucket == "hardware":
                hard += 1
            elif bucket == "software":
                soft += 1
            elif bucket == "crypto":
                crypto += 1
            else:
                # print(name)
                unknown += 1
                unknown_names.append(name)
        unknown_names = sorted(unknown_names)
        for n in unknown_names:
            print(n)

    print("#######################################", hard, soft, crypto, unknown)
    buckets.close()


#######################################################################################################

def conjoin_buckets(data_dir, bucket, case_studies):
    # get names of all instances that are in the correct bucket
    # give bucket == "app" to conjoin all sat comps
    name_lists = {}
    for c in case_studies:
        name_lists[c] = []
    for c in case_studies:
        curr_data_dir = data_dir + "/" + c + "/"
        buckets_list = open(curr_data_dir + c + ".bucket")
        for line in buckets_list:
            arr = line.strip().split(",")
            if bucket == 'app' or arr[1] == bucket:
                name_lists[c].append(arr[0])

    # create the bucket data dir
    if not os.path.exists(data_dir + bucket):
        os.makedirs(data_dir + bucket)

    # sanity check and pray this doesn't fail, but if it does just prepend names with benchmarks
    big_name_list = []
    for c in case_studies:
        big_name_list = big_name_list + [c + "__" + i for i in name_lists[c]]
    assert len(big_name_list) == len(set(big_name_list))

    # get all data files and conjoin
    suffix_map = {}
    for c in case_studies:
        curr_data_dir = data_dir + "/" + c + "/"
        print(curr_data_dir)
        files = [curr_data_dir + f for f in list(os.listdir(curr_data_dir))]
        for f in files:
            print(f)
            suffix = f.split(".")[-1]
            out_file = suffix_map.get(suffix)
            if not out_file:
                out_file = open(data_dir + bucket + "/" + bucket + "." + suffix, 'w')
                suffix_map[suffix] = out_file
            with open(f) as stream:
                for line in stream:
                    name = c + "__" + line.split(",")[0]
                    print(len(big_name_list))
                    if name in big_name_list:
                        out_file.write(line)

    # close out files
    for f in suffix_map.values():
        f.close()


#######################################################################################################


def process_simple_ratios(data_dir, case_study, n, d, name):
    files = [data_dir + "/" + case_study + "." + i for i in [n, d]]
    m = join_csvs(*files)
    out = open(data_dir + "/" + case_study + "." + name, 'w')
    for key in m.keys():
        if len(m[key]) != 2:
            continue
        try:
            num = coerce(m[key][n])
            denom = coerce(m[key][d])
        except ValueError:
            continue
        if denom == 0:
            continue
        out.write(key + "," + str(num / denom) + "\n")
    out.close()


def process_all(base_dir, data_dir, case_studies=None, processes=None, simple_processes=None,
                simple_ratios=None, buckets=True, init_setup=None):
    """
    Runs all data processing on the given benchmark, or all benchmarks if no case_study explicitly given.
    simple_processes is assumed to be a list of strings corresponding to the data_types.
    """
    comps = ["sc14-app", "sc13-app", "sc11-app", "sc09-app"]

    # regroup the sc files into app
    # subprocess.call("/home/ezulkosk/git/backdoors/shell/regroup_app.sh")

    #_,data_dir,_,fields = init_setup()
    #create_sat_db_csv(data_dir, fields)
    #sys.exit()
    if not case_studies:
        case_studies = ['feature_models']
    if len(processes) == 0:
        processes = [process_pocr_structure_output, process_fractal_dim_alpha ] #, process_sat_header, process_cdcl_structure_output_updated,
                     #process_backbones, process_weak_backdoors, process_lsr_backdoors,
                     #sprocess_cmtys, process_solver_output, process_mateescu_tw, process_lsr_sat_min] #process_lsr_heuristics, process_solver_output] # lll_lsr, process_cmtys_backdoor_overlaps]
    #if len(simple_processes) == 0:
    #    simple_processes = ['result', 'time']
    if len(simple_ratios) == 0:
        simple_ratios = [("num_clauses", "num_vars", "cvr"), ('weak_size', 'num_vars', 'wvr'),
                         ('lsr_size', 'num_vars', 'lvr'), ('backbones', 'num_vars', 'bvr'),
                         ('num_vars_in_any_weak', 'num_vars', 'unionwvr'), ('num_cmtys', 'num_vars', 'cmtysvr'),
                         ('lsr_lll_same_card_overlap', 'lsr_size', 'lsr_lll_same_card_ratio')]


    for c in case_studies:
        curr_data_dir = data_dir + "/" + c + "/"
        print("Warning --changed process all for CP")
        # TODO only doing simp for now!! <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        #
        for p in processes:
            if not p:
                break
            p(base_dir, curr_data_dir, c)
        for p in processes:  #[process_cdcl_structure_output_updated, process_lsr_backdoors, process_lsr_sat_min]: #processes:
            if not p:
                break
            p(base_dir, curr_data_dir, c, simp=True)
        if c == "agile":
            process_cdcl_structure_output(base_dir, curr_data_dir, c, simp=True)
            process_maplesat_restart_output(base_dir, curr_data_dir, c, simp=True)
        process_solver_output(base_dir, curr_data_dir, c, simp=False)

        #for s in simple_processes:
        #    if not s:
        #        break
        #    process_simple_output(base_dir, curr_data_dir, c, s)

    print("simple ratios")

    if simple_ratios[0]:
        for c in case_studies:
            curr_data_dir = data_dir + "/" + c + "/"
            for (n, d, name) in simple_ratios:
                process_simple_ratios(curr_data_dir, c, n, d, name)

    if buckets:
        # process sat comp subcategories
        for c in comps:
            curr_data_dir = data_dir + "/" + c + "/"
            process_buckets(base_dir, curr_data_dir, c)

        # create sat comp buckets data
        for b in ['hardware', 'software', 'crypto', 'app']:
            conjoin_buckets(data_dir, b, comps)

    # fields = ['buckets', 'num_clauses', 'num_vars', 'cvr', 'lsr_size', 'lvr', 'minisat', 'glucose', 'maplecomsps',
    #           'num_cmtys', 'q', 'weak_size', 'num_min_weak', 'num_vars_in_any_weak', 'wvr', 'result']
    _,data_dir,_,fields = init_setup()
    create_sat_db_csv(data_dir, fields)



def main():
    ###################################################
    # change these appropriately
    buckets = True
    simple_processes = set()  # None # ['result', 'time']  # set()
    simple_ratios = []  # [('lsr_lll_same_card_overlap', 'lsr_size', 'lsr_lll_same_card_ratio')]

    #case_studies = ["sc14-app", "sc13-app", "sc11-app", "sc09-app", "agile", "crafted", "random"]
    #case_studies = ["datapath_crafted", "mathcheck"]
    #case_studies = []
    # #,"morphed_graph_coloring", "random_3sat", "planning", "feature_models", "graph_coloring", "sat8",
    # "cdl_feature_models", "kconfig_feature_models"]
    # case_studies = ["feature_models",
    #                 "cdl_feature_models", "kconfig_feature_models", "splot_feature_models", "random_feature_models"]
    # case_studies = ["sc14-app"]
    # case_studies = None #["sc14-app"]
    base_dir = "/home/ezulkosk/cp2017_benchmarks/"
    data_dir = base_dir + "data/"
    init_setup = init_cp2017_setup
    processes = [process_solver_output, process_fractal_dim_alpha] # [process_pocr_structure_output, process_sat_header, process_cdcl_structure_output_updated, process_lsr_sat_min, process_all_decs, process_lsr_all_decs_overlap,
                     #process_backbones, process_weak_backdoors, process_lsr_backdoors, process_cmtys_backdoor_overlaps,
                     #process_cmtys, process_solver_output, process_mateescu_tw] # set() #[process_cdcl_structure_output]


    # SCALING
    #base_dir = "/home/ezulkosk/backdoors_benchmarks/scaling/"
    #data_dir = base_dir + "data/"
    #init_setup = init_scaling_setup
    base_dir, data_dir, case_studies, fields = init_setup()
    # case_studies = ["sc14-app", "sc13-app", "sc11-app", "sc09-app", "agile", "crafted", "random"]
    case_studies = ["sc13-app"]

    # QFBV
    #case_studies = ["qfbv_subset"]
    #base_dir = "/home/ezulkosk/qfbv/"
    #data_dir = base_dir + "data/"
    #processes =  [process_cmty_pp_output]  #set()  # , process_var_total_contradictions]  # None #[process_tw]
    #init_setup = init_qfbv_setup

    # gtn
    #case_studies = [""]
    #base_dir = "/home/ezulkosk/tests/gtn/gtn/"
    #data_dir = base_dir + "data/"
    #processes = [process_gtn_ls_output] # [process_gtn_solver_output, process_gtn_clauses]  # set()  # , process_var_total_contradictions]  # None #[process_tw]
    #init_setup = init_gtn_setup

    # gtn proof (local
    #case_studies = ["gtn_subset"]
    #base_dir = "/home/ezulkosk/backdoors_benchmarks/gtn/"
    #data_dir = base_dir + "data/"
    #processes = [process_gtn_ls_output]  # [process_gtn_solver_output, process_gtn_clauses]  # set()  # , process_var_total_contradictions]  # None #[process_tw]
    #init_setup = init_gtn_setup
    ###################################################
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    for c in case_studies:
        if not os.path.exists(data_dir + "/" + c):
            os.makedirs(data_dir + "/" + c)
    process_all(base_dir, data_dir, case_studies=case_studies,
                processes=processes, simple_processes=simple_processes, simple_ratios=simple_ratios,
                buckets=buckets, init_setup=init_setup)  # remove data_type for other processes
    #for s in ["maplesat", "maplesat_nr", "maplesat_ar", "rwr", "rnr"]:
    #    for t in ["units", "binary", "avg_clause", "largest", "num_clauses"]:
    #        print("('" + s + "_proof_" + t + "', FieldType.DOUBLE),")


if __name__ == '__main__':
    main()
