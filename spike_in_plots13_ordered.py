import matplotlib
matplotlib.use('Agg')
import pandas, random, math, argparse
from collections import defaultdict, OrderedDict
from matplotlib import pyplot

def spike_in(df, case_col, new_col, rel, base_freq):
    df[new_col] = [random.random() < (base_freq*rel if is_case else base_freq) for is_case in df[case_col]]

def case_control_split(df, new_col, fraction, n):
    df[new_col] = [random.random() < fraction for i in range(n)]

def OR(df, colname, maxincol):
    dfS = df[df['outcome']==True]
    dfH = df[df['outcome']==False]
    if colname in ('A', 'B'):
        return dfS[colname].sum() / dfH[colname].sum() / abs(maxincol-dfS[colname]).sum() * abs(maxincol-dfH[colname]).sum()
    else:
        return dfS[colname].sum() / dfH[colname].sum() / dfS['A0B0'].sum() * dfH['A0B0'].sum()

def RR(df, colname, maxincol):
    dfS = df[df['outcome']==True]
    dfH = df[df['outcome']==False]
    if colname in ('A', 'B'):
        return dfS[colname].sum() / df[colname].sum() / abs(maxincol-dfS[colname]).sum() * abs(maxincol-df[colname]).sum()
    else:
        return dfS[colname].sum() / df[colname].sum() / dfS['A0B0'].sum() * df['A0B0'].sum()

def shuffle(df, colname, select_rows=None):
    if select_rows is None:
        values = list(df[colname])
        random.shuffle(values)
        df[colname] = values
    else:
        values = list(df.loc[select_rows, colname])
        random.shuffle(values)
        df.loc[select_rows, colname] = values

def invert(v):
    return 1/v

def calc_observed_expected(df, useRR):
    maxA = 1
    maxB = 1
    maxAB = maxA * maxB
    
    df['A'] = df['A'].astype(int)
    df['B'] = df['B'].astype(int)
    df['A1B1'] = df['A'] * df['B']
    df['A1B0'] = df['A'] * (maxB-df['B'])
    df['A0B1'] = (maxA-df['A']) * df['B']
    df['A0B0'] = (maxA-df['A']) * (maxB-df['B'])
    
    ORorRR = RR if useRR else OR
    expected_OR11_additive = ORorRR(df, 'A1B0', maxAB) + ORorRR(df, 'A0B1', maxAB) - 1
    expected_OR11_multiplicative = ORorRR(df, 'A1B0', maxAB) * ORorRR(df, 'A0B1', maxAB)
    observed_OR11 = ORorRR(df, 'A1B1', maxAB)
    if o.correlation:
        from scipy import stats
        dfS = df[df['outcome']==True]
        dfH = df[df['outcome']==False]
        return stats.pearsonr(df['A'], df['B'])[0], stats.pearsonr(dfS['A'], dfS['B'])[0], stats.pearsonr(dfH['A'], dfH['B'])[0]
    else:
        return (observed_OR11/expected_OR11_multiplicative), (observed_OR11/expected_OR11_additive)


def separate_cause_sim(a_rel, b_rel, n, bg_freq1, bgfreq2, prevalence):
    df_a = pandas.DataFrame()
    df_b = pandas.DataFrame()
    case_control_split(df_a, 'outcome', prevalence, n//2)
    case_control_split(df_b, 'outcome', prevalence, n-n//2)
    spike_in(df_a, 'outcome', 'A', a_rel, bg_freq1)
    spike_in(df_b, 'outcome', 'A', 1, bg_freq1)
    spike_in(df_a, 'outcome', 'B', 1, bg_freq2)
    spike_in(df_b, 'outcome', 'B', b_rel, bg_freq2)
    df = pandas.concat((df_a, df_b), ignore_index=True)
    return df

def separate2_cause_sim(a_rel, b_rel, n, bg_freq1, bgfreq2, prevalence):
    df = pandas.DataFrame()
    case_control_split(df, 'outcome', prevalence, n)
    spike_in(df, 'outcome', 'A', a_rel, bg_freq1)
    spike_in(df, 'outcome', 'B', b_rel, bg_freq2)
    case_control_split(df, 'group', 0.5, n)
    shuffle(df, 'A', df['group'])
    shuffle(df, 'B', ~df['group'])
    return df

def separate2_cause_sim_v2(a_rel, b_rel, n, bg_freq1, bgfreq2, prevalence):
    df_a = pandas.DataFrame()
    df_b = pandas.DataFrame()
    case_control_split(df_a, 'outcome', prevalence, n//2)
    case_control_split(df_b, 'outcome', prevalence, n-n//2)
    spike_in(df_a, 'outcome', 'A', a_rel, bg_freq1)
    spike_in(df_b, 'outcome', 'A', a_rel, bg_freq1)
    spike_in(df_a, 'outcome', 'B', b_rel, bg_freq2)
    spike_in(df_b, 'outcome', 'B', b_rel, bg_freq2)
    shuffle(df_a, 'B')
    shuffle(df_b, 'A')
    df = pandas.concat((df_a, df_b), ignore_index=True)
    return df

def same_cause_sim(a_rel, b_rel, n, bg_freq1, bgfreq2, prevalence):
    df = pandas.DataFrame()
    case_control_split(df, 'outcome', prevalence, n)
    spike_in(df, 'outcome', 'A', a_rel, bg_freq1)
    spike_in(df, 'outcome', 'B', b_rel, bg_freq2)
    return df

def OR_logic_sim(a_rel, b_rel, n, bg_freq1, bgfreq2, prevalence):
    df = pandas.DataFrame()
    case_control_split(df, 'a_cases', 1-math.sqrt(1-prevalence), n)
    case_control_split(df, 'b_cases', 1-math.sqrt(1-prevalence), n)
    spike_in(df, 'a_cases', 'A', a_rel, bg_freq1)
    spike_in(df, 'b_cases', 'B', b_rel, bg_freq2)
    df['outcome'] = df['a_cases'] | df['b_cases']
    return df

def AND_logic_sim(a_rel, b_rel, n, bg_freq1, bgfreq2, prevalence):
    df = pandas.DataFrame()
    case_control_split(df, 'a_cases', math.sqrt(prevalence), n)
    case_control_split(df, 'b_cases', math.sqrt(prevalence), n)
    spike_in(df, 'a_cases', 'A', a_rel, bg_freq1)
    spike_in(df, 'b_cases', 'B', b_rel, bg_freq2)
    df['outcome'] = df['a_cases'] & df['b_cases']
    return df

def add_to_dict(d, key_prefix, values):
    if o.correlation:
        keys = [key_prefix+'\n'+suffix for suffix in ('rall', 'rcase', 'rctrl')]
    else:
        keys = [key_prefix+'\n'+suffix for suffix in ('mult', 'add', 'het')]
    for k, v in zip(keys, values):
        if k not in d: d[k] = []
        d[k].append(v)

random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('calc', choices=['RR', 'OR'])
parser.add_argument('protective', choices=['no', 'one', 'two'])
parser.add_argument('-n', default=[1000, 1000000], type=int, nargs=2)
parser.add_argument('--correlation', action='store_true')
o = parser.parse_args()

nr, n= o.n
if o.protective == 'two':
    bg_freq_arr1 = [0.05+random.random()*0.4 for i in range(nr)]
    bg_freq_arr2 = [0.05+random.random()*0.2 for i in range(nr)]
    a_rel_arr = [1/(1.1+random.random()*3.9) for i in range(nr)]
    b_rel_arr = [1/(1.1+random.random()*0.9) for i in range(nr)]
elif o.protective == 'one':
    bg_freq_arr1 = [0.05+random.random()*0.1 for i in range(nr)]
    bg_freq_arr2 = [0.05+random.random()*0.2 for i in range(nr)]
    a_rel_arr = [1.1+random.random()*3.9 for i in range(nr)]
    b_rel_arr = [1/(1.1+random.random()*0.9) for i in range(nr)]
elif o.protective == 'no':
    bg_freq_arr1 = [0.05+random.random()*0.1 for i in range(nr)]
    bg_freq_arr2 = [0.05+random.random()*0.2 for i in range(nr)]
    a_rel_arr = [1.1+random.random()*3.9 for i in range(nr)]
    b_rel_arr = [1.1+random.random()*0.9 for i in range(nr)]

log2_obs_to_exp = OrderedDict()
for prevalence in (0.5, 0.15, 0.05, 0.015, 0.005):
    for a_rel, b_rel, bg_freq1, bg_freq2 in zip(a_rel_arr, b_rel_arr, bg_freq_arr1, bg_freq_arr2):
        add_to_dict(log2_obs_to_exp, '%.3f\n1group'%prevalence, calc_observed_expected(same_cause_sim(a_rel, b_rel, n, bg_freq1, bg_freq2, prevalence), o.calc=='RR'))
        add_to_dict(log2_obs_to_exp, '%.3f\n2group2'%prevalence, calc_observed_expected(separate2_cause_sim(a_rel, b_rel, n, bg_freq1, bg_freq2, prevalence), o.calc=='RR'))
        add_to_dict(log2_obs_to_exp, '%.3f\n2group'%prevalence, calc_observed_expected(separate_cause_sim(a_rel, b_rel, n, bg_freq1, bg_freq2, prevalence), o.calc=='RR'))
        add_to_dict(log2_obs_to_exp, '%.3f\nsynth'%prevalence, calc_observed_expected(AND_logic_sim(a_rel, b_rel, n, bg_freq1, bg_freq2, prevalence), o.calc=='RR'))
        add_to_dict(log2_obs_to_exp, '%.3f\nhetero'%prevalence, calc_observed_expected(OR_logic_sim(a_rel, b_rel, n, bg_freq1, bg_freq2, prevalence), o.calc=='RR'))
        #add_to_dict(log2_obs_to_exp, '%.3f\n2group2v2'%prevalence, calc_observed_expected(separate2_cause_sim_v2(a_rel, b_rel, n, bg_freq1, bg_freq2, prevalence), o.calc=='RR'))

xarr = list(range(1,1+len(log2_obs_to_exp)))
if o.correlation:
    pyplot.gca().axhline(0)
else:
    pyplot.gca().axhline(1)
pyplot.boxplot(list(log2_obs_to_exp.values()), sym='')
pyplot.xticks(xarr, list(log2_obs_to_exp.keys()), fontsize=2)
if o.correlation:
    pyplot.ylabel('correlation between risk factors')
else:   
    pyplot.ylabel('observed/expected for %s(A1B1)'%o.calc)
    pyplot.ylim(0, 2.5)
#pyplot.xlim(min(xarr)-1, min(xarr)+16)
pyplot.savefig('plot13_for_spike_in_models_%s_%sprotective.pdf'%('correlation' if o.correlation else o.calc, o.protective))
