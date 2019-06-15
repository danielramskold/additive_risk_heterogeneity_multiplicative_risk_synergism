import matplotlib
matplotlib.use('Agg')
import pandas, random, math, argparse, numpy
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

def signedsquare(v):
    return v**2 if v>=0 else -v**2

def calc_observed_expected(df, useRR, use_z, t):
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
    a = ORorRR(df, 'A1B0', maxAB)-1
    b = ORorRR(df, 'A0B1', maxAB)-1
    expected_OR11_additive = ORorRR(df, 'A1B0', maxAB) + ORorRR(df, 'A0B1', maxAB) - 1
    expected_OR11_multiplicative = ORorRR(df, 'A1B0', maxAB) * ORorRR(df, 'A0B1', maxAB)
    coefficient_threshold = ((t-1)/(5-1))**0.5
    guess_OR11_threshold = 1 + a + b + a*b*coefficient_threshold
    observed_OR11 = ORorRR(df, 'A1B1', maxAB) 
    if use_z:
        #return signedsquare((observed_OR11 - expected_OR11_additive)/(expected_OR11_multiplicative - expected_OR11_additive)),
        return (observed_OR11 - expected_OR11_additive)/(expected_OR11_multiplicative - expected_OR11_additive),
    return (observed_OR11/expected_OR11_multiplicative), (observed_OR11/expected_OR11_additive), observed_OR11/guess_OR11_threshold

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

def separate2_cause_sim(a_rel, b_rel, n, bg_freq1, bg_freq2, prevalence):
    df = pandas.DataFrame()
    case_control_split(df, 'outcome', prevalence, n)
    spike_in(df, 'outcome', 'A', a_rel, bg_freq1)
    spike_in(df, 'outcome', 'B', b_rel, bg_freq2)
    case_control_split(df, 'group', 0.5, n)
    shuffle(df, 'A', df['group'])
    shuffle(df, 'B', ~df['group'])
    return df

def separate2_cause_sim_v2(a_rel, b_rel, n, bg_freq1, bg_freq2, prevalence):
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

def same_cause_sim(a_rel, b_rel, n, bg_freq1, bg_freq2, prevalence):
    df = pandas.DataFrame()
    case_control_split(df, 'outcome', prevalence, n)
    spike_in(df, 'outcome', 'A', a_rel, bg_freq1)
    spike_in(df, 'outcome', 'B', b_rel, bg_freq2)
    return df

def OR_logic_sim(a_rel, b_rel, n, bg_freq1, bg_freq2, prevalence):
    df = pandas.DataFrame()
    case_control_split(df, 'a_cases', 1-math.sqrt(1-prevalence), n)
    case_control_split(df, 'b_cases', 1-math.sqrt(1-prevalence), n)
    spike_in(df, 'a_cases', 'A', a_rel, bg_freq1)
    spike_in(df, 'b_cases', 'B', b_rel, bg_freq2)
    df['outcome'] = df['a_cases'] | df['b_cases']
    return df

def AND_logic_sim(a_rel, b_rel, n, bg_freq1, bg_freq2, prevalence):
    df = pandas.DataFrame()
    case_control_split(df, 'a_cases', math.sqrt(prevalence), n)
    case_control_split(df, 'b_cases', math.sqrt(prevalence), n)
    spike_in(df, 'a_cases', 'A', a_rel, bg_freq1)
    spike_in(df, 'b_cases', 'B', b_rel, bg_freq2)
    df['outcome'] = df['a_cases'] & df['b_cases']
    return df

def threshold_two_of_three(a_rel, b_rel, n, bg_freq1, bg_freq2, prevalence):
    df = pandas.DataFrame()
    case_control_split(df, 'a_cases', prevalence*2/3, n)
    case_control_split(df, 'b_cases', prevalence*2/3, n)
    case_control_split(df, 'c_cases', prevalence*2/3, n)
    spike_in(df, 'a_cases', 'A', a_rel, bg_freq1)
    spike_in(df, 'b_cases', 'B', b_rel, bg_freq2)
    df['outcome'] = df['a_cases'].astype(int) + df['b_cases'].astype(int) + df['c_cases'].astype(int) >= 2
    return df

def threshold_three_of_five(a_rel, b_rel, n, bg_freq1, bg_freq2, prevalence, threshold):
    df = pandas.DataFrame()
    subcasefreq = -math.log(threshold+prevalence, prevalence)   # not really based on anything, but sometimes it comes close ...and sometimes not
    
    if prevalence == 0.005:
        if threshold==1: subcasefreq = 0.001  # 0.005/5 - overlap~o(prev^2)
        elif threshold==2: subcasefreq = 0.022
        elif threshold==3: subcasefreq = 0.085
        elif threshold==4: subcasefreq = 0.182
        elif threshold==5: subcasefreq = 0.005**0.2 # 0.3465724215775732
    
    case_control_split(df, 'a_cases', subcasefreq, n)
    case_control_split(df, 'b_cases', subcasefreq, n)
    case_control_split(df, 'c_cases', subcasefreq, n) # for prevalence=0.5, **3 for threshold 1, **1.71 for threshold 2, **1 for threshold 3, **0.53 fo threshold 4, **0.2 for threshold 5
    case_control_split(df, 'd_cases', subcasefreq, n)
    case_control_split(df, 'e_cases', subcasefreq, n)
    spike_in(df, 'a_cases', 'A', a_rel, bg_freq1)
    spike_in(df, 'b_cases', 'B', b_rel, bg_freq2)
    df['outcome'] = df['a_cases'].astype(int) + df['b_cases'].astype(int) + df['c_cases'].astype(int) + df['d_cases'].astype(int) + df['e_cases'].astype(int) >= threshold
    return df

def add_to_dict(d, key_prefix, values, use_z):
    keys = [key_prefix+'\n'+suffix for suffix in (('z',) if use_z else('mult', 'add', 'thr'))]
    for k, v in zip(keys, values):
        if k not in d: d[k] = []
        d[k].append(v)

random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('calc', choices=['RR', 'OR'])
parser.add_argument('protective', choices=['no', 'one', 'two'])
parser.add_argument('-n', default=[1000, 1000000], type=int, nargs=2)
parser.add_argument('-z', '--add_mult_fraction', action='store_true', dest='z')
parser.add_argument('--notches', action='store_true')
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

prev_obs = defaultdict(list)
log2_obs_to_exp = OrderedDict()
for prevalence in (0.005,):
    for a_rel, b_rel, bg_freq1, bg_freq2 in zip(a_rel_arr, b_rel_arr, bg_freq_arr1, bg_freq_arr2):
        for threshold in (1, 2, 3, 4, 5):
            df = threshold_three_of_five(a_rel, b_rel, n, bg_freq1, bg_freq2, prevalence, threshold)
            prev_obs[ '%.3f\n3of5\n%i'%(prevalence, threshold)].append(numpy.mean(df['outcome']))
            #print(threshold, prev_obs, prevalence)
            add_to_dict(log2_obs_to_exp, '%.3f\n3of5\n%i'%(prevalence, threshold), calc_observed_expected(df, o.calc=='RR', o.z, threshold), o.z)

print({k:numpy.median(V) for k,V in prev_obs.items()})

xarr = list(range(1,1+len(log2_obs_to_exp)))
pyplot.gca().axhline(1)
if o.z: pyplot.gca().axhline(0)
pyplot.boxplot(list(log2_obs_to_exp.values()), sym='', notch=o.notches)
pyplot.xticks(xarr, list(log2_obs_to_exp.keys()), fontsize=2)
if o.z:
    #pyplot.ylabel('signedsquare((observed-expadd)/(expmult-expadd)) for %s(A1B1)'%o.calc)
    pyplot.ylabel('(observed-expadd)/(expmult-expadd) for %s(A1B1)'%o.calc)
    pyplot.ylim(-2, 3)
    #pyplot.plot([1,5], [0,1], 'k-')
    
    xarr = [x for x in numpy.arange(1, 5.00001, 0.02)]
    yarr = [math.sqrt((x-1)/4) for x in xarr]
    pyplot.plot(xarr, yarr, 'k-')
    
    pyplot.savefig('plot_for_threeoffive_z_%s_%sprotective%s.pdf'%(o.calc, o.protective, '_notched' if o.notches else ''))
else:
    pyplot.ylabel('observed/expected for %s(A1B1)'%o.calc)
    pyplot.ylim(0, 2.5)
    #pyplot.xlim(min(xarr)-1, min(xarr)+16)
    pyplot.savefig('plot_for_threeoffive_%s_%sprotective%s.pdf'%(o.calc, o.protective, '_notched' if o.notches else ''))
