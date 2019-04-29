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
    elif len(colname)==4:
        return dfS[colname].sum() / dfH[colname].sum() / dfS['A0B0'].sum() * dfH['A0B0'].sum()
    else:
        return dfS[colname].sum() / dfH[colname].sum() / dfS['A0B0C0'].sum() * dfH['A0B0C0'].sum()

def RR(df, colname, maxincol):
    dfS = df[df['outcome']==True]
    dfH = df[df['outcome']==False]
    if colname in ('A', 'B'):
        return dfS[colname].sum() / df[colname].sum() / abs(maxincol-dfS[colname]).sum() * abs(maxincol-df[colname]).sum()
    elif len(colname)==4:
        return dfS[colname].sum() / df[colname].sum() / dfS['A0B0'].sum() * df['A0B0'].sum()
    else:
        return dfS[colname].sum() / df[colname].sum() / dfS['A0B0C0'].sum() * df['A0B0C0'].sum()

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

def expected_3ways(combine, annotate=False):
    ret = []
    for XYrel in ('mult', 'add'):
        for XZrel in ('mult', 'add'):
            for YZrel in ('mult', 'add'):
                if YZrel==XZrel:
                    exp2 = combine(XYrel, 'A1B0C0', 'A0B1C0')
                    exp3 = combine(YZrel, exp2, 'A0B0C1')
                elif XYrel==XZrel:
                    exp2 = combine(YZrel, 'A0B1C0', 'A0B0C1')
                    exp3 = combine(XYrel, exp2, 'A1B0C0')
                elif XYrel==YZrel:
                    exp2 = combine(XZrel, 'A1B0C0', 'A0B0C1')
                    exp3 = combine(XYrel, exp2, 'A0B1C0')
                if annotate:
                    ret.append('%s,%s,%s\n'%(XYrel, XZrel, YZrel)+exp3)
                else:
                    ret.append(exp3)
    return ret

def calc_observed_expected(df, useRR):
    maxA = 1
    maxB = 1
    maxC = 1
    maxAB = maxA * maxB
    maxAC = maxA * maxC
    maxBC = maxB * maxC
    maxABC = maxAB * maxC
    
    df['A'] = df['A'].astype(int)
    df['B'] = df['B'].astype(int)
    df['A1B1'] = df['A'] * df['B']
    df['A1B0'] = df['A'] * (maxB-df['B'])
    df['A0B1'] = (maxA-df['A']) * df['B']
    df['A0B0'] = (maxA-df['A']) * (maxB-df['B'])
    df['A0B0C0'] = (maxA-df['A']) * (maxB-df['B']) * (maxC-df['C'])
    df['A1B0C0'] = df['A'] * (maxB-df['B']) * (maxC-df['C'])
    df['A0B1C0'] = (maxA-df['A']) * df['B'] * (maxC-df['C'])
    df['A0B0C1'] = (maxA-df['A'])* (maxB-df['B']) * df['C']
    df['A1B1C1'] = df['A'] * df['B'] * df['C']
    
    ORorRR = RR if useRR else OR
    observed_OR11 = ORorRR(df, 'A1B1', maxAB)
    
    def combine_calc(calc, val1, val2):
        val1 = ORorRR(df, val1, maxABC) if isinstance(val1, str) else val1
        val2 = ORorRR(df, val2, maxABC) if isinstance(val2, str) else val2
        if calc == 'add': return val1 + val2 - 1
        else: return val1 * val2
    
    obs = ORorRR(df, 'A1B1C1', maxABC)
    return [obs/exp for exp in expected_3ways(combine_calc)]   # gives 9 values, because 9 = n^(m*(m-1)/2) where m=2(len(add, mult)) and n=3

def mult_mult_add_3way(a_rel, b_rel, c_rel, n, bg_freq1, bg_freq2, bg_freq3, prevalence):
    # case if A and (B or C)
    df = pandas.DataFrame()
    case_control_split(df, 'a_cases', math.sqrt(prevalence), n)
    case_control_split(df, 'b_cases', 1-math.sqrt(1-math.sqrt(prevalence)), n)
    case_control_split(df, 'c_cases', 1-math.sqrt(1-math.sqrt(prevalence)), n)
    spike_in(df, 'a_cases', 'A', a_rel, bg_freq1)
    spike_in(df, 'b_cases', 'B', b_rel, bg_freq2)
    spike_in(df, 'c_cases', 'C', c_rel, bg_freq3)
    df['outcome'] = df['a_cases'] & (df['b_cases'] | df['c_cases'])
    return df

def separate_cause_sim(a_rel, b_rel, n, bg_freq1, bg_freq2, prevalence):
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

def add_to_dict(d, key_prefix, values):
    
    def combine_write(calc, val1, val2):
        val1 = val1.replace('A','').replace('B','').replace('C','')
        val2 = val2.replace('A','').replace('B','').replace('C','')
        if calc == 'add': return '(%s + %s - 1)'%(val1, val2)
        else: return '(%s * %s)'%(val1, val2)
    
    keys = [key_prefix+'\n'+suffix for suffix in expected_3ways(combine_write, True)]
    for k, v in zip(keys, values):
        if k not in d: d[k] = []
        d[k].append(v)

random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('calc', choices=['RR', 'OR'])
parser.add_argument('protective', choices=['no', 'one', 'two'])
parser.add_argument('-n', default=[1000, 1000000], type=int, nargs=2)
o = parser.parse_args()

nr, n= o.n
if o.protective == 'two':
    bg_freq_arr1 = [0.05+random.random()*0.4 for i in range(nr)]
    bg_freq_arr2 = [0.05+random.random()*0.2 for i in range(nr)]
    bg_freq_arr3 = [0.05+random.random()*0.25 for i in range(nr)]
    a_rel_arr = [1/(1.1+random.random()*3.9) for i in range(nr)]
    b_rel_arr = [1/(1.1+random.random()*0.9) for i in range(nr)]
    c_rel_arr = [1/(1.1+random.random()*1.9) for i in range(nr)]
elif o.protective == 'one':
    bg_freq_arr1 = [0.05+random.random()*0.1 for i in range(nr)]
    bg_freq_arr2 = [0.05+random.random()*0.2 for i in range(nr)]
    bg_freq_arr3 = [0.05+random.random()*0.25 for i in range(nr)]
    a_rel_arr = [1.1+random.random()*3.9 for i in range(nr)]
    b_rel_arr = [1/(1.1+random.random()*0.9) for i in range(nr)]
    c_rel_arr = [1/(1.1+random.random()*1.9) for i in range(nr)]
elif o.protective == 'no':
    bg_freq_arr1 = [0.05+random.random()*0.1 for i in range(nr)]
    bg_freq_arr2 = [0.05+random.random()*0.2 for i in range(nr)]
    bg_freq_arr3 = [0.05+random.random()*0.25 for i in range(nr)]
    a_rel_arr = [1.1+random.random()*3.9 for i in range(nr)]
    b_rel_arr = [1.1+random.random()*0.9 for i in range(nr)]
    c_rel_arr = [1.1+random.random()*1.9 for i in range(nr)]

prev_obs = defaultdict(list)
log2_obs_to_exp = OrderedDict()
for prevalence in (0.005, 0.0015, 0.0005,):
    for a_rel, b_rel, c_rel, bg_freq1, bg_freq2, bg_freq3 in zip(a_rel_arr, b_rel_arr, c_rel_arr, bg_freq_arr1, bg_freq_arr2, bg_freq_arr3):
        df = mult_mult_add_3way(a_rel, b_rel, c_rel, n, bg_freq1, bg_freq2, bg_freq3, prevalence)
        prev_obs[ '%.3f'%(prevalence)].append(numpy.mean(df['outcome']))
        add_to_dict(log2_obs_to_exp, '%.3f\n3way_mult_mult_add'%(prevalence), calc_observed_expected(df, o.calc=='RR'))

print({k:numpy.median(V) for k,V in prev_obs.items()})

xarr = list(range(1,1+len(log2_obs_to_exp)))
pyplot.gca().axhline(1)
pyplot.boxplot(list(log2_obs_to_exp.values()), sym='')
pyplot.xticks(xarr, list(log2_obs_to_exp.keys()), fontsize=2)
pyplot.ylabel('observed/expected for %s(A1B1)'%o.calc)
pyplot.ylim(0, 2.5)
#pyplot.xlim(min(xarr)-1, min(xarr)+16)
pyplot.savefig('plot_for_3way_mult_mult_add_%s_%sprotective.pdf'%(o.calc, o.protective))
