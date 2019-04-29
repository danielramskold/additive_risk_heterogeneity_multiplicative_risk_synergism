#! /usr/bin/python3
import argparse, pandas, numpy, math, random
from scipy import stats

# derived from use_linlee-logprism_v5.py and use_linlee-logprism_v6.py

def OR(df, colname, maxincol, reraise_error=False):
    dfS = df[df['outcome']==True]
    dfH = df[df['outcome']==False]
    try:
        if colname in ('A', 'B'):
            return dfS[colname].sum() / dfH[colname].sum() / abs(maxincol-dfS[colname]).sum() * abs(maxincol-dfH[colname]).sum()
        else:
            return dfS[colname].sum() / dfH[colname].sum() / dfS['A0B0'].sum() * dfH['A0B0'].sum()
    except ZeroDivisionError:
        if reraise_error: raise
        return float('nan')

def state_from_two(model, bases_p, yes_val):
    if model == 'dom': # yes_val is dominant
        return [(v1==yes_val or v2==yes_val) if v1 != '0' and v2 != '0' else None for v1,v2 in zip(bases_p[::2], bases_p[1::2])]
    elif model == 'rec': # yes_val is recessive
        return [(v1==yes_val and v2==yes_val) if v1 != '0' and v2 != '0' else None for v1,v2 in zip(bases_p[::2], bases_p[1::2])]
    elif model == '1al': # pick one allele, alternate which one
        return [V[i%2]==yes_val if V[i%2] != '0' else None for i, V in enumerate(zip(bases_p[::2], bases_p[1::2]))]
    elif model == '1st': # pick one allele
        return [V[0]==yes_val if V[0] != '0' else None for i, V in enumerate(zip(bases_p[::2], bases_p[1::2]))]
    elif model == '1rand': # pick one allele
        return [random.choice(V)==yes_val if V[0] != '0' and V[1] != '0' else None for i, V in enumerate(zip(bases_p[::2], bases_p[1::2]))]
    elif model == 'sum': # allelic, a form of co-dominant
        return [(int(v1==yes_val) + int(v2==yes_val)) if v1 != '0' and v2 != '0' else None for v1,v2 in zip(bases_p[::2], bases_p[1::2])]
# add allelic model: counts of alleles (codominant, which is how SE is)


if '__main__' == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument('tfam')
    parser.add_argument('tped12')
    parser.add_argument('SE_tfam')
    parser.add_argument('SE_tpedYN')
    parser.add_argument('--rand', choices=['samegroup', 'midfreqsepgroups', 'bgfreqsepgroups', 'bgfreqsepgroupsv2'])
    parser.add_argument('--model', choices=['dom', 'rec', '1al', '1st', '1rand', 'sum'], required=True)
    parser.add_argument('--SEmodel', choices=['dom', 'rec', '1al', '1st', '1rand', 'sum'])
    parser.add_argument('--bootstrap_replicate', '--bootstrap', action='store_true', dest='bootstrap_replicate')
    o = parser.parse_args()
    o.risk_as_1 = True
    
    if o.SEmodel is None: o.SEmodel = o.model
    maxA = 2 if o.SEmodel == 'sum' else 1
    maxB = 2 if o.model == 'sum' else 1
    maxAB = maxA * maxB
    
    
    
    df_master = pandas.DataFrame()
    
    casecontol = []
    SE_samplelist = []
    with open(o.SE_tfam, 'r') as infh:
        for li, line in enumerate(infh):
            p = line.split()
            casecontol.append(None if p[5] not in '12' else (p[5] == '2'))
            SE_samplelist.append(p[0])
    
    df_master['outcome'] = pandas.Series(casecontol, index=SE_samplelist).astype(bool)
    
    with open(o.SE_tpedYN, 'r') as infh:
        for line in infh:
            p = line.split()
            bases_p = p[4:]
            df_master['A'] = state_from_two(o.SEmodel, bases_p, 'Y')
    
    if o.rand == 'samegroup':
        df_master.loc[df_master['outcome']==True, 'A'] = list(df_master.loc[df_master['outcome']==True, 'A'].sample(frac=1).reset_index(drop=True))
        df_master.loc[df_master['outcome']==False, 'A'] = list(df_master.loc[df_master['outcome']==False, 'A'].sample(frac=1).reset_index(drop=True))
    elif o.rand == 'midfreqsepgroups':
        #split into two groups, with equal fraction of cases between them. in one, shuffle X, in the other shuffle Y.   This will be the mid-freq sep-group model.
        # odds ratio for A will drop a bit unfortunately
        df_master['group'] = numpy.random.choice(2, df_master.shape[0])
        df_master.loc[df_master['group']==0, 'A'] = list(df_master.loc[df_master['group']==0, 'A'].sample(frac=1).reset_index(drop=True))
    elif o.rand == 'bgfreqsepgroups':
        # split into two groups, with equal fraction of cases between them. in one, shuffle controls for X and spike in X into cases at same frquency as in controls, e.g. by randomly drawing from controls. same for Y in the other.    This will be the low-freq sep-group model.
        df_master['group'] = numpy.random.choice(2, df_master.shape[0])
        df_master.loc[df_master['group']==0, 'A'] = list(df_master.loc[(df_master['group']==0) & (df_master['outcome']==False), 'A'].sample(n=sum(df_master['group']==0), replace=True).reset_index(drop=True))
    elif o.rand == 'bgfreqsepgroupsv2':
        # try to deal with the too-high variance
        df_master['group'] = numpy.random.choice(2, df_master.shape[0])
        df_master.loc[(df_master['group']==0) & (df_master['outcome']==True), 'A'] = list(df_master.loc[(df_master['group']==0) & (df_master['outcome']==False), 'A'].sample(n=sum((df_master['group']==0) & (df_master['outcome']==True)), replace=True).reset_index(drop=True))
    
    SNPs_samplelist = []
    with open(o.tfam, 'r') as infh:
        for li, line in enumerate(infh):
            p = line.split()
            SNPs_samplelist.append(p[0])
    
    df_master = df_master.reindex(SNPs_samplelist)
    
    outputs = 'SNP', 'expected_AB_add', 'expected_AB_mult', 'ratio_A1B1', 'ratio_A1B0', 'ratio_A0B1', 'OR_A', 'OR_B', 'min_n_ABoutcomeTetrad', 'OR_A1B1', 'OR_A1B0', 'OR_A0B1'
    print('\t'.join(list(map(str, outputs))))
    
    with open(o.tped12, 'r') as infh:
        for line in infh:
            p = line.split()
            bases_p = p[4:]
            df_master['B'] = state_from_two(o.model, bases_p, '1')
            df = df_master.dropna().copy()
            try:
                if o.risk_as_1 and OR(df, 'B', maxB, True) < 1:
                    df['B'] = maxB - df['B'].astype(int)     # should this be here or after the shuffle/bootstrap steps? would give some srtange effects, esp OR confidence interval, if it was placed just before df['A1B1'] = df['A'] * df['B'] I figure
            except ZeroDivisionError:
                continue
            
            if o.rand == 'samegroup':
                df.loc[df['outcome']==True, 'B'] = list(df.loc[df['outcome']==True, 'B'].sample(frac=1).reset_index(drop=True))
                df.loc[df['outcome']==False, 'B'] = list(df.loc[df['outcome']==False, 'B'].sample(frac=1).reset_index(drop=True))
            elif o.rand == 'midfreqsepgroups':
                df.loc[df['group']==1, 'B'] = list(df.loc[df['group']==1, 'B'].sample(frac=1).reset_index(drop=True))
            elif o.rand == 'bgfreqsepgroups':
                df.loc[df['group']==1, 'B'] = list(df.loc[(df['group']==1) & (df['outcome']==False), 'B'].sample(n=sum(df['group']==1), replace=True).reset_index(drop=True))
            elif o.rand == 'bgfreqsepgroupsv2':
                df.loc[(df['group']==1) & (df['outcome']==True), 'B'] = list(df.loc[(df_master['group']==1) & (df['outcome']==False), 'B'].sample(n=sum((df['group']==1) & (df['outcome']==True)), replace=True).reset_index(drop=True))
            
            if o.bootstrap_replicate:
                df = df.sample(frac=1, axis=0, replace=True).reset_index(drop=True)    # not setting random_state, thus breaking LD
            
            df['A'] = df['A'].astype(int)
            df['B'] = df['B'].astype(int)
            df['A1B1'] = df['A'] * df['B']
            df['A1B0'] = df['A'] * abs(maxB-df['B'])
            df['A0B1'] = abs(maxA-df['A']) * df['B']
            df['A0B0'] = abs(maxA-df['A']) * abs(maxB-df['B'])
            
            
            
            ABoutcomeTetrads_sums = [df[df['outcome']==outcome][snps].sum() for outcome in [0, 1] for snps in ['A1B1', 'A1B0', 'A0B1', 'A0B0']]
            
            ratio_tetrad = [vYes/vNo for vNo, vYes in zip(ABoutcomeTetrads_sums[:4], ABoutcomeTetrads_sums[4:])]
            ratio_tetrad = [v/ratio_tetrad[-1] for v in ratio_tetrad]
            
            expected_AB_mult = ratio_tetrad[1]*ratio_tetrad[2]
            expected_AB_add = ratio_tetrad[1]+ratio_tetrad[2]-1
            
            try:
                outputs = p[1], expected_AB_add, expected_AB_mult, *ratio_tetrad[:3], OR(df, 'A', maxB), OR(df, 'B', maxB), min(ABoutcomeTetrads_sums), OR(df, 'A1B1', maxAB), OR(df, 'A1B0', maxAB), OR(df, 'A0B1', maxAB)
                print('\t'.join(list(map(str, outputs))))
                
            except BrokenPipeError:
                break
