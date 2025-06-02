from hyperopt import fmin, hp, STATUS_OK, tpe
import pandas as pd
import numpy as np
from pymatgen.core import Composition

descriptor_df = pd.read_excel('discriptor.xlsx' , index_col=0)

A_elements = ['Bi', 'Na', 'Ba', 'Ag', 'K', 'Ca', 'Sr', 'La', 'Sm']
B_elements = ['Ti', 'Nb', 'Al', 'Mg', 'Sn', 'Ta', 'Zr']

def generate_features(formula):
   
    comp = Composition(formula)
    
    a_content = {elem: 0 for elem in A_elements}
    b_content = {elem: 0 for elem in B_elements}
    
    for elem, amt in comp.items():
        if elem.symbol == 'O':
            continue
        if elem.symbol in A_elements:
            a_content[elem.symbol] = amt
        elif elem.symbol in B_elements:
            b_content[elem.symbol] = amt
    
    a_sum = sum(a_content.values())
    b_sum = sum(b_content.values())
    if a_sum > 0:
        for elem in a_content:
            a_content[elem] /= a_sum
    if b_sum > 0:
        for elem in b_content:
            b_content[elem] /= b_sum
    
    
    features = []
    for descriptor in descriptor_df.columns:
        XA = 0
        for elem in A_elements:
            content = a_content[elem]
            if content > 0 and elem in descriptor_df.index:
                XA += content * descriptor_df.loc[elem, descriptor]
        
        XB = 0
        for elem in B_elements:
            content = b_content[elem]
            if content > 0 and elem in descriptor_df.index:
                XB += content * descriptor_df.loc[elem, descriptor]
        
        features.append(XA)
        features.append(XB)
    
    #  t & u
    a_radius = 0
    b_radius = 0
    o_radius = 1.40  
    for elem in A_elements:
        content = a_content[elem]
        if content > 0 and elem in descriptor_df.index:
            a_radius += content * descriptor_df.loc[elem, 'RS']
    for elem in B_elements:
        content = b_content[elem]
        if content > 0 and elem in descriptor_df.index:
            b_radius += content * descriptor_df.loc[elem, 'RS']
    
    t = (a_radius + o_radius) / (np.sqrt(2) * (b_radius + o_radius)) if (b_radius + o_radius) != 0 else np.nan
    u = b_radius / o_radius if o_radius != 0 else np.nan
    
    features.append(t)
    features.append(u)
    
    return features

def generate_formulas(formular_info, rounds=1000):
   
    formulas_with_features = []
    
    a_fixed = ['Bi', 'Na']
    a_optional = [elem for elem in formular_info[0] if elem not in a_fixed]
    
    b_fixed = ['Ti']
    b_optional = [elem for elem in formular_info[1] if elem not in b_fixed]
    
    space = {
        'A': {
            'ratios': {
                'r_bi': hp.uniform('r_bi', 0, 1),
                'r_na': hp.uniform('r_na', 0, 1),
                'r_extra': hp.uniform('r_extra', 0, 1)
            },
            'extra': hp.choice('a_extra', a_optional)
        },
        'B': {
            'ratios': {
                'r_ti': hp.uniform('r_ti', 0, 1),
                'r_extra1': hp.uniform('r_extra1', 0, 1),
                'r_extra2': hp.uniform('r_extra2', 0, 1)
            },
            'extra1': hp.choice('b_extra1', b_optional),
            'extra2': hp.choice('b_extra2', ['None'] + b_optional)
        }
    }
    
    def f(params):
        formular_name = ""
        
        a_ratios = [params['A']['ratios']['r_bi'], params['A']['ratios']['r_na'], params['A']['ratios']['r_extra']]
        a_sum = sum(a_ratios)
        if a_sum == 0:
            return {"loss": 0, "status": STATUS_OK}
        
        a_elements = a_fixed + [params['A']['extra']]
        for elem, ratio in zip(a_elements, a_ratios):
            ratio = ratio / a_sum
            if ratio >= 0.01:
                formular_name += elem
                if ratio != 1:
                    formular_name += f"{ratio:.4f}"
        
        b_ratios = [params['B']['ratios']['r_ti'], params['B']['ratios']['r_extra1'], params['B']['ratios']['r_extra2']]
        b_sum = sum(b_ratios)
        if b_sum == 0:
            return {"loss": 0, "status": STATUS_OK}
        
        b_elements = b_fixed + [params['B']['extra1']]
        if params['B']['extra2'] != 'None':
            b_elements.append(params['B']['extra2'])
        else:
            b_ratios[2] = 0
        
        for elem, ratio in zip(b_elements, b_ratios[:len(b_elements)]):
            ratio = ratio / b_sum
            if ratio >= 0.01:
                formular_name += elem
                if ratio != 1:
                    formular_name += f"{ratio:.4f}"
        
        formular_name += "O3"
        
        if not any(f[0] == formular_name for f in formulas_with_features):
            features = generate_features(formular_name)
            if len(features) == 62:  
                formulas_with_features.append([formular_name, features])
        
        return {"loss": 0, "status": STATUS_OK}
    
    fmin(fn=f, space=space, algo=tpe.suggest, max_evals=rounds)
    return formulas_with_features

formular_info = [
    ['Bi', 'Na', 'Ba', 'Ag', 'K', 'Ca', 'Sr', 'La', 'Sm'],  # A site
    ['Ti', 'Nb', 'Al', 'Mg', 'Sn', 'Ta', 'Zr']  # B site
]

d2_descriptors = ['SF', 'CD', 'AD', 'M', 'CED', 'VED', 'No', 'EG-MB', 'EG-P', 'EC', 
                  'FEI', 'SE', 'MP', 'NM', 'NEC-C', 'NEC-S', 'QN', 'CR', 'PCR', 'VEN/NC', 
                  'V', 'RB', 'RS', 'A-O', 'EVR', 'CVR', 'EA', 'EP', 'P', 'ED']

formulas_with_features = generate_formulas(formular_info, rounds=50000)

output_file = "generated_formulas_with_features.csv"
with open(output_file, "w") as f:
    header = "formula," + ",".join([f"{desc}_{suffix}" for desc in d2_descriptors 
                                    for suffix in ['A', 'B']]) + ",t,u\n"
    f.write(header)
    for formula, features in formulas_with_features:
        f.write(f"{formula},{','.join(map(str, features))}\n")

print(f"Generated {len(formulas_with_features)} formuals and saced to {output_file}")