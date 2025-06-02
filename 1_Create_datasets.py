import pandas as pd
import numpy as np
import os

def create_ratio_and_product_datasets(content_file, descriptor_file):
    
    # XA/B = (Σi∈A fi*Xi) / (Σi∈B fi*Xi)
    # XA*B = (Σi∈A fi*Xi) * (Σi∈B fi*Xi)
    # XA = Σi∈A fi*Xi
    # XB = Σi∈B fi*Xi
    
    content_df = pd.read_excel(content_file)
    descriptor_df = pd.read_excel(descriptor_file, index_col=0)
    
    num_A_cols = 9
    A_site_cols = content_df.columns[:num_A_cols]
    B_site_cols = content_df.columns[num_A_cols:]
    
    ratio_dict = {}
    product_dict = {}
    XA_dict = {}
    XB_dict = {}
    
    for descriptor in descriptor_df.columns:
        ratio_list = []
        product_list = []
        XA_list = []
        XB_list = []
        
        for idx in content_df.index:
            composition = content_df.loc[idx]
        
            XA = 0
            for element in A_site_cols:
                content = composition[element] 
                if content > 0:  
                    if element in descriptor_df.index:
                        descriptor_value = descriptor_df.loc[element, descriptor]
                        XA += content * descriptor_value
            
            XB = 0
            for element in B_site_cols:
                content = composition[element]
                if content > 0:
                    if element in descriptor_df.index:
                        descriptor_value = descriptor_df.loc[element, descriptor]
                        XB += content * descriptor_value
            
            XA_list.append(XA)
            XB_list.append(XB)
            ratio_list.append(XA / XB if XB != 0 else np.nan)
            product_list.append(XA * XB)
        
        ratio_dict[f'{descriptor}_1'] = ratio_list
        product_dict[f'{descriptor}_2'] = product_list
        XA_dict[f'{descriptor}_A'] = XA_list
        XB_dict[f'{descriptor}_B'] = XB_list
    
    D1_dict = {**ratio_dict, **product_dict}
    D1_df = pd.DataFrame(D1_dict)
    
    D2_dict = {**XA_dict, **XB_dict}
    D2_df = pd.DataFrame(D2_dict)
    
    save_path = r"C:\Users\Desktop"
    os.makedirs(save_path, exist_ok=True)
    
    
    wn_df = pd.read_excel(os.path.join(save_path, 'w&n.xlsx'))
    wn_first_three = wn_df.iloc[:, :3]
    wn_last_two = wn_df.iloc[:, -2:]
    
    D1_df = pd.concat([wn_first_three, D1_df, wn_last_two], axis=1)
    D2_df = pd.concat([wn_first_three, D2_df, wn_last_two], axis=1)
    
    D1_df.to_excel(os.path.join(save_path, 'D1_dataset.xlsx'), index=False)
    D2_df.to_excel(os.path.join(save_path, 'D2_dataset.xlsx'), index=False)

if __name__ == "__main__":
    content_file = 'content.xlsx'
    descriptor_file = 'discriptor.xlsx'
    
    try:
        create_ratio_and_product_datasets(content_file, descriptor_file)
        print("Done")
    except Exception as e:
        print(str(e))
