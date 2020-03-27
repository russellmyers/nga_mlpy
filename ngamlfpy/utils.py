"""
  Utility routines
"""

import pandas as pd
import numpy as np


def unscramble(x):
    '''
    Unscramble data
    '''

    transl = {'k': '1', 'C': '3', 'e': '5', 'J': '7', 't': '0'}

    x_str = ''
    for ch in x:
        if ch in transl:
            x_str += transl[ch]
        else:
            x_str += ch


    x_adj_part_2 = ''
    for i in range(4,len(x_str)):
        j = int(x_str[i]) - int(x_str[i-4])
        if j < 0:
            j = int(x_str[i]) + 10 - int(x_str[i-4])
        x_adj_part_2 += str(j)[-1]
    x_adj_part_1 = ''
    for i in range(0,4):
        j = int(x_str[i]) - int(x_adj_part_2[i])

        if j < 0:
            j = int(x_str[i]) + 10 - int(x_adj_part_2[i])

        x_adj_part_1 += str(j)[-1]

    x_adj = x_adj_part_1 + x_adj_part_2
    x_adj_2 = ''
    for i in range(0,len(x_str)):
        j = int(x_adj[i]) - (i + 1)
        if j < 0:
            j = int(x_adj[i]) + 10 - (i + 1)
        x_adj_2 += str(j)[-1]

    return x_adj_2


def find_in_list_partial(x,seq):
    '''
    find all items in list seq which commencing with  string x
    '''
    partial_matches = []
    for s in seq:
        if s[:len(x)] == x:
           partial_matches.append(s)
    return partial_matches

def df_to_csv(df,out_file_name):
    df.to_csv(out_file_name, index=False)


# routines to convert pulled data to json and visa versa

def conv_dict_to_value_list(dict):
    out_list = [value for key, value in dict.items()]

    return out_list


def conv_dict_to_value_title_list(dict):
    out_list = []
    for key,value in dict.items():
        out_dict_entry = {'title':key,'value':value}
        out_list.append(out_dict_entry)

    return out_list

def conv_value_title_list_to_dict(value_title_list):
    out_dict = {}
    for row in value_title_list:
        out_dict[row['title']] = row['value']
    return out_dict



def pulled_df_to_json(df,model_config,period,use_first_data_line_as_selection=False,use_value_title_format=False, values_only=False, clip_emps=None):
    j = {}

    j['selection'] = {}
    if use_first_data_line_as_selection:
        first_row = df.iloc[0]
        for col in model_config.get_data_source_field_names():
            if col in df:
                if df[col].dtype == np.int64:
                    j['selection'][col] = int(first_row[col])
                else:
                    j['selection'][col] = first_row[col]
    else:
        j['selection'] = model_config.selection  #.j_config['selection']
        j['selection']['period'] = period


    if use_value_title_format:
        j['selection'] = conv_dict_to_value_title_list(j['selection'])

    def df_to_list_of_dict(df,use_value_title_format = False):
        keys = df.columns
        out_list = []
        for i, row in df.iterrows():
            if clip_emps is None:
                pass
            else:
                if i >= clip_emps:
                    break

            out_dict = {}
            for j, col in enumerate(df):
                out_dict[keys[j]] = row[col]
            if use_value_title_format:
                out_list.append(conv_dict_to_value_title_list(out_dict))
            else:
                out_list.append(out_dict)

        return out_list

    def df_to_list_of_value_lists(df):
        keys = df.columns
        out_list = []
        for i, row in df.iterrows():
            if clip_emps is None:
                pass
            else:
                if i >= clip_emps:
                    break
            out_list_row = conv_dict_to_value_list(row)
            out_list.append(out_list_row)
        return out_list

    if values_only:
        l = df_to_list_of_value_lists(df)
    else:
        l = df_to_list_of_dict(df,use_value_title_format = use_value_title_format)

    j['values'] = l


    return j


def pulled_json_to_df(j_data,use_value_title_format=False):

    if use_value_title_format:
        sel_dict = conv_value_title_list_to_dict(j_data['selection'])
    else:
        sel_dict = j_data['selection']
    sel_keys = list(sel_dict.keys())
    sel_vals   = list(sel_dict.values())

    num_sel_fields = len(sel_keys)

    if use_value_title_format:
     first_rec_dict = conv_value_title_list_to_dict(j_data['values'][0])
    else:
     first_rec_dict = j_data['values'][0]
    val_keys = list(first_rec_dict.keys())

    num_val_fields = len(val_keys)

    out_list = []

    for i,emp_entry in enumerate(j_data['values']):
        #out_entry = sel_vals[:]
        out_entry = []
        if use_value_title_format:
            for row in emp_entry:

                out_entry.append(row['value'])
        else:
            for v_key in val_keys:
                out_entry.append(emp_entry[v_key])

        out_list.append(out_entry)

    #df = pd.DataFrame(out_list,columns = sel_keys + val_keys)

    df = pd.DataFrame(out_list, columns=val_keys)

    return df


if __name__ == '__main__':
    dict = {'pernr':999,'abkrs':'AK'}
    val_list = conv_dict_to_value_title_list(dict)
    print(val_list)

    dict_back_again = conv_value_title_list_to_dict(val_list)
    print(dict_back_again)

    y = unscramble('2kJJJJkt')
    print(y)