import pandas as pd

def get_star(p):
    if p > 0.05:
        return ''
    elif p > 0.01:
        return '$^{*}$'
    elif p > 0.005:
        return '$^{**}$'
    else:
        return '$^{***}$'


def draw_reg_col(result,col_name,var_names=None,digit=3):
    
    if var_names is None:
        var_names = result.params.index
    
    col_result = pd.DataFrame(columns=[col_name])

    for r in range(len(var_names)):
        _var = var_names[r]
        _param = str(round(result.params.loc[_var],digit)) + get_star(result.pvalues.loc[_var])
        _se = '(' + str(round(result.bse.loc[_var],digit)) +')'
        col_result.loc['b_'+_var] = _param
        col_result.loc['se_'+_var] = _se

    col_result.loc['nobs'] = int(result.nobs)
    col_result.loc['AIC'] = str(round(result.aic,digit))

    return col_result


def add_border(input_string):

    # Replace '\toprule', '\midrule', '\bottomrule' with '\hline'
    output_string = input_string.replace('\\toprule', '\\hline').replace('\\midrule', '\\hline').replace('\\bottomrule', '\\hline')
    
    # Insert '\hline' before '\nobservations'
    index = output_string.find('\nobservations')
    output_string = output_string[:index] + '\\hline\n' + output_string[index:]

    return output_string


def make_table(input_df,output_path):
    with open(output_path,'w') as f:
        # tex_code = '\\documentclass[12px]{article} \n \\begin{document} \n' + input_df.to_latex() + '\n \end{document}'
        tex_code = input_df.to_latex()
        tex_code = add_border(tex_code)
        f.write(tex_code)