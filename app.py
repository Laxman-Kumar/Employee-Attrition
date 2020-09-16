import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import mlxtend as ml
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder
import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

df2 = pd.read_csv('./HW 01(5).csv')
df2.drop('Unnamed: 0', axis=1, inplace=True)

# In[131 ]:
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.layout = html.Div([
	
	html.Div(
          [
             html.P('Enter Minimum Support'),
             dcc.Input(id='Min_Support', value=0.5, type='number', min=0, step=0.01,max = 1),
          ],
      className='row'
   ),
    
    html.P('Enter Minimum Confidence'),
    dcc.Input(id='Min_Confidence',value=0.5, type='number', min=0, step=0.1,max = 1),
    html.P('Enter Minimum Lift'),
    dcc.Input(id='Min_Lift',value=1.0, type='number',min=0,step=0.1,max = 2),
    html.P('Use the below buttons to select fixed values as consequents'),
    html.P('Select "Attrition = No"?'),
    dcc.RadioItems(
                id='attrition_no',
                options=[{'label': i, 'value': i} for i in ['Yes','No']],
                value='Yes',
            ),
    html.P('Select "Attrition = Yes"?'),
    dcc.RadioItems(
                id='attrition_yes',
                options=[{'label': i, 'value': i} for i in ['Yes','No']],
                value='Yes',
            ),
    html.P('Filter Top N Rules'),
    dcc.Input(id='return_values',value=10, type='number',min=0,step=1,max = 30),
    html.P('Based on which parameter would you like to sort the top rules?'),
    dcc.Dropdown(
                id='Parameters',
                options=[{'label': i, 'value': i} for i in ['support','confidence','lift']],
                value='support',
                style={'width': '160px'}
            ),
    dcc.Graph(id='parameter-graphic'),
    html.P(id='err', style={'color': 'red'}),
    html.P(id='out'),
    html.Div(id='output-data-upload')
])    
    
@app.callback(
        [dash.dependencies.Output('output-data-upload', 'children'),
        dash.dependencies.Output('parameter-graphic', 'figure')],
        [dash.dependencies.Input('Min_Lift', 'value'),
         dash.dependencies.Input('Min_Support', 'value'),
         dash.dependencies.Input('Min_Confidence', 'value'),
         dash.dependencies.Input('attrition_no', 'value'),
         dash.dependencies.Input('attrition_yes', 'value'),
         dash.dependencies.Input('return_values', 'value'),
         dash.dependencies.Input('Parameters', 'value')]
)
def arm(min_lift,min_supp,min_conf,attrition_no,attrition_yes,return_values,para):
    consequent = []
    if attrition_no == "Yes":
        consequent.append('Attrition= No')
    if attrition_yes == "Yes":
        consequent.append('Attrition= Yes')
    
    frequent_itemsets = apriori(df2, min_supp, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_conf)
    #filter according to lift
    rules = rules[rules['lift'] > min_lift]
    sup_rules = pd.DataFrame()
    for i in consequent:
        df = rules[rules['consequents'] == {i}]
        sup_rules = sup_rules.append(df,ignore_index = True)
        
    if consequent == []:
        sup_rules = sup_rules.append(rules,ignore_index = True)
      
    if not sup_rules.empty:
        for name in sup_rules.columns[2:]:
            sup_rules[name] = sup_rules[name].astype('str')
        for i,val in enumerate(sup_rules['antecedents']):
            sup_rules.at[i,'antecedents'] = list(val)
        for i,val in enumerate(sup_rules['consequents']):
            sup_rules.at[i,'consequents'] = list(val) 
            
    sup_rules = sup_rules.sort_values(by=para,ascending=False).head(return_values)
    fig={
        'data': [{'x': sup_rules['support'] , 'y' : sup_rules['confidence'],'type': 'scatter',
                  'mode':'markers',
                  'marker': {'size':10,'color':sup_rules['lift'],'showscale':True},
                  'text': 'lift= '+sup_rules['lift']}],
        'layout':{
            'xaxis':{ 
                'title':'Support'},
            'yaxis':{
                 'title':'Confidence'
            }}}
    print("-------------")
    print(sup_rules)
    return html.Div([
			dash_table.DataTable(
				id='table',
				columns=[{"name": i, "id": i} for i in sup_rules.columns],
                data=sup_rules.to_dict("rows"),
				style_cell={'width': '300px',
				'height': '60px',
				'textAlign': 'left'})
            ]), fig
    
if __name__ == '__main__':
    app.run_server(debug=True)