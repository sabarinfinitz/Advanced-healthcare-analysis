import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import numpy as np
from dash.exceptions import PreventUpdate
from datetime import datetime, timedelta

# Load the dataset
df = pd.read_csv('healthcare_dataset.csv')

# List of photo URLs (ensure these are valid paths)
photo_urls = [
    '/assets/image-generated-by-ai-portrait-handsome-asian-man_803126-1182.jpg',
    '/assets/images (4).jpg',
    '/assets/images (5).jpg',
    '/assets/images (6).jpg',
    '/assets/sampphoto.jpg'
]

# Custom CSS for enhanced styling
custom_css = '''
    body {
        font-family: 'Roboto', sans-serif;
        background-color: #f0f2f5;
    }
    .container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px;
    }
    .header {
        background-color: #2c3e50;
        color: white;
        padding: 20px;
        margin-bottom: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .card {
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 20px;
        margin-bottom: 20px;
    }
    .button {
        background-color: #3498db;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .button:hover {
        background-color: #2980b9;
    }
    .input {
        width: 100%;
        padding: 10px;
        margin: 10px 0;
        border: 1px solid #bdc3c7;
        border-radius: 5px;
    }
'''

# Initialize the Dash app
app = dash.Dash(__name__)
app.title = "Advanced Medical Dashboard"

# App layout
app.layout = html.Div([
    html.Link(
        rel='stylesheet',
        href='https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap'
    ),
    # Instead of html.Style, we'll use the app's assets folder for custom CSS
    html.Div([
        html.H1("Advanced Patient Health Monitoring Dashboard", className="header"),
        
        html.Div([
            dcc.Dropdown(
                id='patient-name-dropdown',
                options=[{'label': name, 'value': name} for name in df['Name'].unique()],
                placeholder='Select or Enter Patient Name',
                className="input"
            ),
            html.Button('Submit', id='submit-button', className="button"),
        ], className="card"),
        
        html.Div(id='patient-info', className="card"),
        
        dcc.Graph(id='treatment-timeline', className="card"),
        
        html.Div([
            dcc.Graph(id='vital-signs', style={'width': '70%', 'display': 'inline-block'}),
            html.Div([
                html.Label('Select Health Metrics:'),
                dcc.Checklist(
                    id='vital-signs-metrics',
                    options=[
                        {'label': 'Heart Rate', 'value': 'Heart Rate'},
                        {'label': 'Systolic BP', 'value': 'Systolic BP'},
                        {'label': 'Diastolic BP', 'value': 'Diastolic BP'},
                        {'label': 'Temperature', 'value': 'Temperature'},
                        {'label': 'Oxygen Saturation', 'value': 'Oxygen Saturation'},
                        {'label': 'Respiratory Rate', 'value': 'Respiratory Rate'}
                    ],
                    value=['Heart Rate', 'Systolic BP', 'Diastolic BP', 'Temperature']
                ),
                html.Label('Select Timeframe (Days):'),
                dcc.Slider(id='timeframe-slider', min=7, max=60, step=1, value=30, marks={i: str(i) for i in range(7, 61, 7)})
            ], style={'width': '25%', 'display': 'inline-block', 'vertical-align': 'top', 'margin-left': '20px'})
        ], className="card"),
        
        html.Div([
            html.Button('Download Patient Data as CSV', id='download-csv-button', className="button"),
            dcc.Download(id='download-csv-data'),
        ], className="card"),
        
        html.H2("Compare Patients", className="header"),
        
        html.Div([
            dcc.Input(id='patient-name-1', type='text', placeholder='Enter First Patient Name', className="input"),
            dcc.Input(id='patient-name-2', type='text', placeholder='Enter Second Patient Name', className="input"),
            html.Button('Compare', id='compare-button', className="button"),
        ], className="card"),
        
        html.Div(id='comparison-info', className="card"),
        dcc.Graph(id='comparison-timeline', className="card"),
        dcc.Graph(id='comparison-vital-signs', className="card"),
        
        html.H2("Search Patients by Blood Group", className="header"),
        
        html.Div([
            dcc.Dropdown(
                id='blood-group-dropdown',
                options=[{'label': bg, 'value': bg} for bg in df['Blood Type'].unique()],
                placeholder='Select Blood Group',
                className="input"
            ),
        ], className="card"),
        
        html.Div(id='blood-group-results', className="card"),
        
        html.Div([
            html.H2("Medical Alerts", className="header"),
            html.Div(id='medical-alerts', className="card")
        ])
    ], className="container")
])

# Function to generate random vital signs data
def generate_random_vital_signs(days):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days-1)
    dates = pd.date_range(start=start_date, end=end_date, periods=days)
    
    return pd.DataFrame({
        'Date': dates,
        'Heart Rate': np.random.randint(60, 100, size=days),
        'Systolic BP': np.random.randint(100, 140, size=days),
        'Diastolic BP': np.random.randint(60, 90, size=days),
        'Temperature': np.random.uniform(97, 100, size=days).round(1),
        'Oxygen Saturation': np.random.randint(95, 100, size=days),
        'Respiratory Rate': np.random.randint(12, 20, size=days)
    })

# Callback to update individual patient details
@app.callback(
    [Output('patient-info', 'children'),
     Output('treatment-timeline', 'figure'),
     Output('vital-signs', 'figure'),
     Output('medical-alerts', 'children')],
    [Input('submit-button', 'n_clicks'),
     Input('vital-signs-metrics', 'value'),
     Input('timeframe-slider', 'value')],
    [State('patient-name-dropdown', 'value')]
)
def update_individual_patient(n_clicks, selected_metrics, days, patient_name):
    if not patient_name:
        raise PreventUpdate

    patient_data = df[df['Name'].str.contains(patient_name, case=False, na=False)]

    if patient_data.empty:
        return html.Div("No data found for this patient."), {}, {}, []

    patient_info = patient_data.iloc[0].to_dict()
    patient_info_text = [
        html.H3(f"Patient Information: {patient_info['Name']}"),
        html.P(f"Gender: {patient_info['Gender']}"),
        html.P(f"Age: {patient_info['Age']}"),
        html.P(f"Blood Type: {patient_info['Blood Type']}"),
        html.P(f"Hospital: {patient_info['Hospital']}"),
        html.P(f"Medical Condition: {patient_info['Medical Condition']}"),
        html.Img(src=photo_urls[0], style={'width': '150px', 'height': '150px', 'border-radius': '10px'})
    ]

    vital_signs_data = generate_random_vital_signs(days)
    fig_vital_signs = px.line(vital_signs_data, x='Date', y=selected_metrics, title=f'Vital Signs Over {days} Days')
    fig_vital_signs.update_layout(xaxis_title='Date', yaxis_title='Value', legend_title='Metrics')

    # Create a more detailed treatment timeline
    treatments = [
        {'Date': '2023-01-15', 'Treatment': 'Initial Diagnosis'},
        {'Date': '2023-02-01', 'Treatment': 'Medication Started'},
        {'Date': '2023-03-10', 'Treatment': 'Follow-up Checkup'},
        {'Date': '2023-04-05', 'Treatment': 'Therapy Session'},
        {'Date': '2023-05-20', 'Treatment': 'Medication Adjusted'}
    ]
    df_timeline = pd.DataFrame(treatments)
    df_timeline['Date'] = pd.to_datetime(df_timeline['Date'])
    fig_timeline = px.timeline(df_timeline, x_start='Date', x_end='Date', y='Treatment', title='Treatment Timeline')
    fig_timeline.update_yaxes(autorange="reversed")

    # Generate medical alerts
    alerts = []
    last_vitals = vital_signs_data.iloc[-1]
    if last_vitals['Heart Rate'] > 100:
        alerts.append(html.P("⚠️ High Heart Rate detected", style={'color': 'red'}))
    if last_vitals['Systolic BP'] > 140 or last_vitals['Diastolic BP'] > 90:
        alerts.append(html.P("⚠️ High Blood Pressure detected", style={'color': 'red'}))
    if last_vitals['Temperature'] > 99:
        alerts.append(html.P("⚠️ Elevated Temperature detected", style={'color': 'red'}))
    if last_vitals['Oxygen Saturation'] < 95:
        alerts.append(html.P("⚠️ Low Oxygen Saturation detected", style={'color': 'red'}))
    
    if not alerts:
        alerts.append(html.P("No critical alerts at this time", style={'color': 'green'}))

    return html.Div(patient_info_text), fig_timeline, fig_vital_signs, alerts

# Callback to handle CSV download
@app.callback(
    Output('download-csv-data', 'data'),
    [Input('download-csv-button', 'n_clicks')],
    [State('patient-name-dropdown', 'value')]
)
def download_patient_data_as_csv(n_clicks, patient_name):
    if n_clicks == 0 or not patient_name:
        raise PreventUpdate

    patient_data = df[df['Name'].str.contains(patient_name, case=False, na=False)]
    if patient_data.empty:
        raise PreventUpdate

    return dcc.send_data_frame(patient_data.to_csv, filename=f'{patient_name}_data.csv')

# Callback for comparing patients
@app.callback(
    [Output('comparison-info', 'children'),
     Output('comparison-timeline', 'figure'),
     Output('comparison-vital-signs', 'figure')],
    [Input('compare-button', 'n_clicks')],
    [State('patient-name-1', 'value'), State('patient-name-2', 'value')]
)
def compare_patients(n_clicks, patient_name_1, patient_name_2):
    if n_clicks == 0 or not patient_name_1 or not patient_name_2:
        raise PreventUpdate

    patient_data_1 = df[df['Name'].str.contains(patient_name_1, case=False, na=False)]
    patient_data_2 = df[df['Name'].str.contains(patient_name_2, case=False, na=False)]

    if patient_data_1.empty or patient_data_2.empty:
        return html.Div("One or both patients not found."), go.Figure(), go.Figure()

    comparison_text = [
        html.H3(f"Comparison between {patient_name_1} and {patient_name_2}"),
        html.Div([
            html.Div([
                html.P(f"Age: {patient_data_1.iloc[0]['Age']}"),
                html.P(f"Blood Type: {patient_data_1.iloc[0]['Blood Type']}"),
                html.P(f"Medical Condition: {patient_data_1.iloc[0]['Medical Condition']}")
            ], style={'width': '50%', 'display': 'inline-block'}),
            html.Div([
                html.P(f"Age: {patient_data_2.iloc[0]['Age']}"),
                html.P(f"Blood Type: {patient_data_2.iloc[0]['Blood Type']}"),
                html.P(f"Medical Condition: {patient_data_2.iloc[0]['Medical Condition']}")
            ], style={'width': '50%', 'display': 'inline-block'})
        ])
    ]

    # Generate comparison timeline
    treatments_1 = generate_random_vital_signs(30)
    treatments_2 = generate_random_vital_signs(30)
    
    fig_comparison_timeline = go.Figure()
    fig_comparison_timeline.add_trace(go.Scatter(x=treatments_1['Date'], y=treatments_1['Heart Rate'], name=f'{patient_name_1} Heart Rate'))
    fig_comparison_timeline.add_trace(go.Scatter(x=treatments_2['Date'], y=treatments_2['Heart Rate'], name=f'{patient_name_2} Heart Rate'))
    fig_comparison_timeline.update_layout(title='Heart Rate Comparison', xaxis_title='Date', yaxis_title='Heart Rate')

    # Generate comparison vital signs
    fig_comparison_vital_signs = go.Figure()
    for metric in ['Systolic BP', 'Diastolic BP', 'Temperature']:
        fig_comparison_vital_signs.add_trace(go.Scatter(x=treatments_1['Date'], y=treatments_1[metric], name=f'{patient_name_1} {metric}'))
        fig_comparison_vital_signs.add_trace(go.Scatter(x=treatments_2['Date'], y=treatments_2[metric], name=f'{patient_name_2} {metric}'))
    fig_comparison_vital_signs.update_layout(title='Vital Signs Comparison', xaxis_title='Date', yaxis_title='Value')

    return html.Div(comparison_text), fig_comparison_timeline, fig_comparison_vital_signs

# Callback for searching patients by blood group
@app.callback(
    Output('blood-group-results', 'children'),
    [Input('blood-group-dropdown', 'value')]
)
def search_patients_by_blood_group(selected_blood_group):
    if not selected_blood_group:
        raise PreventUpdate

    filtered_patients = df[df['Blood Type'] == selected_blood_group]
    if filtered_patients.empty:
        return html.Div("No patients found with this blood group.")
    
    patient_list = [html.P(f"Name: {row['Name']}, Age: {row['Age']}, Hospital: {row['Hospital']}") for _, row in filtered_patients.iterrows()]
    return html.Div([
        html.H4(f"Patients with Blood Group {selected_blood_group}:"),
        html.Div(patient_list)
    ])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)