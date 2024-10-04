import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import numpy as np
from dash.exceptions import PreventUpdate
from datetime import datetime, timedelta
from fpdf import FPDF
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load the dataset
df = pd.read_csv('healthcare_dataset.csv')


# Check if 'Length of Stay' column exists, if not, create it with random values
if 'Length of Stay' not in df.columns:
    print("'Length of Stay' column not found. Creating it with random values.")
    df['Length of Stay'] = np.random.randint(1, 30, size=len(df))  # Random values between 1 and 30 days

# Create mapping dictionaries for human-readable labels
gender_map = {0: 'Male', 1: 'Female'}
blood_type_map = dict(enumerate(df['Blood Type'].unique()))
medical_condition_map = dict(enumerate(df['Medical Condition'].unique()))

# Preprocess the data for ML tasks
le = LabelEncoder()
df['Gender_encoded'] = le.fit_transform(df['Gender'])
df['Blood Type_encoded'] = le.fit_transform(df['Blood Type'])
df['Medical Condition_encoded'] = le.fit_transform(df['Medical Condition'])

# Prepare features for ML models
features = ['Age', 'Gender_encoded', 'Blood Type_encoded', 'Medical Condition_encoded']
X = df[features]
y = df['Length of Stay']

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Train a Random Forest Regressor for Length of Stay prediction
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Save the model
joblib.dump(rf_model, 'random_forest_los_model.joblib')

# Perform K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# List of photo URLs (ensure these are valid paths)
photo_urls = [
    '/assets/image-generated-by-ai-portrait-handsome-asian-man_803126-1182.jpg',
    '/assets/images (4).jpg',
    '/assets/images (5).jpg',
    '/assets/images (6).jpg',
    '/assets/sampphoto.jpg'
]

# Initialize the Dash app
app = dash.Dash(__name__)
app.title = "Advanced Medical Dashboard with ML"

# App layout
app.layout = html.Div([
    html.Link(
        rel='stylesheet',
        href='https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap'
    ),
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
        
        html.Div([
            html.Button('Download Patient Data as PDF', id='download-pdf-button', className="button"),
            dcc.Download(id='download-pdf-data'),
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
        ]),

        html.H2("Patient analysis", className="header"),
        
        html.Div([
            html.H3("Predict Length of Stay"),
            dcc.Input(id='age-input', type='number', placeholder='Enter Age', className="input"),
            dcc.Dropdown(
                id='gender-input',
                options=[{'label': 'Male', 'value': 'Male'}, {'label': 'Female', 'value': 'Female'}],
                placeholder='Select Gender',
                className="input"
            ),
            dcc.Dropdown(
                id='blood-type-input',
                options=[{'label': bg, 'value': bg} for bg in df['Blood Type'].unique()],
                placeholder='Select Blood Type',
                className="input"
            ),
            dcc.Dropdown(
                id='condition-input',
                options=[{'label': cond, 'value': cond} for cond in df['Medical Condition'].unique()],
                placeholder='Select Medical Condition',
                className="input"
            ),
            html.Button('Predict', id='predict-button', className="button"),
            html.Div(id='prediction-output', className="card")
        ], className="card"),
        
        html.Div([
            html.H3("Patient Clustering"),
            dcc.Graph(id='cluster-graph', className="card")
        ]),
        
        html.Div([
            html.H3("Model Performance"),
            html.Div(id='model-performance', className="card")
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
        {'Date': '2023-05-20', 'Treatment': 'Condition Stabilized'},
    ]
    timeline_fig = go.Figure(
        data=[go.Scatter(
            x=[t['Date'] for t in treatments],
            y=[i for i in range(len(treatments))],
            mode='markers+lines',
            text=[t['Treatment'] for t in treatments],
            marker=dict(size=10, color='blue')
        )]
    )
    timeline_fig.update_layout(title='Patient Treatment Timeline', xaxis_title='Date', yaxis_title='Treatment Progress', yaxis=dict(showticklabels=False))

    # Example medical alerts
    medical_alerts = [
        html.H4("Medical Alerts"),
        html.P("Recent condition deteriorating. Immediate follow-up required."),
        html.P("Respiratory rate abnormal in last checkup.")
    ]

    return patient_info_text, timeline_fig, fig_vital_signs, medical_alerts

# Callback to download patient data as CSV
@app.callback(
    Output('download-csv-data', 'data'),
    Input('download-csv-button', 'n_clicks'),
    State('patient-name-dropdown', 'value'),
    prevent_initial_call=True
)
def download_patient_data_as_csv(n_clicks, patient_name):
    if not patient_name:
        raise PreventUpdate

    patient_data = df[df['Name'].str.contains(patient_name, case=False, na=False)]
    if patient_data.empty:
        raise PreventUpdate

    return dcc.send_data_frame(patient_data.to_csv, f"{patient_name}_patient_data.csv")

# Function to create PDF from patient data
def create_pdf(patient_data):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font('Arial', 'B', 12)
    pdf.cell(200, 10, txt="Patient Data", ln=True, align='C')

    pdf.set_font('Arial', '', 10)
    for col, value in patient_data.items():
        pdf.cell(200, 10, txt=f"{col}: {value}", ln=True)

    return pdf.output(dest='S').encode('latin1')

# Callback to download patient data as PDF
@app.callback(
    Output('download-pdf-data', 'data'),
    Input('download-pdf-button', 'n_clicks'),
    State('patient-name-dropdown', 'value'),
    prevent_initial_call=True
)
def download_patient_data_as_pdf(n_clicks, patient_name):
    if not patient_name:
        raise PreventUpdate

    patient_data = df[df['Name'].str.contains(patient_name, case=False, na=False)].iloc[0]
    pdf_content = create_pdf(patient_data.to_dict())

    return dcc.send_bytes(pdf_content, f"{patient_name}_patient_data.pdf")

# Callback to compare two patients
@app.callback(
    [Output('comparison-info', 'children'),
     Output('comparison-timeline', 'figure'),
     Output('comparison-vital-signs', 'figure')],
    Input('compare-button', 'n_clicks'),
    State('patient-name-1', 'value'),
    State('patient-name-2', 'value')
)
def compare_patients(n_clicks, patient_name_1, patient_name_2):
    if not patient_name_1 or not patient_name_2:
        raise PreventUpdate

    patient_data_1 = df[df['Name'].str.contains(patient_name_1, case=False, na=False)]
    patient_data_2 = df[df['Name'].str.contains(patient_name_2, case=False, na=False)]

    if patient_data_1.empty or patient_data_2.empty:
        return html.Div("No data found for one or both patients."), {}, {}

    patient_1 = patient_data_1.iloc[0].to_dict()
    patient_2 = patient_data_2.iloc[0].to_dict()

    comparison_info = html.Div([
        html.H3(f"Comparing {patient_1['Name']} and {patient_2['Name']}"),
        html.P(f"{patient_1['Name']} - Age: {patient_1['Age']}, Condition: {patient_1['Medical Condition']}"),
        html.P(f"{patient_2['Name']} - Age: {patient_2['Age']}, Condition: {patient_2['Medical Condition']}")
    ])

    treatments = [
        {'Date': '2023-01-15', 'Patient': patient_1['Name'], 'Treatment': 'Initial Diagnosis'},
        {'Date': '2023-02-01', 'Patient': patient_2['Name'], 'Treatment': 'Initial Diagnosis'},
        {'Date': '2023-02-15', 'Patient': patient_1['Name'], 'Treatment': 'Medication Started'},
        {'Date': '2023-03-10', 'Patient': patient_2['Name'], 'Treatment': 'Medication Started'},
        {'Date': '2023-03-20', 'Patient': patient_1['Name'], 'Treatment': 'Follow-up Checkup'},
        {'Date': '2023-04-05', 'Patient': patient_2['Name'], 'Treatment': 'Therapy Session'},
    ]
    timeline_fig = go.Figure(
        data=[go.Scatter(
            x=[t['Date'] for t in treatments],
            y=[i for i in range(len(treatments))],
            mode='markers+lines',
            text=[f"{t['Patient']}: {t['Treatment']}" for t in treatments],
            marker=dict(size=10, color='green')
        )]
    )
    timeline_fig.update_layout(title='Treatment Timeline Comparison', xaxis_title='Date', yaxis_title='Treatment', yaxis=dict(showticklabels=False))

    vital_signs_data_1 = generate_random_vital_signs(30)
    vital_signs_data_2 = generate_random_vital_signs(30)

    comparison_fig = go.Figure()
    for metric in ['Heart Rate', 'Systolic BP', 'Diastolic BP', 'Temperature']:
        comparison_fig.add_trace(go.Scatter(x=vital_signs_data_1['Date'], y=vital_signs_data_1[metric], mode='lines', name=f"{patient_1['Name']} - {metric}"))
        comparison_fig.add_trace(go.Scatter(x=vital_signs_data_2['Date'], y=vital_signs_data_2[metric], mode='lines', name=f"{patient_2['Name']} - {metric}"))

    comparison_fig.update_layout(title='Vital Signs Comparison', xaxis_title='Date', yaxis_title='Values')

    return comparison_info, timeline_fig, comparison_fig

# Callback to display patients by blood group
@app.callback(
    Output('blood-group-results', 'children'),
    Input('blood-group-dropdown', 'value')
)
def search_by_blood_group(blood_group):
    if not blood_group:
        raise PreventUpdate

    patients_with_blood_group = df[df['Blood Type'] == blood_group]

    if patients_with_blood_group.empty:
        return html.Div("No patients found with the selected blood group.")

    patient_details = [
        html.H3(f"Patients with Blood Group {blood_group}"),
        html.Ul([html.Li(f"{row['Name']} - Age: {row['Age']}, Condition: {row['Medical Condition']}") for index, row in patients_with_blood_group.iterrows()])
    ]

    return patient_details

# Callback to predict Length of Stay
@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    [State('age-input', 'value'),
     State('gender-input', 'value'),
     State('blood-type-input', 'value'),
     State('condition-input', 'value')]
)

# Additional function to update model performance display



# Callback for clustering visualization
@app.callback(
    Output('cluster-graph', 'figure'),
    Input('blood-group-dropdown', 'value')  # We'll use this dropdown to trigger the graph update
)
def update_cluster_graph(blood_group):
    fig = px.scatter(df, x='Age', y='Length of Stay', color='Cluster',
                     hover_data=['Name', 'Blood Type', 'Medical Condition'],
                     title='Patient Clustering based on Age and Length of Stay')
    return fig

@app.callback(
    Output('model-performance', 'children'),
    Input('submit-button', 'n_clicks')
)
def update_model_performance(n_clicks):
    if n_clicks is None:
        raise PreventUpdate

    # Calculate performance metrics
    y_pred = rf_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Changed display format
    return [
        html.P(f"Model Performance Metrics:", style={'font-weight': 'bold'}),
        html.P(f"Mean Squared Error (MSE): {mse:.2f}"),
        html.P(f"R-squared (R²): {r2:.2f}")
    ]

if __name__ == '__main__':
    app.run_server(debug=True)
                                                