
import dash
from dash import html
from dash.dependencies import Input, Output
from dash_handsontable import HotTable


app = dash.Dash(__name__)
app.title = 'Dash Handsontable'

app.layout = html.Div([
    HotTable(
        id='table',
        data=[
            ['', 'Tesla', 'Volvo', 'Toyota', 'Ford'],
            ['2019', 10, 11, 12, 13],
            ['2020', 20, 11, 14, 13],
            ['2021', 30, 15, 12, 13]
        ],
        rowHeaders=False,
        colHeaders=False,
        height='auto',
        autoWrapRow=True,
        autoWrapCol=True,
        contextMenu=True,
        licenseKey='non-commercial-and-evaluation'
    ),
    html.Button("Export to CSV", id="export-btn")
])

@app.callback(
    Output('table', 'exportData'),
    Output('table', 'exportDataParams'),
    [Input('export-btn', 'n_clicks')]
)
def export_data(n_clicks):
    if n_clicks:
        return True, {"filename": "export.csv"}
    return dash.no_update


if __name__ == '__main__':
    app.run_server(debug=True)