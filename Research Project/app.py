from dash import Dash, html, dcc
import dash_bootstrap_components as dbc

app = Dash(
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)

nav = dbc.Nav(
    [
        dbc.NavLink("Humpback Whale Identification with Few Shot Learning", disabled=True, href="#"),
    ]
)

def create_support_class_card(idx):
    '''
    Dash Card Component for uploading images to a class that will be used in the support set
    '''
    return dbc.Card(
        [
            dbc.CardBody(
                [
                    html.P(
                        f"Upload images of individual whale for class #{idx}",
                        className="card-text",
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                dcc.Input(id=f"classidx_{idx}", type="text", placeholder="Enter Class ID")
                            ),
                            dbc.Col(
                                children=[
                                    dcc.Upload(
                                        id=f'upload-data-{idx}',
                                        children=html.Div(
                                            [
                                                'Drag and Drop or ',
                                                html.A('Select Files')
                                            ]
                                        ),
                                        style={
                                            'width': '100%',
                                            'height': '60px',
                                            'lineHeight': '60px',
                                            'borderWidth': '1px',
                                            'borderStyle': 'dashed',
                                            'borderRadius': '5px',
                                            'textAlign': 'center',
                                            'margin': '10px'
                                        },
                                        # Allow multiple files to be uploaded
                                        multiple=True
                                    ),
                                    html.Div(id=f'support-set-images-upload-{idx}'),
                                ]
                            )
                        ]
                    )
                ]
            ),
        ],
        # style={"width": "18rem"},
    )


app.layout = html.Div(
    children=[
        nav,
        html.Div(
            children=[
                html.H4("Support Set", className="card-title"),
                dbc.Row(
                    children=[
                        dbc.Col(create_support_class_card(idx))
                        for idx in range(5)
                    ]
                )
            ]
        )
    ]
)

if __name__ == '__main__':
    app.run(debug=True)