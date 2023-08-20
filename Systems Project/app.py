from dash import Dash, dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import requests

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(
    __name__,
    external_stylesheets=external_stylesheets,
    suppress_callback_exceptions=True
)

def create_upload_component(component_div_id, upload_div_id):
    '''
    TODO: FILL THS IN
    '''
    return html.Div(
                dcc.Upload(
                    id=upload_div_id,
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
                id=component_div_id
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
                    dcc.Input(id=f"label_input_class{idx}", type="text", placeholder="Enter Class ID"),
                    create_upload_component(f'class{idx}-card', f'upload-images-{idx}'),
                    dbc.Row(id=f'support-set-images-upload-{idx}'),
                ]
            ),
        ],
        style={"display": "none", "width": "90%"},
        id=f'support-class-card-{idx}'
    )

def display_images(image_str_list):
    '''
    Function for rendering uploaded images.
    '''
    return [
        html.Div(
            [
                html.Img(
                    src=image_str,
                    style = {
                        'height': '60px',
                        'width': '120px',
                        'float': 'left',
                        'position': 'relative',
                        'padding': 5,
                    }
                )
                for image_str in image_str_list
            ]
        )
    ]

app.layout = html.Div(
    children=[
        dcc.Store(id='support-labels', storage_type='session'),
        dcc.Store(id='support-images-dict', storage_type='session'),
        dcc.Store(id='query-images-list', storage_type='session'),
        html.Header("Humpback Whale Identification with Few Shot Learning",
                    style={'text-align': 'center', 'font-weight': 'bold'}),
        html.Div(id='classes_selected_div', style={'display': 'none'}),
        html.Div(
            children=[
                dbc.Row(
                    children=[
                        html.H4('Upload Support Images'),
                        dbc.Col(
                            html.Div(children=[create_support_class_card(i) for i in range(5)]),
                            style={'max-width': '95%'}
                        ),
                        dbc.Col(
                            html.Button('Add Support Class', id='add-class', n_clicks=0)
                        ),
                    ]
                ),
                dbc.Row(
                    dbc.Col(
                        children=[
                            html.H4('Upload Query Images'),
                            create_upload_component(
                                'query-images-div', 
                                'upload-query-images'
                            ),
                            dbc.Row(id='query-images-output')
                        ]
                    )
                ),
                dbc.Row(dbc.Button('Classify query images', id='classify-button')),
                dbc.Row(id='classify-results')
            ]
        )
    ]
)

@callback(
    [Output(component_id=f'support-class-card-{i}', component_property='style') for i in range(5)],
    Input('add-class', 'n_clicks')
)
def add_support_class(n_clicks):
    '''
    Function for rendering class cards. 
    The n_clicks for this button will be how many class cards that are rendered.
    '''
    num_support_class = n_clicks if n_clicks <= 5 else 5

    visible = [{'display': 'block'} for _ in range(num_support_class)]
    if num_support_class < 5:
        invisible = [{'display': 'none'} for _ in range(5-num_support_class)]
        styles = visible + invisible
    else:
        styles = visible
    return styles

@callback(
    Output("support-labels", "data"),
    [Input(f"label_input_class{i}", "value") for i in range(5)])
def update_support_labels(*vals):
    
    '''TODO: FILL THIS IN'''
    return [v for v in vals if v is not None]

@callback(
        [Output(f'support-set-images-upload-{idx}', 'children') for idx in range(5)],
        Output('support-images-dict', 'data'),
        [Input(f'upload-images-{idx}', 'contents') for idx in range(5)])
def update_support_set_images(*contents):
    '''
    TODO: FILL THIS IN
    '''
    support_set_dict = dict(enumerate(contents))
    num_support_classes = len([k for k,v in support_set_dict.items() if v is not None])

    image_displays = [
        display_images(image_str_list)
        for image_str_list in support_set_dict.values()
        if image_str_list is not None
    ]
    if num_support_classes < 5:
        image_displays += [
            html.Div(style={'display': 'none'})
            for _ in range(5-num_support_classes)
        ]

    # Hack for unraveling List of List into tuple
    items = (
        image_displays[0],
        image_displays[1],
        image_displays[2],
        image_displays[3],
        image_displays[4],
        support_set_dict
    )

    return items

@callback(
        Output("query-images-output", "children"),
        Output('query-images-list', 'data'),
        Input('upload-query-images', 'contents'))
def update_query_set_images(image_b64_str_list):
    '''
    TODO: FILL THIS IN
    '''
    displays = []
    if image_b64_str_list is not None:
        displays = display_images(image_b64_str_list)

    return displays, image_b64_str_list

@callback(
    Output('classify-results', 'children'),
    Input('classify-button', 'n_clicks'),
    State('support-labels', 'data'),
    State('support-images-dict', 'data'),
    State('query-images-list', 'data')
)
def get_classify_results(_, support_labels, support_images, query_images):
    '''
    TODO: FILL ME IN
    '''
    if support_images is not None and support_labels is not None and query_images is not None:
        support_images = {k:v for k,v in support_images.items() if v is not None}
        support_images = {support_labels[int(idx)]: v for idx, v in support_images.items()}

        post_body = {
            "support_set_labels": support_labels,
            "support_set_images": support_images,
            "query_set_images": query_images
        }

        resp = requests.post('http://127.0.0.1:8000/classify', json = post_body, timeout=1000)
        print(resp.json())

if __name__ == '__main__':
    app.run(debug=True)
