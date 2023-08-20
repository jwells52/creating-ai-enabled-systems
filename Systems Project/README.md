# Systems Project

## Overview
This folder contains the code for running my System Project, for details on this system project please reference this [paper](./systems_project_paper.ipynb).

This system project is essentially just a web interface and REST API. Right now, the only way to run this project is locally, you will need to run both the front-end and back-end in order to have a fully functional system.

## Steps for starting the system
### Prerequisites
1. Make this folder your working directory: `cd Systems\ Project`
2. Install the required packages: `pip install -r requirements`

#### Start the REST API
1. Run the command to start the REST API: `uvicorn api:app`

### Start the Web Interface
1. Run the command to the start the Web Interface: `python app.py`
2. In your browser, go to `localhost:8050`

## Example classes to use for testing out the system