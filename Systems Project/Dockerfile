# Dockerfile for setting up environment for both web interface and REST API
# They could be two seperate files however this is the easiest way to make sure both components will work
# in a docker container
FROM pytorch/pytorch

WORKDIR /workspaces/

ADD app.py /workspaces/app.py
ADD api.py /workspaces/api.py
ADD requirements.txt /workspaces/requirements.txt
ADD models/prototypical_network_5-way_5-shot_last_epoch /workspaces/models/prototypical_network_5-way_5-shot_last_epoch

RUN pip install -r requirements.txt

ENV MODEL_CHECKPOINT_PATH=/workspaces/models/prototypical_network_5-way_5-shot_last_epoch

# RUN echo ${MODEL_CHECKPOINT_PATH}