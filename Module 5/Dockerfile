FROM python:3.10

# Creating working directory
WORKDIR /assignment_code/

RUN mkdir -p /output
# Define docker build arguments for setting environment variables
# in the image
ARG LOG_PATH=/output/
ARG USED_CAR_DATA_PATH=/assignment_code/cars.csv

# Define necessary environment variables for the
# python script
ENV LOG_PATH=$LOG_PATH
ENV USED_CAR_DATA_PATH=$USED_CAR_DATA_PATH

# Copy necessary files to working directory
COPY requirements.txt .
COPY jwells52_assignment4.py .
COPY cars.csv .

# Install required packages
RUN pip install -r requirements.txt

# Run python script on start-up
CMD ["jwells52_assignment4.py"]
ENTRYPOINT [ "python" ]

