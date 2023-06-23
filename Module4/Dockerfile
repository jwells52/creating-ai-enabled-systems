FROM python:3.10

# Creating working directory
WORKDIR /assignment_code/

# Copy necessary files to working directory
COPY requirements.txt .
COPY jwells52_assignment3.py .
COPY Musical_instruments_reviews.csv .

# Install required packages
RUN pip install -r requirements.txt

# Download NLTK package data
RUN python -m nltk.downloader punkt
RUN python -m nltk.downloader wordnet


# Run python script on start-up
CMD ["jwells52_assignment3.py"]
ENTRYPOINT [ "python" ]

