FROM python:3.6.10-stretch

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app 
ADD . /app

# Install the dependencies
RUN pip install -r requirements.txt

# RUN apt-get install -y libsm6 libxext6 libxrender-dev

# run the command to start uWSGI
CMD ["uwsgi", "app.ini"]
