FROM ubuntu:18.04

RUN apt-get update

RUN apt-get install -y build-essential python3.6 python3.6-dev python3-pip python3.6-venv
RUN apt-get install -y git

# update pip
RUN python3.6 -m pip install pip --upgrade
RUN python3.6 -m pip install wheel

# We copy just the requirements.txt first to leverage Docker cache
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt

RUN apt-get install -y libsm6 libxext6 libxrender-dev

ENTRYPOINT [ "uwsgi" ]

CMD [ "dev.ini" ]
