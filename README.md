# Back-end/Data scientist Challenge - Pickcells

## X-ray analysis and pneumonia detection:

ScanHealth is a web application that offers free pneumonia diagnosis. The client only needs to upload an x-ray photo of the chest region and wait a few moments for the diagnosis. No need to worry, ScanHealth will analyze the x-ray with discretion and guarantee the patient's anonymity.

The web application was developed as part of the challenge proposed by Pickcells® for the Back-end + Data science job.

## Build

The project was developed in <i>Python3.6</i>. Docker and Docker-compose are necessary for the next steps. After cloning the repository to the server, change your working directory to the project's main folder.

```
  cd scan-health
```

If you'd like to use your own domain name, in the <i>nginx.conf</i> file, replace ```<your_ip_address>``` with the desired domain/Public IP (by default, the file has <i>localhost</i>):

```
server {
    server_name <your_ip_address>;
    listen 80;

    location / {
        include uwsgi_params;
        uwsgi_pass flask:8080;
    }

}

```

Build, start and attach containers for the service:

```
  docker-compose up --build
```
