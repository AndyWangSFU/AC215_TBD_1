
# AC215 - Milestone5 - “Multimodal Fake News Detector”
After completions of building a robust ML Pipeline in our previous milestone we have built a backend api service and frontend app. This will be our user-facing application that ties together the various components built in previous milestones.

**Team Members**
Kyle Ke, Boshen Yan, Fuchen Li, Zihan Wang, Qassi Gaba

**Group Name**
TBD_1


Project Organization
--------------------
    
    ├── LICENSE
    ├── README.md
    ├── references                  <- Reference materials such as papers
    ├── presentations               <- Folder containing our midterm presentation
    │   └── midterm.pdf
    ├── requirements.txt
    ├── src
    │   ├── preprocessing          <- Code for data processing. See previous milestone for content details.
    │   │   ├── ...
    │   ├── model_compression       <- Code for model compression to ensure efficient run. See previous milestone for content details
    │   │   ├── ...
    │   └── training                 <- Entire code for model training. See previous milestone for content details
    │       ├── ...
    │   ├── workflow                 <- Scripts for automating data collection, preprocessing, modeling. See previous milestone for content details
    │       ├── ...
    │   ├── api-service              <- Code for app backend APIs
    │   │   ├── api
    │   │   ├── Dockerfile
    │   │   ├── docker-entrypoint.sh
    │   │   ├── requirements.txt
    │   │   ├── run_docker.sh
    │   ├── frontend-react           <- Code for App frontend
    │   │   ├── conf/conf.d
    │   │   │   ├── default.conf
    │   │   ├── public
    │   │   │   ├── favicon.ico
    │   │   │   ├── index.html
    │   │   │   ├── manifest.json
    │   │   ├── src
    │   │   │   ├── app
    │   │   │   ├── common
    │   │   │   ├── components
    │   │   │   ├── services
    │   │   │   ├── index.css
    │   │   │   ├── index.js
    │   │   ├── .env.development
    │   │   ├── .env.production
    │   │   ├── .eslintcache
    │   │   ├── .gitignore
    │   │   ├── Dockerfile
    │   │   ├── Dockerfile.dev
    │   │   ├── docker-shell.bat
    │   │   ├── docker-shell.sh
    │   │   ├── package.json
    │   │   ├── yarn.lock
    ├── deployment                  <- Code for app deployment to GCP
    │   ├── deploy-create-instance.yml
    │   ├── deploy-docker-images.yml
    │   ├── deploy-provision-instance.yml
    │   ├── deploy-setup-containers.yml
    │   ├── deploy-setup-webserver.yml
    │   ├── inventory.yml
    │   ├── Dockerfile
    │   ├── docker-entrypoint.sh
    │   └── docker-shell.sh
    │   └── nginx-conf/nginx

------------

**Application Design**

Before we start implementing the app we built a detailed design document outlining the application’s architecture. We built a Solution Architecture abd Technical Architecture to ensure all our components work together.

Here is our Solution Architecture:

<img width="1264" alt="Screenshot 2023-10-04 at 7 19 39 PM" src= "https://github.com/AndyWangSFU/AC215_TBD_1/assets/112672824/c022ad7a-57ed-4bb2-975a-798e71e6e7f1)">
Figure 1: Solution architecture of project


Here is our Technical Architecture:
<img width="1264" alt="Screenshot 2023-10-04 at 7 19 39 PM" src= "https://github.com/AndyWangSFU/AC215_TBD_1/assets/112672824/f8662812-f84c-4f20-a878-9683353c67cf">
Figure 2: Technical architecture of project


**Backend API**

We built backend api service using fast API to expose model functionality to the frontend. We also added apis that will help the frontend display some key information about the model and data. 

<img width="1264" alt="Screenshot 2023-10-04 at 7 19 39 PM" src= "https://github.com/AndyWangSFU/AC215_TBD_1/assets/112672824/7d71f975-77c9-4014-8545-d2c1f6b9cb9d">

Figure 3: List of APIs


**Frontend** 

A user friendly React app was built to identify fake news using deep learning model from the backend. Using the app a user can take upload the main image and title of a news article. The app will send the image and text to the backend api to get prediction results on the fake risk (likelihood) of the particular news article.

Here are some screenshots of our app:
<img width="1264" alt="Screenshot 2023-10-04 at 7 19 39 PM" src= "https://github.com/AndyWangSFU/AC215_TBD_1/assets/112672824/1db880ab-0262-4bc3-b977-1e7fa720bee4">

Figure 4: screenshot of frontend UI

**Deployment**

We used Ansible to create, provision, and deploy our frontend and backend to GCP in an automated fashion. Ansible helps us manage infrastructure as code and this is very useful to keep track of our app infrastructure as code in GitHub. It helps use setup deployments in a very automated way.

Here is our deployed app on a single VM in GCP:
<img width="1264" alt="Screenshot 2023-10-04 at 7 19 39 PM" src= "https://github.com/AndyWangSFU/AC215_TBD_1/assets/112672824/0df72a19-13dc-4309-a0ee-7ce20c915a78">

Figure 5: app deployed to a single VM in GCP

### Code Structure

The following are the folders from the previous milestones:
```
- preprocessing
- model_compression
- training
- workflow

```

**API Service Container**
This container has all the python files to run and expose the backend apis.

To run the container locally:
- Open a terminal and go to the location where `AC215_TBD_1/src/api-service`
- Run `sh docker-shell.sh`
- Once inside the docker container run `uvicorn_server`
- To view and test APIs go to `http://localhost:9000/docs`

**Frontend Container**
This container contains all the files to develop and build a react app. There are dockerfiles for both development and production.

To run the container locally:
- Open a terminal and go to the location where `AC215_TBD_1/src/frontend`
- Run `sh docker-shell.sh`
- If running the container for the first time, run `yarn install`
- Once inside the docker container run `yarn start`
- Go to `http://localhost:3000` to access the app locally


**Deployment Container**
This container helps manage building and deploying all our app containers. The deployment is to GCP and all docker images go to Google Container Registry (GCR). 

To run the container locally:
- Open a terminal and go to the location where `AC215_TBD_1/src/deployment`
- Run `sh docker-shell.sh`
- Build and Push Docker Containers to GCR

```
ansible-playbook deploy-docker-images.yml -i inventory.yml
```

- Create Compute Instance (VM) Server in GCP
```
ansible-playbook deploy-create-instance.yml -i inventory.yml --extra-vars cluster_state=present
```

- Provision Compute Instance in GCP
Install and setup all the required things for deployment.
```
ansible-playbook deploy-provision-instance.yml -i inventory.yml
```

- Setup Docker Containers in the  Compute Instance
```
ansible-playbook deploy-setup-containers.yml -i inventory.yml
```

- Setup Webserver on the Compute Instance
```
ansible-playbook deploy-setup-webserver.yml -i inventory.yml
```
Once the command runs go to `http://<External IP>/` 

---
