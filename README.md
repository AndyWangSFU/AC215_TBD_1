
# AC215 - Milestone5 - “Multimodal Fake News Detector”
After completions of building a robust ML Pipeline in our previous milestone we have built a backend api service and frontend app. This will be our user-facing application that ties together the various components built in previous milestones.

**Team Members**
Kyle Ke, Boshen Yan, Fuchen Li, Zihan Wang, Qassi Gaba

**Group Name**
TBD_1


Project Organization
--------------------
.
├── LICENSE
├── README.md
├── references                  <- Reference materials such as papers
├── presentations               <- Folder containing your midterm presentation
│   └── midterm.pdf
├── requirements.txt
├── src
│   ├── preprocessing          <- Scripts for dataset creation
│   │   ├── ...
│   ├── model_compression       <- Code for data processing
│   │   ├── ...
│   └── training                 <- Model training, evaluation, and prediction code
│       ├── ...
│   ├── workflow                 <- Scripts for automating data collection, preprocessing, modeling
│       ├── ...
│   ├── api-service              <- Code for App backend APIs
│   │   ├── api
│   │   ├── Dockerfile
│   │   ├── docker-entrypoint.sh
│   │   ├── docker-shell.sh
│   │   ├── Pipfile
│   │   ├── Pipfile.lock
│   ├── frontend-react           <- Code for App frontend
│   │   ├── conf
│   │   │   ├── conf.d
│   │   │   │   ├── default.conf
│   │   ├── public
│   │   │   ├── favicon.ico
│   │   │   ├── index.html
│   │   │   ├── manifest.json
├── deployment                  <- Code for App deployment to GCP
│   ├── deploy-create-instance.yml
│   ├── deploy-docker-images.yml
│   ├── deploy-provision-instance.yml
│   ├── deploy-setup-containers.yml
│   ├── deploy-setup-webserver.yml
│   ├── inventory.yml
│   ├── Dockerfile
│   ├── docker-entrypoint.sh
│   └── docker-shell.sh

------------

**Application Design**

Before we start implementing the app we built a detailed design document outlining the application’s architecture. We built a Solution Architecture abd Technical Architecture to ensure all our components work together.

Here is our Solution Architecture:
<img src="![WhatsApp Image 2023-11-20 at 16 56 45](https://github.com/AndyWangSFU/AC215_TBD_1/assets/112672824/95425b32-0c62-442f-b9b5-3a9c1a66c426)"  width="800">

![Uploading WhatsApp Image 2023-11-20 at 16.56.45.jpeg…]()


Here is our Technical Architecture:
<img src="images/technical-arch.png"  width="800">


**Backend API**

We built backend api service using fast API to expose model functionality to the frontend. We also added apis that will help the frontend display some key information about the model and data. 

<img src="images/api-list.png"  width="800">

**Frontend**

A user friendly React app was built to identify fake news using deep learning model from the backend. Using the app a user can take upload the main image and title of a news article. The app will send the image and text to the backend api to get prediction results on the fake risk (likelihood) of the particular news article.

Here are some screenshots of our app:
<img src="images/frontend-1.png"  width="800">

<img src="images/frontend-2.png"  width="800">

**Deployment**

We used Ansible to create, provision, and deploy our frontend and backend to GCP in an automated fashion. Ansible helps us manage infrastructure as code and this is very useful to keep track of our app infrastructure as code in GitHub. It helps use setup deployments in a very automated way.

Here is our deployed app on a single VM in GCP:
<img src="images/deployment-single-vm.png"  width="800">


### Code Structure

The following are the folders from the previous milestones:
```
- data-collector
- data-processor
- model-training
- model-deploy
- workflow
```

**API Service Container**
This container has all the python files to run and expose thr backend apis.

To run the container locally:
- Open a terminal and go to the location where `AC215_TBD_1/src/api-service`
- Run `sh docker-shell.sh`
- Once inside the docker container run `uvicorn_server`
- To view and test APIs go to `http://localhost:9000/docs`

**Frontend Container**
This container contains all the files to develop and build a react app. There are dockerfiles for both development and production

To run the container locally:
- Open a terminal and go to the location where `AC215_TBD_1/src/frontend`
- Run `sh docker-shell.sh`
- If running the container for the first time, run `yarn install`
- Once inside the docker container run `yarn start`
- Go to `http://localhost:3000` to access the app locally


**Deployment Container**
This container helps manage building and deployeing all our app containers. The deployment is to GCP and all docker images go to GCR. 

To run the container locally:
- Open a terminal and go to the location where `AC215_TBD_1/src/deployment`
- Run `sh docker-shell.sh`
- Build and Push Docker Containers to GCR (Google Container Registry)
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
