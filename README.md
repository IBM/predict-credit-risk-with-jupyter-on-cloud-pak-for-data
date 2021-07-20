# Predict Credit Risk With Jupyter on Cloud Pak for data

In this Code Pattern, we will go through the process of exploring a data set and building a predictive model that can be used to determine the likelihood of a credit loan default. For this use case, the machine learning model we are building is a classification model that will return a prediction of 'Risk' (the features of the loan applicant predict that there is a good chance of default on the loan) or 'No Risk' (the applicant's inputs predict that the loan will be paid off). The approach we will take in this lab is to use some fairly popular libraries / frameworks to build the model in Python using a Jupyter notebook.

After building the model, we will go through the process of deploying a machine learning model so it can be used by others. Deploying a model allows us to put a model into production, so that data can be passed to it to return a prediction. The deployment will result in an endpoint that makes the model available for wider use in applications and to make business decisions. There are several types of deployments available ([depending on the model framework used](https://www.ibm.com/support/producthub/icpdata/docs/content/SSQNUZ_current/wsj/analyze-data/pm_service_supported_frameworks.html)), of which we will explore:

* Online Deployments - Creates an endpoint to generate a score or prediction in real time.
* Batch Deployments - Creates an endpoint to schedule the processing of bulk data to return predictions.

When you have completed this code pattern, you will understand how to:

* Use a Jupyter notebook in Watson Studio to build a Machine Learning model.
* Deploy the model using Watson Machine Learning.
* Test the model by scoring on a REST endpoint for the deployed model.
* Test the model in batch mode.
* Build and use a Web Python Flask app to demonstrate the use of the model.
* Use IBM Cloud Pak for Data to orchestrate the entire process on a consistent platform.

![architecture diagram](doc/source/images/architecture.png)

## Flow

1. User loads the Jupyter notebook into the Cloud Pak for Data platform.
1. [Credit Risk data set](https://raw.githubusercontent.com/IBM/predict-credit-risk-with-jupyter-on-cloud-pak-for-data/main/data/german_credit_data.csv)  is loaded into the Jupyter Notebook, either directly from the github repo, or as Virtualized Data after following the [Data Virtualization Tutorial](https://developer.ibm.com/tutorials/virtualizing-db2-warehouse-data-with-data-virtualization) from the [IBM Cloud Pak for Data Learning Path](https://developer.ibm.com/series/cloud-pak-for-data-learning-path/).
1. Preprocess the data, build machine learning models and save to Watson Machine Learning on Cloud Pak for Data.
1. Deploy a selected machine learning model into production on the Cloud Pak for Data platform and obtain a scoring endpoint.
1. Use the model for credit prediction using a frontend application.

## Included components

* [IBM Cloud Pak for Data](https://www.ibm.com/products/cloud-pak-for-data)
* [Watson Machine Learning Add On for Cloud Pak for Data](https://www.ibm.com/cloud/machine-learning)

## Featured technologies

* [Jupyter Notebooks](https://jupyter.org/): An open-source web application that allows you to create and share documents that contain live code, equations, visualizations, and explanatory text.
* [Pandas](https://pandas.pydata.org/):  An open source library providing high-performance, easy-to-use data structures and data analysis tools for the Python programming language.
* [Seaborn](https://seaborn.pydata.org/): A Python data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics.
* [Spark MLib](https://spark.apache.org/mllib/): Apache Spark's scalable machine learning library.

## Prerequisites

* [IBM Cloud Pak for Data](https://www.ibm.com/analytics/cloud-pak-for-data)
* [Watson Machine Learning Add On for Cloud Pak for Data](https://www.ibm.com/support/producthub/icpdata/docs/content/SSQNUZ_current/wsj/analyze-data/ml-install-overview.html)

Steps:

1. [Create a new Project](#1-create-a-new-project)
1. [Create a Space for Machine Learning Deployments](#2-create-a-space-for-machine-learning-deployments)
1. [Import notebooks to Cloud Pak for Data](#3-import-notebooks-to-cloud-pak-for-data)
1. [Build and Save a model](#4-build-and-save-a-model)
1. [Online Deployment for a Model](#5-online-model-deployment)
   * Create Online Deployment
   * Test model using Cloud Pak for Data tooling
   * (Optional) Test model using cURL
1. [Batch Deployment for a Model](#6-batch-model-deployment)
   * Create Batch Deployment
   * Create and Schedule a Job
1. [Integrate Model to an External Application](#7-integrate-model-to-python-flask-application)

### 1. Create a new project

> Note: If you are following the [IBM Cloud Pak for Data Learning Path](https://developer.ibm.com/series/cloud-pak-for-data-learning-path/) and you have already created a project, you can skip to [Create a Space for Machine Learning Deployments](#2-create-a-space-for-machine-learning-deployments)

In Cloud Pak for Data, we use the concept of a project to collect / organize the resources used to achieve a particular goal (resources to build a solution to a problem). Your project resources can include data, collaborators, and analytic assets like notebooks and models, etc.

* Go the (☰) navigation menu and under the *Projects* section click on *`All Projects`*.

  ![(☰) Menu -> Projects](doc/source/images/menu-projects.png)

* Click on the **`New project`** button on the top right.

  ![Start a new project](doc/source/images/new-project.png)

* Select the `Analytics project` radio button and click the **`Next`** button.

  ![New analytics project](doc/source/images/new-project-type.png)

* Select **`Create an empty project`**.

  ![Create empty project](doc/source/images/cpd-create-empty-project.png)

* Provide a name and optional description for the project and click **`Create`**.

  ![Pick a name](doc/source/images/project-name.png)

### Download the dataset for this experiment and load it into you project.

* Clone this repository:

```bash
git clone https://github.com/IBM/predict-credit-risk-with-jupyter-on-cloud-pak-for-data
cd predict-credit-risk-with-jupyter-on-cloud-pak-for-data
```

* In your project, on the `Assets` tab click the `01/00` icon and the `Load` tab, then either drag the `data/german_credit_data.csv` file from the cloned repository to the window or navigate to it using `browse for files to upload`:

  ![Add data set](doc/source/images/cpd-add-data-set.png)

## 2. Create a Space for Machine Learning Deployments

Cloud Pak for Data uses the concept of `Deployment Spaces` to configure and manage the deployment of a set of related deployable assets. These assets can be data files, machine learning models, etc.

* Go the (☰) navigation menu and click **`Deployments`**.

  ![(☰) Menu -> Analytics deployments](doc/source/images/menu-analytics-deployments.png)

* Click on the **`New deployment space`** button.

  ![Add New deployment space](doc/source/images/new-deployment-space.png)

* We will create an empty deployment space, so click on the **`Create an empty space`** option.

  ![Create empty deployment space](doc/source/images/new-deployment-space-empty.png)

* Give your deployment space a unique name, optional description, then click the **`Create`** button.

  ![Deployment space name](doc/source/images/deployment-space-name.png)

* From the deployment space creation pop up window, click on the **`View new space`** button.

  ![Import project success](doc/source/images/depspace-create-success.png)

### 3. Import notebooks to Cloud Pak for Data

* In your project, either click the `Add to project +` button, and choose `Notebook`, or, if the *Notebooks* section exists,  to the right of *Notebooks* click `New notebook +`:

  ![Add notebook](doc/source/images/wml-1-add-asset.png)

* On the next screen, select the *From URL* tab, give your notebook a *name* and an optional *description*, provide the following URL as the *Notebook URL*, and choose the `Python 3.6` environment as the *Runtime*:

```bash
https://raw.githubusercontent.com/IBM/predict-credit-risk-with-jupyter-on-cloud-pak-for-data/master/notebooks/machinelearning-creditrisk-sparkmlmodel.ipynb
```

  ![Add notebook name and URL](doc/source/images/notebook-add-name-and-url.png)

* When the Jupyter notebook is loaded and the kernel is ready then we can start executing cells.

  ![Notebook loaded](doc/source/images/notebook-loaded.png)

* Repeat the steps above for the other notebook (used for batch scoring):

```bash
https://raw.githubusercontent.com/IBM/predict-credit-risk-with-jupyter-on-cloud-pak-for-data/master/notebooks/machinelearning-creditrisk-batchscoring.ipynb
```

## 4. Build and Save a model

For this part of the exercise we're going to use a Jupyter notebook to create the model. The Jupyter notebook is already included as an asset in the project you imported earlier.

### Open the Jupyter notebook

* Go the (☰) navigation menu and under the *Projects* section click on *`All Projects`*.

  ![(☰) Menu -> Projects](doc/source/images/menu-projects.png)

* Click the project name you created in the pre-work section.

* From your `Project` overview page, click on the *`Assets`* tab to open the assets page where your project assets are stored and organized.

* Scroll down to the `Notebooks` section of the page and click on the pencil icon at the right of the `machinelearning-creditrisk-sparkmlmodel` notebook.

  ![Notebook Open](doc/source/images/mljupyter-open-notebook.png)

* When the Jupyter notebook is loaded and the kernel is ready, we will be ready to start executing it in the next section.

  ![Notebook loaded](doc/source/images/mljupyter-notebook-loaded.png)

### Run the Jupyter notebook

Spend some time looking through the sections of the notebook to get an overview. A notebook is composed of text (markdown or heading) cells and code cells. The markdown cells provide comments on what the code is designed to do.

You will run cells individually by highlighting each cell, then either click the `Run` button at the top of the notebook or hitting the keyboard short cut to run the cell (`Shift + Enter` but can vary based on platform). While the cell is running, an asterisk (`[*]`) will show up to the left of the cell. When that cell has finished executing a sequential number will show up (i.e. `[17]`).

> **Note: Some of the comments in the notebook (those in bold red) are directions for you to modify specific sections of the code. Perform any changes as indicated before running / executing the cell.**

#### Load and Prepare Dataset

* Section `1.0 Install required packages` will install some of the libraries we are going to use in the notebook (many libraries come pre-installed on Cloud Pak for Data). Note that we upgrade the installed version of Watson Machine Learning Python Client. Ensure the output of the first code cell is that the python packages were successfully installed.

    * Run the code cells in section 1.1 and 1.2. Ensuring that the cells complete before continuing.

    ![Imported packages](doc/source/images/mljupyter-packages-installed.png)

* Section `2.0 Load and Clean data` will load the data set we will use to build our machine learning model. In order to import the data into the notebook, we are going to use the code generation capability of Watson Studio.

    * Highlight the code cell below by clicking it. Ensure you place the cursor below the first comment line.

    * Click the `01/00` "Find data" icon in the upper right of the notebook to find the data asset you need to import.

    * If you are using virtualized data, then choose your virtualized merged view (i.e. `USERXXXX.APPLICANTFINANCIALPERSONALLOANSDATA`). If you are using this notebook without virtualized data, you can use the `german_credit_data.csv` [CSV file version of the data set](data/german_credit_data.csv) that has been included in the git repository that you cloned.

    * For your dataset, Click `Insert to code` and choose `Insert Pandas DataFrame`. The code to bring the data into the notebook environment and create a Pandas DataFrame will be added to the cell below.

    * Run the cell and you will see the first five rows of our dataset.

    ![Add the data as a Pandas DataFrame](doc/source/images/mljupyter-insert-dataframe.png)

    ![Generated code to handle Pandas DataFrame](doc/source/images/mljupyter-generated-code-dataframe.png)

* Since we are using generated code to import the data, you will need to update the next cell to assign the `df` variable. Copy the variable that was generated in the previous cell ( it will look like `df=data_df_1`, `data_df_2`, etc) and assign it to the `df` variable (for example `df=df_data_1`).

  ![Update df variable](doc/source/images/mljupyter-update-dataframe-variable.png)

* Continue to run the remaining cells in section 2 to explore and clean the data.

#### Build Machine Learning Model

* Section `3.0 Create a model` cells will run through the steps to build a model pipeline.

    * We will split our data into training and test data, encode the categorial string values, create a model using the Random Forest Classifier algorithm, and evaluate the model against the test set.
    * Run all the cells in section 3 to build the model.

  ![Building the pipeline and model](doc/source/images/mljupyter-buid-pipeline-and-model.png)

#### Save the model

* Section `4.0 Save the model` will save the model to your project.

* We will be saving and deploying the model to the Watson Machine Learning service within our Cloud Pak for Data platform. In the first code cell in section 4.1, be sure to update the `wml_credentials` variable as follows:

    * The url should be the full hostname of the Cloud Pak for Data instance, which you can copy from your browsers address bar (for example, it may look like this: `https://zen.clustername.us-east.containers.appdomain.cloud`)
    * The username and password should be the same credentials you used to log into Cloud Pak for Data.

* You will update the `MODEL_NAME` and `DEPLOYMENT_SPACE_NAME` variables. For the `MODEL_NAME`, create a unique and easily identifiable model name. For the `DEPLOYMENT_SPACE_NAME`, copy the name of your deployment space which was output in the previous code cell.

```python
MODEL_NAME = "user123 credit risk model"
DEPLOYMENT_SPACE_NAME = "Name you used for deployment space"
```

  ![Model and DS Name](doc/source/images/mljupyter-model-ds-name.png)

* Continue to run the cells in the section to save the model to Cloud Pak for Data. Once your model is saved, the call to `wml_client.repository.list_models()` will show it in the output.

  ![Model SAvede](doc/source/images/mljupyter-listmodels-output.png)

**We've successfully built and saved a machine learning model programmatically. Congratulations!**

## Stop the Environment

**Important**: In order to conserve resources, make sure that you stop the environment used by your notebook(s) when you are done.

* Navigate back to your project information page by clicking on your project name from the navigation drill down on the top left of the page.

  ![Back to project](doc/source/images/navigate-to-project.png)

* Click on the 'Environments' tab near the top of the page. Then in the 'Active environment runtimes' section, you will see the environment used by your notebook (i.e the `Tool` value is `Notebook`). Click on the three vertical dots at the right of that row and select the `Stop` option from the menu.

  ![Stop environment](doc/source/images/stop-notebook-environment.png)

* Click the `Stop` button on the subsequent pop up window.

## 5. Online Model Deployment

After a model has been created and saved / promoted to our deployment space, we will want to deploy the model so it can be used by others. For this section, we will be creating an online deployment. This type of deployment will make an instance of the model available to make predictions in real time via an API. Although we will use the Cloud Pak for Data UI to deploy the model, the same can be done programmatically.

* Navigate to the left-hand (☰) hamburger menu and click on `Deployments`.

  ![Analytics Analyze deployments](doc/source/images/menu-analytics-deployments.png)

* Click on the `Spaces` tab and then choose the deployment space you setup previously by clicking on the name of your space.

  ![Deployments space](doc/source/images/select-depspace.png)

* From your deployment space overview, in the table, find the model name for the model you previously built and now want to create a deployment against. Use your mouse to hover over the right side of that table row and click the `Deploy` rocket icon (the icons are not visible by default until you hover over them).

> Note: There may be more than one model listed in the 'Models' section. This can happen if you have run the Jupyter notebook more than once or if you have run through both the Jupyter notebook and AutoAI modules to create models. Although you could select any of the models you see listed in the page, the recommendation is to start with whichever model is available that is using a `spark-mllib_2.4` software specification.

  ![Actions Deploy model](doc/source/images/deploy-spark-model.png)

* On the 'Create a deployment' screen, choose `Online` for the `Deployment Type`, give the Deployment a name and optional description and click the *`Create`* button.

  ![Online Deployment Create](doc/source/images/deploy-online-deployment.png)

* Click on the `Deployments` tab. The online deployment will show as `In progress` and then switch to `Deployed` when done.

  ![Status Deployed](doc/source/images/deploy-status-deployed.png)

### Test Online Model Deployment

Cloud Pak for Data offers tools to quickly test out Watson Machine Learning models. We begin with the built-in tooling.

* From the Model deployment page, once the deployment status shows as `Deployed`, click on the name of your deployment. The deployment `API reference` tab shows how to use the model using `cURL`, `Java`, `Javascript`, `Python`, and `Scala`.

* To get to the built-in test tool, click on the `Test` tab and then click on the *`Provide input data as JSON`* icon.

  ![Test deployment with JSON](doc/source/images/deploy-model-test-page.png)

* Copy and paste the following data objects into the `Body` panel (replace the text that was in the input panel).

    > *Note: Click the tab appropriate for the model you are testing (either an AutoAI model or one built using the Jupyter notebook). Also make sure the input below is the only content in the field. Do not append it to the default content `{ "input_data": [] }` that may already be in the test input panel.*

    === "Jupyter Spark Model"

        ```json
        { "input_data": [{
            "fields": [ "CheckingStatus", "LoanDuration", "CreditHistory", "LoanPurpose", "LoanAmount", "ExistingSavings", "EmploymentDuration", "InstallmentPercent", "Sex", "OthersOnLoan", "CurrentResidenceDuration", "OwnsProperty", "Age", "InstallmentPlans", "Housing", "ExistingCreditsCount", "Job", "Dependents", "Telephone", "ForeignWorker"],
            "values": [[ "no_checking", 13, "credits_paid_to_date", "car_new", 1343, "100_to_500", "1_to_4", 2, "female", "none", 3, "savings_insurance", 46, "none", "own", 2, "skilled", 1, "none", "yes"]]
        }]}
        ```

    === "AutoAI Model"

        ```json
        { "input_data": [{
            "fields": [ "CustomerId", "CheckingStatus", "LoanDuration", "CreditHistory", "LoanPurpose", "LoanAmount", "ExistingSavings", "EmploymentDuration", "InstallmentPercent", "Sex", "OthersOnLoan", "CurrentResidenceDuration", "OwnsProperty", "Age", "InstallmentPlans", "Housing", "ExistingCreditsCount", "Job", "Dependents", "Telephone", "ForeignWorker"],
            "values": [[ "", "no_checking", 13, "credits_paid_to_date", "car_new", 1343, "100_to_500", "1_to_4", 2, "female", "none", 3, "savings_insurance", 46, "none", "own", 2, "skilled", 1, "none", "yes"]]
        }]}
        ```

* Click the *`Predict`* button. The model will be called with the input data and the results will display in the *Result* window. Scroll down to the bottom of the result to see the prediction (i.e "Risk" or "No Risk"):

  ![Testing the deployed model](doc/source/images/deploy-test-model-prediction.png)

> *Note: For some deployed models (for example AutoAI based models), you can provide the request payload using a generated form by clicking on the `Provide input using form` icon and providing values for the input fields of the form. If the form is not available for the model you deployed, the icon will not be displayed.*
  > ![Input to the fields](doc/source/images/deploy-test-input-form.png)

### (Optional) Test Online Model Deployment using cURL

Now that the model is deployed, we can also test it from external applications. One way to invoke the model API is using the cURL command.

> NOTE: Windows users will need the *cURL* command. It's recommended to [download gitbash](https://gitforwindows.org/) for this, as you'll also have other tools and you'll be able to easily use the shell environment variables in the following steps. Also note that if you are not using gitbash, you may need to change *export* commands to *set* commands.

* In a terminal window (or command prompt in Windows), run the following command to get a token to access the API. Replace `<username>` and `<password>` with the username and password you used to log into the Cloud pak for data cluster. Replace `<cluster-url>` with just the hostname of the cloud pak for data cluster (i.e the url from your web browser address bar)

```bash
curl -k -X GET https://<cluster-url>/v1/preauth/validateAuth -u <username>:<password>
```

* A json string will be returned with a value for "accessToken" that will look *similar* to this:

```json
{"username":"scottda","role":"Admin","permissions":["access_catalog","administrator","manage_catalog","can_provision"],"sub":"scottda","iss":"KNOXSSO","aud":"DSX","uid":"1000331002","authenticator":"default","accessToken":"eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VybmFtZSI6InNjb3R0ZGEiLCJyb2xlIjoiQWRtaW4iLCJwZXJtaXNzaW9ucyI6WyJhY2Nlc3NfY2F0YWxvZyIsImFkbWluaXN0cmF0b3IiLCJtYW5hZ2VfY2F0YWxvZyIsImNhbl9wcm92aXNpb24iXSwic3ViIjoic2NvdHRkYSIsImlzcyI6IktOT1hTU08iLCJhdWQiOiJEU1giLCJ1aWQiOiIxMDAwMzMxMDAyIiwiYXV0aGVudGljYXRvciI6ImRlZmF1bHQiLCJpYXQiOjE1NzM3NjM4NzYsImV4cCI6MTU3MzgwNzA3Nn0.vs90XYeKmLe0Efi5_3QV8F9UK1tjZmYIqmyCX575I7HY1QoH4DBhon2fa4cSzWLOM7OQ5Xm32hNUpxPH3xIi1PcxAntP9jBuM8Sue6JU4grTnphkmToSlN5jZvJOSa4RqqhjzgNKFoiqfl4D0t1X6uofwXgYmZESP3tla4f4dbhVz86RZ8ad1gS1_UNI-w8dfdmr-Q6e3UMDUaahh8JaAEiSZ_o1VTMdVPMWnRdD1_F0YnDPkdttwBFYcM9iSXHFt3gyJDCLLPdJkoyZFUa40iRB8Xf5-iA1sxGCkhK-NVHh-VTS2XmKAA0UYPGYXmouCTOUQHdGq2WXF7PkWQK0EA","_messageCode_":"success","message":"success"}
```

* You will save this access token to a temporary environment variable in your terminal. Copy the access token value (without the quotes) in the terminal and then use the following export command to save the "accessToken" to a variable called `WML_AUTH_TOKEN`.

```bash
export WML_AUTH_TOKEN=<value-of-access-token>
```

* Back on the model deployment page, gather the `URL` to invoke the deployed model from the *API reference* by copying the `Endpoint`.

  ![Model Deployment Endpoint](doc/source/images/deploy-model-endpoint.png)

* Now save that endpoint to a variable named `URL` in your terminal by exporting it.

```bash
export URL=<value-of-endpoint>
```

* Now run this curl command from the terminal to invoke the model with the same payload we used previousy:

```bash
curl -k -X POST --header 'Content-Type: application/json' --header 'Accept: application/json' --header "Authorization: Bearer  $WML_AUTH_TOKEN" -d '{"input_data": [{"fields": [ "CheckingStatus", "LoanDuration", "CreditHistory", "LoanPurpose", "LoanAmount", "ExistingSavings", "EmploymentDuration", "InstallmentPercent", "Sex", "OthersOnLoan", "CurrentResidenceDuration", "OwnsProperty", "Age", "InstallmentPlans", "Housing", "ExistingCreditsCount", "Job", "Dependents", "Telephone", "ForeignWorker"],"values": [[ "no_checking", 13, "credits_paid_to_date", "car_new", 1343, "100_to_500", "1_to_4", 2, "female", "none", 3, "savings_insurance", 46, "none", "own", 2, "skilled", 1, "none", "yes"]]}]}' $URL
```

* A json string will be returned with the response, including a  prediction from the model (i.e a "Risk" or "No Risk" at the end indicating the prediction of this loan representing risk).

## 6. Batch Model Deployment

Another approach to expose the model to be consumed by other users/applications is to create a batch deployment. This type of deployment will make an instance of the model available to make predictions against data assets or groups of records. The model prediction requests are scheduled as jobs, which are exected asynchronously. For the lab, we will break this into two steps:

1. Creating the deployment (which we will do using the UI).
1. Creating and scheduling a job with values (which we will do using a Jupyter Notebook).

Lets start by creating the deployment:

* Navigate to the left-hand (☰) hamburger menu and click on `Deployments`.

  ![Analytics Analyze deployments](doc/source/images/menu-analytics-deployments.png)

* Click on the `Spaces` tab and then choose the deployment space you setup previously by clicking on the name of your space.

  ![Deployments space](doc/source/images/select-depspace.png)

* From your deployment space overview, in the table, find the model name for the model you previously built and now want to create a deployment against. Use your mouse to hover over the right side of that table row and click the `Deploy` rocket icon (the icons are not visible by default until you hover over them).

> Note: There may be more than one model listed in them 'Models' section. This can happen if you have run the Jupyter notebook more than once or if you have run through both the Jupyter notebook and AutoAI modules to create models. Although you could select any of the models you see listed in the page, the recommendation is to start with whicever model is available that is using a `spark-mllib_2.4` software specification.

  ![Actions Deploy model](doc/source/images/deploy-spark-model.png)

* On the 'Create a deployment' screen: choose `Batch` for the `Deployment Type`, give the deployment a name and optional description. From the 'Hardware definition' drop down, select the smallest option (`1 standard CPU, 4GB RAM` in this case though for large or frequent batch jobs, you might choose to scale the hardware up). Click the *`Create`* button.

  ![Batch Deployment Create](doc/source/images/deploy-batch-deployment.png)

* Once the status shows as `Deployed` you will be able to start submitting jobs to the deployment.

  ![Status Deployed](doc/source/images/deploy-batch_dep_status.png)

### Create and Schedule a Job

Next we can schedule a job to run against our batch deployment. We could create a job, with specific input data (or data asset) and schedule, either programmatically or through the UI. For this lab, we are going to do this programmatically using the Python client SDK. For this part of the exercise we're going to use a Jupyter notebook to create and submit a batch job to our model deployment.

>*Note: The batch job input is impacted by the machine learning framework used to build the model. Currently, SparkML based model batch jobs require inline payload to be used. For other frameworks, we can use data assets (i.e CSV files) as the input payload.*

#### Run the Batch Notebook

The Jupyter notebook is already included as an asset in the project you imported earlier.

* Go the (☰) navigation menu and under the *Projects* section click on *`All Projects`*.

  ![(☰) Menu -> Projects](doc/source/images/menu-projects.png)

* Click the project name you created in the pre-work section.

* From your `Project` overview page, click on the *`Assets`* tab to open the assets page where your project assets are stored and organized.

* Scroll down to the `Notebooks` section of the page and click on the pencil icon at the right of the `machinelearning-creditrisk-sparkmlmodel` notebook.

  ![Notebook Open](doc/source/images/deploy_batch_open_nb.png)

* When the Jupyter notebook is loaded and the kernel is ready, we will be ready to start executing it in the next section.

##### Notebook sections

With the notebook open, spend a minute looking through the sections of the notebook to get an overview. A notebook is composed of text (markdown or heading) cells and code cells. The markdown cells provide comments on what the code is designed to do. You will run cells individually by highlighting each cell, then either click the `Run` button at the top of the notebook or hitting the keyboard short cut to run the cell (Shift + Enter but can vary based on platform). While the cell is running, an asterisk (`[*]`) will show up to the left of the cell. When that cell has finished executing a sequential number will show up (i.e. `[17]`).

**Please note that some of the comments in the notebook are directions for you to modify specific sections of the code. Perform any changes as indicated before running / executing the cell.**

* Section `1.0 Install required packages` will install some of the libraries we are going to use in the notebook (many libraries come pre-installed on Cloud Pak for Data). Note that we upgrade the installed version of Watson Machine Learning Python Client. Ensure the output of the first code cell is that the python packages were successfully installed.

  ![NB Section 1 Complete](doc/source/images/deploy-batchnb-packageinstall.png)

* Section `2.0 Create Batch Deployment Job` will create a job for the batch deployment. To do that, we will use the Watson Machine Learning client to get our deployment and create a job.

  * In the first code cell for `Section2.1`, be sure to update the `wml_credentials` variable.

    * The url should be the hostname of the Cloud Pak for Data instance.
    * The username and password should be the same credentials you used to log into Cloud Pak for Data.

  * In section 2.2, be sure to update the `DEPLOYMENT_SPACE_NAME` variable with your deployment space name (copy and past the name which is within the output of the previous code cell).

  * In section 2.3, be sure to update the `DEPLOYMENT_NAME` variable with the name of the batch deployment you created previously (copy and past the name which is within the output of the previous code cell).

  ![NB Section 2 Complete](doc/source/images/deploy-batchnb-dsname-set.png)

  ![NB Section 2 Complete](doc/source/images/deploy-batchnb-depname-set.png)

* Continue to run the rest of the cells in section 2 which will load the batch input data set and create the job. The last code cell in section 2 will show that the job is in a queued state.

* Section `3.0 Monitor Batch Job Status` will start polling the job status until it completes or fails. The code cell will output the status every 5 seconds as the job goes from queued to running to completed or failed.

  ![Batch Job Status](doc/source/images/deploy_batch_results_poll.png)

* Once the job completes, continue to run the cells until the end of the notebook.

### Cleanup and Stop Environment

**Important**: In order to conserve resources, make sure that you stop the environment used by your notebook(s) when you are done.

* Navigate back to your project information page by clicking on your project name from the navigation drill down on the top left of the page.

  ![Back to project](doc/source/images/navigate-to-project.png)

* Click on the `Environments` tab near the top of the page. Then in the `Active environment runtimes` section, you will see the environment used by your notebook (i.e the `Tool` value is `Notebook`). Click on the three vertical dots at the right of that row and select the `Stop` option from the menu.

  ![Stop environment](doc/source/images/stop-notebook-environment.png)

* Click the `Stop` button on the subsequent pop up window.

## 7. Integrate Model to Python Flask Application

You can also access the online model deployment directly through the REST API. This allows you to use your model for inference in any of your apps. For this workshop we'll be using a Python Flask application to collect information, score it against the model, and show the results.

> **IMPORTANT: This SAMPLE application only runs on python 3.6 and above, so the instructions here are for python 3.6+ only. You will need to have Python 3.6 or later already installed on your machine**
> *Note: The instructions below assume you have completed the pre-work module and thus have the Git repository already on your machine (cloned or downloaded).*

### Install Dependencies

The general recommendation for Python development is to use a virtual environment ([`venv`](https://docs.python.org/3/tutorial/venv.html)). To install and initialize a virtual environment, use the `venv` module on Python 3:

* Initialize a virtual environment with [`venv`](https://docs.python.org/3/tutorial/venv.html). Run the following commands in a terminal (or command prompt):

  ```bash
  # Create the virtual environment using Python.
  # Note, it may be named python3 on your system.
  python -m venv venv       # Python 3.X

  # Source the virtual environment. Use one of the two commands depending on your OS.
  source venv/bin/activate  # Mac or Linux
  ./venv/Scripts/activate   # Windows PowerShell
  ```

  > **TIP** To terminate the virtual environment use the `deactivate` command.

* Unzip the python application zip file that you downloaded in the pre-work section.

* To install the Python requirements, from a terminal (or command prompt) navigate to where you unzipped the python application. Run the following commands:

  ```bash
  pip install -r requirements.txt
  ```

### Update Environment Variables

It's best practice to store configurable information as environment variables, instead of hard-coding any important information. To reference our model and supply an API key, we'll pass these values in via a file that is read, the key-value pairs in this files are stored as environment variables.

* Copy the `env.sample` file to `.env`.

  ```bash
  cp env.sample .env
  ```

* Edit `.env` to and fill in the `MODEL_URL` as well as the `AUTH_URL`, `AUTH_USERNAME`, and `AUTH_PASSWORD`.

  * `MODEL_URL` is your web service URL for scoring which you got from the section above
  * `AUTH_URL` is the preauth url of your CloudPak4Data and will look like this: `https://<cluster_url>/v1/preauth/validateAuth`
  * `AUTH_USERNAME` is your username with which you login to the CloudPak4Data environment
  * `AUTH_PASSWORD` is your password with which you login to the CloudPak4Data environment

  >Note: Alternatively, you can fill in the `AUTH_TOKEN` instead of `AUTH_URL`, `AUTH_USERNAME`, and `AUTH_PASSWORD`. You will have generated this token in the section above. However, since tokens expire after a few hours and you would need to restart your app to update the token, this option is not suggested. Instead, if you use the username/password option, the app can generate a new token every time for you so it will always have a non-expired ones.

* Here's an example of a completed lines of the .env file.

  ```bash
  # Required: Provide your web service URL for scoring.
  # E.g., MODEL_URL=https://<cluster_url>/v4/deployments/<deployment_space_guid>/predictions
  MODEL_URL=https://cp4d.cp4dworkshops.com/v4/deployments/5f939979-14c2-4538-a2af-a970aeb59abd/predictions

  # Required: Please fill in EITHER section A OR B below:

  # #### A: Authentication using username and password
  #   Fill in the authntication url, your CloudPak4Data username, and CloudPak4Data password.
  #   Example:
  #     AUTH_URL=<cluster_url>/v1/preauth/validateAuth
  #     AUTH_USERNAME=my_username
  #     AUTH_PASSWORD=super_complex_password
  AUTH_URL=https://cp4d.cp4dworkshops.com/v1/preauth/validateAuth
  AUTH_USERNAME=username_001
  AUTH_PASSWORD=my_secure_password_!
  ```

### Start Application

* Start the flask server by running the following command:

  ```bash
  python creditriskapp.py
  ```

* Use your browser to go to [http://0.0.0.0:5000](http://0.0.0.0:5000) and try it out.

  > **TIP**: Use `ctrl`+`c` to stop the Flask server when you are done.

#### Test the application

* Either use the default values pre-filled in the input form, or modify the value and then click the `Submit` button. The python application will invoke the predictive model and a risk prediction & probability is returned:

  ![Get the risk percentage as a result](doc/source/images/flaskapp-output.png)

## Conclusion

In this section we covered one approach to building machine learning models on Cloud Pak for Data. We have seen::

* How to build a model using Jupyter Notebook
* Saving models using the Watson Machine Learning SDK.
* Creating and Testing Online Deployments for models.
* Creating and Testing Batch Deployments for models.
* Integrating the model deployment in an external application.

With this knowledge you should feel right at home within the Jupyter notebook. Moreover, you now know how to build a model and use it in a real life scenario.

## License

This code pattern is licensed under the Apache License, Version 2. Separate third-party code objects invoked within this code pattern are licensed by their respective providers pursuant to their own separate licenses. Contributions are subject to the [Developer Certificate of Origin, Version 1.1](https://developercertificate.org/) and the [Apache License, Version 2](https://www.apache.org/licenses/LICENSE-2.0.txt).

[Apache License FAQ](https://www.apache.org/foundation/license-faq.html#WhatDoesItMEAN)
