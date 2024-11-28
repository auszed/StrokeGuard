# StrokeGuard app

![StrokeGuard](results/img/StrokeGuard_background.png)

## Description and objective of the project

Our app, StrokeGuard, is designed to empower individuals with the tools and knowledge
they need to prevent strokes. By combining the latest in medical research with user-friendly 
technology, StrokeGuard offers personalized health assessments.
Users can easily track their blood pressure, BMI, glucose levels,
and other vital health metrics through the app. 
The app will include educational resources to help users understand the signs of a stroke 
and what actions to take in an emergency, promoting awareness and proactive health management.

## Table of Contents

1. [EDA](#EDA)
2. [Inference_Analysis](#Inferential_analysis)
3. [Model_training](#Model_training)
4. [Model_Evaluation](#Model_Evaluation)
5. [Deployment](#Deployment)
6. [Recommendations](#Recommendations)
7. [License](#License)

## Introduction

Welcome to our data science project! We're diving into a world of numbers and information to help an 
insurance company in India. They want to know who might be interested in buying travel insurance.
To figure this out, we're using a bunch of data they've collected over the years,
like who they've talked to and what they've bought. Our goal is to build an ML model that can predict
who's most likely to want travel insurance. By doing this, we hope to give the company useful advice
on how to reach more people in India who might be interested in buying insurance for their travels.

We will be taking into account that we will be selling the insurance vy **5000₹** (Rupee) this will help to have a forecast as a benchmark and understand better the base case for the project.

## EDA

In this summary, we will outline the key points extracted from the analysis. We have identified the following:

[EDA results](./results/EDA_findings.md)

We observed initial indications suggesting that age is correlated with the likelihood of having a stroke,
with individuals above 40 showing increased probabilities. Additionally, we did not find strong correlations
between the dependent variable **stroke** and other categories. However, there is a notable suggestion 
that being married may correlate with higher stroke risk, likely due to the fact that a significant portion 
of individuals above 40 are married, thus implying this connection.

Next, we can locate the workbook where everything was developed.

[EDA workbook](./001_EDA.ipynb)


## Inferential_analysis

We apply a chi-square test for the categorical variables, and we found that there is no association between
**hypertension, heart_disease, ever_married, work_type and smoking_status** meaning that this feature could 
not be helpful at the time to predict the target variable. but because this variables make sense we will leave them
just because our intuition But for **gender and Residence type** we could see a significant dependence between
the target variable **stoke**, in this context residence type was a surprising result!

After that we apply a test to verifying the numerical categories, and we found that both values in the both 
circumstances could have a different distribution when we compared BMI, age and glucose levels; to if they had a stroke,
so we could expect that we will be using this features for the predictive models

Here are the conclusions of the Inference analysis

[Inference and Modeling results ](./results/Inference_and_modeling.md)

Here tou can find the workbook of the analysis

[Inferential analysis workbook ](./002_Inference.ipynb)

## Model_training

We will stick with this 3 metrics for the choosing model
- **F1 score**
- **Recall**
- **ROC curve.**

And we finally get to the results from the models and this are the models choose for the deployment.

| Model Name         | Recall | F1      | ROC AUC  |
|--------------------|--------|---------|----------|
| GNB_001            | 0.8125 | 0.135417| 0.748808 |
| ensemble_model_001 | 0.8750 | 0.15053 | 0.756982 |

had the best performance if we compare them with the other models.

[Modeling results ](./results/Inference_and_modeling.md)

Here you had the workbook were we can see the work

[Modeling workbook ](./003_modeling_selection.ipynb)

## Model_Evaluation

We see that some features, like **Age**, have significant predictive power in these models, which makes sense but 
not in all cases. We could optimize this variable by applying different transformations. Additionally, features 
like **heart disease** and **heart attack** have evident connections with having a stroke. On the other hand, 
some features, such as **residence type** and **gender**, do not show a strong connection with the target variable
and do not contribute much to the prediction capabilities.

[Modeling results ](./results/Interpretability_model.md)

Here you had the workbook were we can see the work

[Modeling workbook ](./004_interpretability_model.ipynb)


## Deployment

Finally, we deploy the model using Streamlit infrastructure, because was easily to manage and deploy, so we could see 
the app here:


[strokeguard app link](https://auszed-strokeguard.streamlit.app/)


## Recommendations

- Our next step involves requesting additional data. While our current dataset is sufficient, acquiring more data is expected to enhance the models' performance.
  
- Furthermore, establishing an infrastructure that allows us to provide weekly updates to the model could further improve its performance over time.
  
- Additionally, developing a dedicated in-app chatbot could assist patients by addressing questions related to the topic effectively.
## License
[MIT licensed](./License.md)
This project is licensed under the [License Name]. See the [License.md](License.md) file for details.

