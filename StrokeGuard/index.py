import streamlit as st
import pandas as pd
from app_helper import predict_or_analyze, prediction_answer

# Main function to run the app
def main():
    st.title(':hospital: Stroke Patient Verification')
    model_selection = st.selectbox('Select the model', ['GNB', 'Ensemble_model'])

    st.write('A stroke happens when blood flow to a part of the brain is cut off or greatly reduced, preventing brain from getting the oxygen and nutrients it needs. Without oxygen, brain cells start to die within minutes. Now we will see whats the probability to that it can happen in the future')

    # other of the columns
    col1, col2 = st.columns([1, 2])
    col3, col4, col5, col6, col7, col8 = st.columns([1, 1, 1, 1, 1, 1])
    col9, col10 = st.columns([1, 1])

    # add the content by columns
    with col1:
        st.markdown('Gender')
        input_gender = st.radio('', ['Male', 'Female', 'Other'], index =1)

    with col2:
        st.markdown('Age')
        input_age = st.slider('Age', min_value=0, max_value=125, value=55)

    with col3:
        st.markdown('Ever Married')
        input_ever_married = st.radio('', ['Yes', 'No'], )

    with col4:
        st.markdown('Work type')
        input_work_type = st.radio('', ['Private', 'Self-employed', 'Govt_job', 'Children', 'Never_worked'])

    with col5:
        st.markdown('Residence type')
        input_Residence_type = st.radio('', ['Urban', 'Rural'])

    with col6:
        st.markdown('Smoking status')
        input_smoking_status = st.radio('', ['never smoked', 'formerly smoked', 'smokes', 'Unknown'])

    with col7:
        st.markdown('Hypertension')
        transform_hypertension = st.radio('__', ['Yes', 'No'],index =1)
        if transform_hypertension == 'Yes':
            input_hypertension = 1
        else:
            input_hypertension = 0
    with col8:
        st.markdown('Heart_disease')
        transform_heart_disease = st.radio('_', ['Yes', 'No'],index =1)
        if transform_heart_disease == 'Yes':
            input_heart_disease = 1
        else:
            input_heart_disease = 0
    with col9:
        st.markdown('AVG glucose level')
        input_avg_glucose_level = st.slider('', min_value=0, max_value=300, value=150)

    with col10:
        st.markdown('BMI')
        input_bmi = st.slider('', min_value=0, max_value=100, value=25)

    # Create a dictionary of default values for each feature
    values_to_analyse = {
        'gender': input_gender,
        'age': input_age,
        'hypertension': input_hypertension,
        'heart_disease': input_heart_disease,
        'ever_married': input_ever_married,
        'work_type': input_work_type,
        'Residence_type': input_Residence_type,
        'avg_glucose_level': input_avg_glucose_level,
        'bmi': input_bmi,
        'smoking_status': input_smoking_status
    }

    # model analyse
    if st.button('Analyze'):
        data = pd.DataFrame(values_to_analyse, index=[0])
        result, result_probabilities = predict_or_analyze(data, model_selection)
        stroke_prob = round(result_probabilities[0][1] * 100, 6)
        no_stroke_prob = round(result_probabilities[0][0]*100,6)

        # Add the content
        st.write(data)
        if result == 0:
            st.success("It appears that the likelihood of us experiencing a stroke is low")

        elif result == 1:
            st.error("We might have experienced a stroke. Here's some advice if we notice any symptoms and what actions we can take to address them!")

        st.metric(label="Probability of having a stroke", value=stroke_prob)
        st.metric(label="Probability of not having a stroke", value=no_stroke_prob)

        # content add it depending on the answer
        answer = prediction_answer(result)
        st.write(answer)


if __name__ == '__main__':
    main()
