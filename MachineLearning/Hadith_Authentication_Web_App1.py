# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 12:36:39 2023

@author: TaleyaHusn
"""
import joblib
import streamlit as st
import pandas as pd


# loading the saved model
loaded_model = joblib.load(open('C:/Users/TaleyaHusn/Desktop/MachineLearning/trained_model.pkl', 'rb'))
loaded_vectorizer = joblib.load((open('C:/Users/TaleyaHusn/Desktop/MachineLearning/tfidf_vectorizer.pkl', 'rb')))


# creating a function for Prediction

def authentication_prediction(new_data):
    
        # Assuming 'new_data' is your DataFrame with the new data
    new_data = pd.DataFrame({'text_column': ['حَدَّثَنَا إِسْحَاقُ بْنُ مَنْصُورٍ، حَدَّثَنَا حَبَّانُ بْنُ هِلاَلٍ، حَدَّثَنَا أَبَانٌ، حَدَّثَنَا يَحْيَى، أَنَّ زَيْدًا، حَدَّثَهُ أَنَّ أَبَا سَلاَّمٍ حَدَّثَهُ عَنْ أَبِي مَالِكٍ الأَشْعَرِيِّ، قَالَ قَالَ رَسُولُ اللَّهِ صلى الله عليه وسلم ‏‏ الطُّهُورُ شَطْرُ الإِيمَانِ وَالْحَمْدُ لِلَّهِ تَمْلأُ الْمِيزَانَ ‏.‏ وَسُبْحَانَ اللَّهِ وَالْحَمْدُ لِلَّهِ تَمْلآنِ - أَوْ تَمْلأُ - مَا بَيْنَ السَّمَوَاتِ وَالأَرْضِ وَالصَّلاَةُ نُورٌ وَالصَّدَقَةُ بُرْهَانٌ وَالصَّبْرُ ضِيَاءٌ وَالْقُرْآنُ حُجَّةٌ لَكَ أَوْ عَلَيْكَ كُلُّ النَّاسِ يَغْدُو فَبَائِعٌ نَفْسَهُ فَمُعْتِقُهَا أَوْ مُوبِقُهَا  ‏.‏']}) # Assuming 'vectorizer' is your TF-IDF vectorizer']})  # Replace with your actual new data

    # Transform the new data using the pre-fitted vectorizer
    new_data_tfidf = loaded_vectorizer.transform(new_data['text_column'])
    
    # Make predictions
    predictions = loaded_model.predict(new_data_tfidf)
    
    # Add predicted labels to the DataFrame
    new_data['predicted_label'] = predictions
    
    # Display the DataFrame with predicted labels
    print(predictions)
    
    if (predictions == 0):
        return "The hadith is not authentic"
    else:
        return "The hadith is authentic"
  
    
  
def main():
    
    
    # giving a title
    st.title('Hadith Authentication Prediction Web App')
    
    
    # getting the input data from the user
    
    
    new_data = st.text_input('Enter the hadith in arabic')
    
    
    # code for Prediction
    checking = ''
    
    # creating a button for Prediction
    
    if st.button('Hadith Authentication Result'):
        checking = authentication_prediction([new_data])
        
        
    st.success(checking)
    
    
    
    

if __name__ == '__main__':
    main()

