# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import joblib

# loading the saved model
loaded_model = joblib.load(open('C:/Users/TaleyaHusn/Desktop/MachineLearning/trained_model.pkl', 'rb'))
loaded_vectorizer = joblib.load((open('C:/Users/TaleyaHusn/Desktop/MachineLearning/tfidf_vectorizer.pkl', 'rb')))


new_data = pd.DataFrame({'text_column': ['حَدَّثَنَا إِسْحَاقُ بْنُ مَنْصُورٍ، حَدَّثَنَا حَبَّانُ بْنُ هِلاَلٍ، حَدَّثَنَا أَبَانٌ، حَدَّثَنَا يَحْيَى، أَنَّ زَيْدًا، حَدَّثَهُ أَنَّ أَبَا سَلاَّمٍ حَدَّثَهُ عَنْ أَبِي مَالِكٍ الأَشْعَرِيِّ، قَالَ قَالَ رَسُولُ اللَّهِ صلى الله عليه وسلم ‏‏ الطُّهُورُ شَطْرُ الإِيمَانِ وَالْحَمْدُ لِلَّهِ تَمْلأُ الْمِيزَانَ ‏.‏ وَسُبْحَانَ اللَّهِ وَالْحَمْدُ لِلَّهِ تَمْلآنِ - أَوْ تَمْلأُ - مَا بَيْنَ السَّمَوَاتِ وَالأَرْضِ وَالصَّلاَةُ نُورٌ وَالصَّدَقَةُ بُرْهَانٌ وَالصَّبْرُ ضِيَاءٌ وَالْقُرْآنُ حُجَّةٌ لَكَ أَوْ عَلَيْكَ كُلُّ النَّاسِ يَغْدُو فَبَائِعٌ نَفْسَهُ فَمُعْتِقُهَا أَوْ مُوبِقُهَا  ‏.‏']}) # Assuming 'vectorizer' is your TF-IDF vectorizer']})  # Replace with your actual new data
    
    # Transform the new data using the pre-fitted vectorizer
new_data_tfidf = loaded_vectorizer.transform(new_data['text_column'])
    
    # Make prediction
predictions = loaded_model.predict(new_data_tfidf)
    
    # Add predicted labels to the DataFrame
new_data['predicted_label'] = predictions
    
    # Display the DataFrame with predicted labels
print(predictions)
    
if (predictions == 0):
    print("The hadith is not authentic")
else:
    print("The hadith is authentic")
    
 
