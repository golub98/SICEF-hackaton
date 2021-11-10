# SICEF-hackaton
COVID-19 classifier based on Light Gradient Boosting. I developed this classifier for my team on SICEF hackathon(6.11.2021.). There we developed a web page to prevent the spreading of the coronavirus. This classifier was used for anonymous self-evaluation on COVID-19. Then the person could send an anonymous email to the list of friends that he or she was in close contact. By doing this we managed to inform individuals much faster than if we would wait for the results of the PCR test. Also, we solved the social pressure that potentially contagious people would experience informing their contacts about possible COVID-19 infection.

## Dataset:(https://github.com/nshomron/covidpred)

Attributes that I used for classification were: Cough (true/false), Fever (true/false), Sore throat (true/false), Shortness of breath (true/false), Headache (true/false), and an indicator of whether a person was in close contact with confirmed COVID-19 case. All attributes are binary. Since the dataset is unbalanced ( 95% negative and 5% positive), I used f1 score for the metric while training LightGBM. In the end, accuracy is high (96.8%), but the false-negative percent is 30%. This is because some attributes are biased. Namely, from all persons that were positive on some symptom (Headache) high percentage was positive on COVID-19. I will cite the paper that used the same dataset, and run to the same problem: "This is reflected in the proportion of persons who were COVID-19 positive from the total number of individuals who were positive for each symptom. Accordingly, we identified features with biased reporting (headache 96.2%, sore throat 92.3%, and shortness of breath 92.4%) and symptoms with balanced reporting (cough 27.4% and fever 45.9%). Mislabeling of symptoms may also arise from an underestimation and underreporting of symptoms among persons who tested negative."(https://www.nature.com/articles/s41746-020-00372-6#Sec5). 

In the close future I will try to lower the false-negative percent by avoiding bias attributes, and exploring different dataset. Also, I plan on implementing random forest and comparing it with LightGBM.

