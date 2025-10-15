## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:

     import pandas as pd

     df=pd.read_csv("C:\\Users\\admin\\Downloads\\data.csv")
     df

<img width="600" height="446" alt="image" src="https://github.com/user-attachments/assets/5ab32a53-2d4c-4f19-be3a-9fa8f26b6943" />

    from sklearn.preprocessing import OrdinalEncoder,LabelEncoder

    df1=df.copy()
    df1=df.copy()
    education=["High School","Diploma","Bachelors","Masters","PhD"]
    enc=OrdinalEncoder(categories=[education])
    enc.fit_transform(df1[['Ord_2']])
    df1['ordinalencoder']=enc.fit_transform(df1[['Ord_2']])
    df1

<img width="791" height="457" alt="image" src="https://github.com/user-attachments/assets/9aa209a3-cb2f-4077-b439-2d86c2e55104" />

      df2=df.copy()
      enc=LabelEncoder()
      df1['LabelEncoder']=enc.fit_transform(df1[['Ord_2']])
      df1

<img width="923" height="457" alt="image" src="https://github.com/user-attachments/assets/be7c8687-8832-4f44-8366-f15f874df3da" />


       from sklearn.preprocessing import OneHotEncoder
       df3=df.copy()
       enc=OneHotEncoder()
       newdata=pd.DataFrame(enc.fit_transform(df3[['City']]))
       df4=pd.concat([df3,newdata],axis=1)
       df4

<img width="693" height="460" alt="image" src="https://github.com/user-attachments/assets/ab4d8d1a-1e53-4a1a-9efd-72ee4b8c4878" />

       pd.get_dummies(df4,columns=['City'])

<img width="1027" height="425" alt="image" src="https://github.com/user-attachments/assets/3789e63f-d961-486d-8f05-99b2726edf84" />

      pip install --upgrade category_encoders

<img width="1032" height="413" alt="image" src="https://github.com/user-attachments/assets/ac13f42c-6344-4cfe-a487-2812aa4c4058" />

       from category_encoders import BinaryEncoder

       df5=df.copy()
       enc=BinaryEncoder()
       newdata=pd.DataFrame(enc.fit_transform(df5[['Ord_1']]))
       df6=pd.concat([df5,newdata],axis=1)
       df6


<img width="903" height="461" alt="image" src="https://github.com/user-attachments/assets/00c3d655-6ce0-4c9c-910c-53b78da46062" />

      from category_encoders import TargetEncoder

      df7=df.copy()
      enc=TargetEncoder()
      newdata=pd.DataFrame(enc.fit_transform(df7[['Ord_1']],df7['Target']))
      df8=pd.concat([df7,newdata],axis=1)
      df8

<img width="727" height="461" alt="image" src="https://github.com/user-attachments/assets/f527520e-ad42-4871-8f33-fd99cd55f1a8" />

2.Data_Transform

    import pandas as pd
    df=pd.read_csv("C:\\Users\\admin\\Downloads\\Data_to_Transform.csv")
    df

<img width="967" height="605" alt="image" src="https://github.com/user-attachments/assets/5e84fbdf-e707-4658-8eea-d912531b6623" />

      df.skew()

<img width="436" height="162" alt="image" src="https://github.com/user-attachments/assets/178b7c30-f9a2-40bb-8d26-a31360ecb26d" />

          import numpy as np
          df1=df.copy()
          df['log transformation']=np.log(df["Moderate Positive Skew"])
          df1

<img width="991" height="590" alt="image" src="https://github.com/user-attachments/assets/2fe94d70-6d8c-45c2-9f2c-f5e6878ea193" />

       import statsmodels.api as sm
       import matplotlib.pyplot as plt

<img width="951" height="677" alt="image" src="https://github.com/user-attachments/assets/0b92cb8d-af39-43d8-a376-3fca8f90f12b" />

       sm.qqplot(df["Highly Positive Skew"],line="45")
       plt.show()

<img width="912" height="645" alt="image" src="https://github.com/user-attachments/assets/61ab6480-5488-400f-b45d-5e062f13bdde" />

       sm.qqplot(df["Moderate Negative Skew"],line="45")
       plt.show()

<img width="875" height="621" alt="image" src="https://github.com/user-attachments/assets/eebe1e67-c5f5-4db6-9c24-5b063e80372c" />

      sm.qqplot(df["Highly Negative Skew"],line="45")
      plt.show()


<img width="915" height="671" alt="image" src="https://github.com/user-attachments/assets/dea17bd8-de85-444a-bcd9-315e8fcfb486" />

     sm.qqplot(df["log transformation"],line="45")
     plt.show()

<img width="897" height="652" alt="image" src="https://github.com/user-attachments/assets/37ae33a4-34e7-4061-9f43-a77a639280db" />

       df2=df.copy()
       df2["sqrt transformation"]=np.sqrt(df["Moderate Positive Skew"])
       df2

<img width="1058" height="462" alt="image" src="https://github.com/user-attachments/assets/6fc35e97-cb01-4e9b-bfa8-4018c5c0e656" />

      sm.qqplot(df["sqrt transformation"],line="45")
      plt.show()

<img width="941" height="686" alt="image" src="https://github.com/user-attachments/assets/b7c35d78-6720-46c2-a65b-f4b8a362f2ff" />

       df3=df.copy()
       df3['square transformation']=np.square(df["Moderate Positive Skew"])
       df3

<img width="1045" height="402" alt="image" src="https://github.com/user-attachments/assets/1c0496f3-8f9e-4fe3-b943-92b31565f3bd" />

     sm.qqplot(df3["square transformation"],line="45")
     plt.show()

<img width="950" height="646" alt="image" src="https://github.com/user-attachments/assets/08f61e08-c892-4ff2-ba32-96a1471437d2" />

     df4=df.copy()
     df4['reciprocal transformation']=np.square(df["Moderate Positive Skew"])
     df4

<img width="1090" height="417" alt="image" src="https://github.com/user-attachments/assets/6495dba6-4e4f-4bcf-a62d-5ab8f583f964" />

     sm.qqplot(df4["reciprocal transformation"],line="45")
     plt.show()

<img width="1008" height="645" alt="image" src="https://github.com/user-attachments/assets/857820be-0ea0-4ab8-ad29-752c6115cce5" />

      from scipy import stats

      df5=df.copy()
      df['boxcox transformation'],p=stats.boxcox(df5["Moderate Positive Skew"])
      df5

<img width="1052" height="435" alt="image" src="https://github.com/user-attachments/assets/5b052ba6-eb4c-4752-910f-228ddff06a62" />

      sm.qqplot(df["boxcox transformation"],line="45")
      plt.show()

<img width="901" height="627" alt="image" src="https://github.com/user-attachments/assets/21a1c491-ec0f-4125-84b3-c14e98ee7c88" />

       df6=df.copy()
       df6['yeojohnson transformation'],p=stats.boxcox(df5["Moderate Positive Skew"])
       df6

<img width="1048" height="386" alt="image" src="https://github.com/user-attachments/assets/96ed0109-a6ff-44bf-b90b-2ec9d4caea36" />

      sm.qqplot(df6["yeojohnson transformation"],line="45")
      plt.show()

<img width="942" height="690" alt="image" src="https://github.com/user-attachments/assets/f781d8eb-61ea-4084-ab60-577c8c72e035" />

        from sklearn.preprocessing import QuantileTransformer

        df7=df.copy()
        qt=QuantileTransformer(output_distribution="normal")
        df7['QuantileTransformation']=qt.fit_transform(df7[["Highly Positive Skew"]])
        df7

<img width="1058" height="418" alt="image" src="https://github.com/user-attachments/assets/330d4646-679d-462d-8cec-6c77099acd34" />

        sm.qqplot(df7['QuantileTransformation'],line="45")
        plt.show()

<img width="883" height="667" alt="image" src="https://github.com/user-attachments/assets/bbce3718-d8b1-43ab-b954-946915369888" />





























     



# RESULT:
       Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.

       
