#########################
#Business Problem:
#Armut, Turkey's largest online service platform, brings together service providers and those who want to receive service.
#It provides easy access to services such as cleaning, modification and transportation with a few touches on your computer or smart phone.
#It is desired to create a product recommendation system with Association Rule Learning by using the data set containing the service users
#and the services and categories these users have received.

# TR: İş Problemi
#########################
# Türkiye’nin en büyük online hizmet platformu olan Armut, hizmet verenler ile hizmet almak isteyenleri buluşturmaktadır.
# Bilgisayarın veya akıllı telefonunun üzerinden birkaç dokunuşla temizlik, tadilat, nakliyat gibi hizmetlere kolayca
# ulaşılmasını sağlamaktadır.
# Hizmet alan kullanıcıları ve bu kullanıcıların almış oldukları servis ve kategorileri içeren veri setini kullanarak
# Association Rule Learning ile ürün tavsiye sistemi oluşturulmak istenmektedir.


#########################

#UserId: Customer number
#ServiceId: Anonymized services belonging to each category. (Example: Upholstery washing service under the cleaning category)
#A ServiceId can be found under different categories and refers to different services under different categories.
#(Example: Service with CategoryId 7 and ServiceId 4 is honeycomb cleaning, while ServiceId with CategoryId 2 is furniture assembly)
#CategoryId: Anonymized categories. (Example: Cleaning, transportation, renovation category)
#CreateDate: The date the service was purchased

#########################
# 1-Data Preprocessing
#########################

import pandas as pd

pd.set_option("display.max_columns", None)
from mlxtend.frequent_patterns import apriori, association_rules

df_ = pd.read_csv(r"C:\Users\User\OneDrive\Masaüstü\veri bilimi\ArmutARL-221114-234936\armut_data.csv")
df = df_.copy()
df.head()

#2-Data Preparation

#ServiceID represents a different service for each CategoryID. Let's combine ServiceID and CategoryID with "_" to create a new variable to represent these services

df["Hizmet"] = [str(row[1]) + "_" + str(row[2]) for row in df.values]
df.head()

# -The data set consists of the date and time the services were received, there is no basket definition (invoice, etc.).
# In order to apply Association Rule Learning, a basket (invoice, etc.) definition must be created.
# Here, the definition of basket is the services that each customer receives monthly. For example; A basket of 9_4, 46_4 services received by the customer with id 7256 in the 8th month of 2017;
# 9_4, 38_4 services received in the 10th month of 2017 represent another basket. Baskets must be identified with a unique ID.
# To do this, first create a new date variable containing only the year and month. Set the UserID and the date variable you just created to "_"
#and assign it to a new variable named ID.

df["CreateDate"] = pd.to_datetime(df["CreateDate"])
df.head()
df["NEW_DATE"] = df["CreateDate"].dt.strftime("%Y-%m")
df.head()
df["SepetID"] = [str(row[0]) + "_" + str(row[5]) for row in df.values]
df.head()

df[df["UserId"] == 7256]


#########################

#Create the cart service pivot table as below.

# Hizmet         0_8  10_9  11_11  12_7  13_11  14_7  15_1  16_8  17_5  18_4..
# SepetID
# 0_2017-08        0     0      0     0      0     0     0     0     0     0..
# 0_2017-09        0     0      0     0      0     0     0     0     0     0..
# 0_2018-01        0     0      0     0      0     0     0     0     0     0..
# 0_2018-04        0     0      0     0      0     1     0     0     0     0..
# 10000_2017-08    0     0      0     0      0     0     0     0     0     0..


#3. Creating Association Rules

#We will use Apriori Algorithm which is an algorithm that allows the implementation of Association Rules.
#Apriori Algorithm consists of Support, Confidence, Lift metrics.
#Support: Probability of X and Y being bought together.
#Confidence: Probability of buying Y when X is bought
#Lift: When X is purchased, the probability of buying Y also increases by a factor of lift.



invoice_product_df = df.groupby(["SepetID", "Hizmet"])["Hizmet"].count().unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)
invoice_product_df.head()


frequent_itemsets = apriori(invoice_product_df, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
rules.head()


#4- Use the arl_recommender function to recommend a service to a user who has received the last 2_0 service.

def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in sorted_rules["antecedents"].items():
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"]))

    recommendation_list = list({item for item_list in recommendation_list for item in item_list})
    return recommendation_list[:rec_count]


arl_recommender(rules, "2_0", 4)


