# Kaggle Santander Product Recommendation

Ref: https://www.kaggle.com/c/santander-product-recommendation/overview

Ready to make a downpayment on your first house? Or looking to leverage the equity in the home you have? To support needs for a range of financial decisions, Santander Bank offers a lending hand to their customers through personalized product recommendations.

Under their current system, a small number of Santander’s customers receive many recommendations while many others rarely see any resulting in an uneven customer experience. In their second competition, Santander is challenging Kagglers to predict which products their existing customers will use in the next month based on their past behavior and that of similar customers.

With a more effective recommendation system in place, Santander can better meet the individual needs of all customers and ensure their satisfaction no matter where they are in life.

Disclaimer: This data set does not include any real Santander Spain's customer, and thus it is not representative of Spain's customer base. 

### Dataset Details

In this competition, you are provided with 1.5 years of customers behavior data from Santander bank to predict what new products customers will purchase. The data starts at 2015-01-28 and has monthly records of products a customer has, such as "credit card", "savings account", etc. You will predict what additional products a customer will get in the last month, 2016-06-28, in addition to what they already have at 2016-05-28. These products are the columns named: ind_(xyz)_ult1, which are the columns #25 - #48 in the training data. You will predict what a customer will buy in addition to what they already had at 2016-05-28. 

The test and train sets are split by time, and public and private leaderboard sets are split randomly.

Please note: This sample does not include any real Santander Spain customers, and thus it is not representative of Spain's customer base. 
File descriptions

    train.csv - the training set
    test.csv - the test set
    sample_submission.csv - a sample submission file in the correct format

