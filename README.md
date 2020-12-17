1. Title: Adding features and exploring non-linearities betrayal linguistics

2. Abstract

The paper explores linguistic features that foretell the impending betrayal in a relationship between two people. The authors of the paper start by identifying possible linguistic features such as politeness, in the communication between the pair, that could be indicators of whether betrayal is coming or not. Then, these features are used to construct a machine learning model using logistic regression that predicts whether a given relationship will end in betrayala or not. Our idea is to use the dataset given to us with the paper and start by adding new features, such as the frequency of supporting actions within a relationship, not considered by the authors of the paper. Then we will also try to predict whether betrayal will take place or not, but this time using other models to see if we can boost the performance. 

3. Research Questions

Q1. Do more balanced out friendships have different outcomes than more unidirectional ones in terms of the amount of help the two individuals provide each other?

Q2. Does taking into account the irregularity of the features (variance) provide additional information that can improve the perfomance of the predictor?

Q3. Can other models improve the predictive power of the model used in the paper based on (mostly) the same features?

Q4. Does approaching the data from a friendship standpoint, instead of a season standpoint as done in the original paper, improve the success rate of the machine learning?

4. Proposed dataset

The preprocessed dataset from the paper. We will process it in a similar way as what we did for Milestone 2.

5. Methods

Additional Features:

- We plan to use all the same features as those used by the authors for our ML model. However, we noticed that the model in the paper is trained based only on seasons of the game. We would like to aggregate all the seasons for every recorded friendship, to compute the means and variances of each feature, because we think that they might contain information, useful for improving the model.

- Frequency of supporting actions within a relationship (mean, variance, imbalance): we suspect that not all friendships are created equal. To test this hypothesis, we want to create an additional feature that takes into account the number and frequency of supporting actions within a friendship. The final goal would be to see if this feature has an influence on the outcome of the relationship.

ML: The authors of the paper use a logistic regression model for their final prediction. This is a model that is based on a linear relationship. We would like to try to use other models such as gradient boosting regressor, or potentially others, to see if we can improve on the authors' performance.

6. Proposed timeline

We see ourselves implementing our creative extension in 3 steps, each taking about one week.

- step 1: Retrieve the features used by the authors and see to what extent we are able to reproduce them.

- step 2: Create the new features (based on the supporting actions) and compute aggregate of all the existing features. Interpret the results.

- step 3: Implement the machine learning models for predicting how a given relationship will end.

7. Organization within the team

We will work on part A of the milestone together as a team.

8. Questions for TAs (optional)

Q1. Would you have any non-linear ML models to suggest for the last step in our proposal?
