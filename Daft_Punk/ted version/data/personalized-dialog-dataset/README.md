# Personalized Dialog Tasks Dataset
This directory contains a set of 5 tasks for testing end-to-end dialog systems in a goal-oriented setting with a focus on personalization of conversation. The dataset has been described in the paper ["Personalization in Goal-Oriented Dialog"](https://arxiv.org/abs/1706.07503) by Chaitanya Joshi, Fei Mi and Boi Faltings. All dialogs are set in the domain of restaurant reservation.

# Data
We provide two variants of the dataset- a full set with 6000/12000 dialogs each for training, development and testing; and a small set restricted to 1000 dialogs each for training, development and testing. We also include a second test set (with suffix -OOV.txt) that contains dialogs including entities not present in training and development sets.

The file format for each task is as follows:
```
1 user_attributes
ID user_utterance [tab] bot_utterances
...
```

The IDs for a given  dialog start at 1 and increase. When the IDs in a file reset back to 1 you can consider the following sentences as a new dialog. The line starting with ID 1 contains attributes associated with a user profile and subsequent lines contain various turns forming the dialog. When the bot speaks two times in a row, we used the special token "<SILENCE>" to fill in for the missing user utterance.

For example (for Task 1):
```
1 male elderly
2 hello	greetings sir what may i assist you with today
3 may i have a table in a moderate price range with italian food for eight	excellent sir i will start the request now
4 <SILENCE>	may i know where the restaurant should be located
5 bombay please	excellent sir please give me a moment to provide you with options
6 <SILENCE>	api_call italian bombay eight moderate
```

The goal of the tasks is to predict the bot utterances, that can be sentences or API calls (sentences starting with the special token "api_call").

Along with the train, dev and test sets, we also include a knowledge base file (personalized-dialog-kb-all.txt) that contain all entities appearing in dialogs for tasks 1-5. We also include a file containing the candidates to select the answer from (personalized-dialog-candidates.txt) for tasks 1-5, that is simply made of all the bot utterances in train, dev, test for these tasks. 

In addition to the small and full datasets, we also provide a split-by-profile dataset where each directory contains 1000 dialogs each for training, development and testing for a specific user profile. This set can be used to analyze multi-task learning capabilities of models.

# License
This dataset is released under Creative Commons Attribution 3.0 Unported license. A copy of this license is included with the data.

# Contact
For more details on the dataset and baselines, see the paper "Personalization in Goal-Oriented Dialog" by Chaitanya Joshi, Fei Mi and Boi Faltings, available at https://arxiv.org/abs/1706.07503. For any information, contact Chaitanya Joshi: chaitjo (at) gmail (dot) com.
