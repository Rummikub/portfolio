[Generative AI Models for Action Based Conversations collaborated with ASAPP]


Goal :
The team aimed to build a Large Language Model (LLM) to help customer service agents respond more effectively and increase productivity by creating a chatbot that can anticipate agent responses and suggest next steps, ultimately improving customer satisfaction and reducing operational errors.


Data Preprocessing:
-  ABCD (Action-Based Conversations Dataset) in JSON format containing customer-agent conversations and agent actions
- Explored two approaches to data preparation:
Concatenating utterances without action tags
Keeping utterances separate with action tags
Created training datasets by converting conversations into appropriate formats for model training


Modeling:
- Implemented and fine-tuned two different models:
DistilGPT2 (smaller model)
GPT2-medium (larger model, 355M parameters)
- Trained each model with two variations of the dataset (with and without action tags)
Used Causal Language Modeling to predict the next agent response based on previous utterances


Evaluation:
Compared models using metrics like BERT scores, precision, recall, and F1 scores


Key findings:

GPT2-medium performed better than DistilGPT2 by approximately 0.06
Including action tags improved performance by 0.02-0.03
Highest BERT score achieved was 0.262 with GPT2-medium using non-concatenated utterances with action tags
