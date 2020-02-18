To activate rasa environment
python -m rasa_sdk.endpoint --actions actions

To train the NLU model
rasa train --config config.yml nlu

To train both the NLU and core models
rasa train



* ask_question
- action_get_answers
- utter_confirmation
