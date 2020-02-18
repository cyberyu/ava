## fallback
- utter_default

## greeting path 1
* greet
- utter_greet

## fine path 1
* fine_normal
- utter_help

## fine path 2
* fine_ask
- utter_reply

## qa path
* ask_question
- utter_ofc
- action_get_answers
- utter_confirmation

## confirm answer path 1
* answer_negative
- utter_repeat_question

## confirm answer path 2
* answer_positive
- utter_anything_else

## thanks path 1
* thanks
- utter_anything_else

## bye path 1
* bye
- utter_bye
