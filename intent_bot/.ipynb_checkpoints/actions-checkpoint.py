from rasa_sdk import Action
import requests
import json
from rasa_sdk.events import SlotSet
import numpy as np
import pickle


class ActionGetAnswers(Action):

    def name(self):
        return 'action_get_answers'


    def run(self, dispatcher, tracker, domain):
        question = (tracker.latest_message)['text']
        label2_text = pickle.load('num2label381classes.p')
        print(question)
        response = requests.post("https://domino.opse.c1.vanguard.com/v1/UTUZ/Ava_chatbot/endpoint",
            headers = {
                "X-Domino-Api-Key": "PBfRAGsouMCedZIBy4URyg5Syf6JxpYbRqmXaL1OpwDR0T4qEcXxs1kC2qxfFUL9",
                "Content-Type": "application/json"
            }, verify=False,
            json = {
                "parameters": [question]
            }
        )
        re = response.json()
        result = re['result']['predictions']
        label = np.argmax(result)
        
        entropy = -np.sum(np.multiply(result, np.log(result)))
        
        message = 'Your intent classification is ' + str(label2_text[label]) + ' and Entropy is ' + str(entropy)
        dispatcher.utter_message(message)
        return[]