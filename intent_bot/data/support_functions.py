import pandas as pd
def generate_questions(df):
    question_list = []
    question_dict = {}
    question_dict[df.iloc[0]['URL']]=['How do I send money from my bank to Vanguard?']
    question_dict[df.iloc[1]['URL']]=['How do I send a wire to my bank?','How much does Vanguard Brokerage charge for the wire fee?','How much does Vanguard Brokerage charge?', 'Do Flagship clients need to pay the wire fee?']
    question_dict[df.iloc[2]['URL']]=['What is EBT?', 'What is a wire transfer?', 'Can I contribute to an IRA account via wire?', 'How long does withdraw via EBT take?', 'Is it free to receive incoming wires?']
    question_dict[df.iloc[3]['URL']]=['How long does it take to verify a bank?']
    question_dict[df.iloc[4]['URL']]=['How do I exchange a Vanguard mutual fund for another Vanguard mutual fund online?']
    question_dict[df.iloc[5]['URL']]=['Do I need to pay tax when withdrawing Roth IRA contributions?', 'How much federal penalty tax do I need to pay if I withdraw my traditional IRA contributions at the age of 55?']
    question_dict[df.iloc[6]['URL']]=['Do I need to have a Vanguard Roth IRA set up to receive converted assets?', 'How do I convert investments from my traditional IRA brokerage?']
    question_dict[df.iloc[7]['URL']]=['How do I contribute to my IRA?']
    question_dict[df.iloc[8]['URL']]=['How to re-establish checkwriting?', 'How to re-establish Agent Authorization?','How to re-establish cost basis information?', 'How to change my dividends and capital gains after an upgrade?']
    question_dict[df.iloc[9]['URL']]=['How do I buy an ETF or stock online?','How much do Vanguard Brokerage Services charge for ETF trading?']
    question_dict[df.iloc[10]['URL']]=['How much do Vanguard Brokerage Services charge for mutual funds?']
    question_dict[df.iloc[11]['URL']]=['How much do Vanguard Brokerage Services charge for stocks?']
    question_dict[df.iloc[12]['URL']]=['How much do Vanguard Brokerage Services charge for options']
    question_dict[df.iloc[13]['URL']]=['How much do Vanguard Brokerage Services charge for cds bonds?']
    question_dict[df.iloc[14]['URL']]=['How much do Vanguard Brokerage Services charge for other service?']
    question_dict[df.iloc[15]['URL']]=['Is there an age limit for traditional IRA?', 'Is there an age limit for Roth IRA?', 'What are the contribution limits for Roth IRA?']
    question_dict[df.iloc[16]['URL']]=['What are share classes of Vanguard mutual funds?']
    question_dict[df.iloc[17]['URL']]=['How do I buy a Vanguard mutual fund online?']
    question_dict[df.iloc[18]['URL']]=['How to use your settlement fund?', 'How does the settlement fund work?']
    question_dict[df.iloc[19]['URL']]=['How do I sell a Vanguard mutual fund online?']
    question_dict[df.iloc[20]['URL']]=['What is the status of my order?']
    question_dict[df.iloc[21]['URL']]=['Why did I not receive a tax form?']
    question_dict[df.iloc[22]['URL']]=['How do you report my cost basis information?']
    question_dict[df.iloc[23]['URL']]=['Will I pay any fees for my transfer?']
    question_dict[df.iloc[24]['URL']]=['Is there any paperwork I need to fill out?']
    question_dict[df.iloc[25]['URL']]=['How long will the transfer take?']
    question_dict[df.iloc[26]['URL']]=['What investments will my account hold after the transfer?']
    question_dict[df.iloc[27]['URL']]=['What is the difference between an account transfer and a rollover?']
    question_dict[df.iloc[28]['URL']]=['Where should I mail my rollover check?']
    question_dict[df.iloc[29]['URL']]=['What is the guide to my 401(k) rollover']
    question_dict[df.iloc[30]['URL']]=['Will I pay any fees for my rollover?']
    question_dict[df.iloc[31]['URL']]=['Can I reinvest my RMD into another Vanguard account?']
    question_dict[df.iloc[32]['URL']]=['Is my automatic RMD scheduled to run?']
    question_dict[df.iloc[33]['URL']]=['Can I take more than my RMD?']
    question_dict[df.iloc[34]['URL']]=['How do I buy an ETF or stock online?']
    question_dict[df.iloc[35]['URL']]=['Which ETFs are commission-free?', 'Which ETFs are not commission-free?']
    question_dict[df.iloc[36]['URL']]=['What are similarities between ETFs and mutual funds?']
    question_dict[df.iloc[37]['URL']]=['How to buy an ETF?']
    question_dict[df.iloc[38]['URL']]=['How many ways can I place orders on most stocks and ETFs']
    question_dict[df.iloc[39]['URL']]=['How to inherit an IRA?']
    question_dict[df.iloc[40]['URL']]=['How to inherit an account that is not an IRA?']
    question_dict[df.iloc[41]['URL']]=['What if I need to retrieve both my user name and my password?']
    question_dict[df.iloc[42]['URL']]=['Why I can not log on to my small business retirement plan?']
    question_dict[df.iloc[43]['URL']]=['What are the character requirements for my user name and password?']
    question_list = list(question_dict.values())
   
    return question_dict, question_list


def concat_context_docs(df):
    # concatenate all the context documents into one document
    complete_context = ''
    start_marker = [0] # marks the start index of a document
    for df_ind in range(len(df)):
        context = df.iloc[df_ind]['Text']
        complete_context += context
        start_marker.append(start_marker[-1]+len(context)-1)
    return complete_context, start_marker
    