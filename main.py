import pandas as pd
from tqdm.asyncio import tqdm
from openai import AsyncOpenAI
import asyncio
import json
import os

from retrieval import retrieve_docs


with open('config.json') as f:
    config = json.load(f)

openai_api_base = f"http://{config['host']}:{config['port']}/v1"
openai_api_key = "EMPTY"   # не требуется указывать
client = AsyncOpenAI(api_key=openai_api_key, base_url=openai_api_base, max_retries=10)

MODEL_NAME = 't-tech/T-lite-it-1.0'


SYSTEM_MESSAGE = '''Ты финансовый ассистент поддержки Альфа-банка, тебе поступает на вход вопрос клиента банка, а затем документы сначала близкие по смыслу, затем по точным совпадениям слов. 
Тщательно изучи документы перед ответом.
Отвечай только на русском языке полным ответом, но кратко и по делу. Если в предоставленных документах нет нужной информации или в вопросе нет конкретной просьбы или он неконкретен и нечего ответить, отвечай: 
'Нет эталонного ответа'

Ниже представлены примеры вопросов и ответов:

1. Вопрос: Добрый день. Скажите пожалуйста почему не доступен выбор кешбека за оплату услуг ЖКХ? 
   Ответ: Выбор кешбека за оплату услуг ЖКХ может быть недоступен из-за особенностей условий кэшбэка или из-за определённых правил и ограничений, применяемых банком. Чаще всего кэшбэк начисляется не на все категории расходов, а только на определённые, такие как покупки в магазинах, ресторанах и т.д. Каждый банк устанавливает свои правила относительно того, на какие услуги и каким образом можно получить кэшбэк.

2. Вопрос: Я захожу в свою кредитную карту, но не вижу опции Alfa Pay!
   Ответ: Нет эталонного ответа

3. Вопрос: Здравствуйте когда придут деньги за то что через меня сделали карту. 
   Ответ: Сроки получения денег за использование вашей карты зависят от правил банка. Обычно карта Альфа - Cash может быть готова через 3 - 10 рабочих дней после оформления заявления. Если вы хотите узнать, когда именно придут деньги, вам стоит обратиться в контактный центр банка или проверить статус в мобильном приложении.
   
4. Вопрос: Сейчас была в банке мне сказали что сразу получить карту я не могу, только после заказа. Зачем пишите что могу получить ее сразу???!!!!! 
   Ответ: При заказе карты в Альфа-Банке, физическую карту вы можете забрать в отделении банка в течение 90 дней после заказа. Если вы не можете получить карту сразу, это связано с тем, что карта должна быть сначала заказана, а затем доставлена или доступна для получения в отделении. Если вам предоставили информацию о моментальном получении карты, это может относиться к услугам выпуска карт через мобильное приложение, где вы можете получить виртуальную карту, которую можно использовать для онлайн-платежей и добавить в Apple Pay или Google Pay сразу же.

5. Вопрос: Мне надо чтобы платеж пришел быстрее, потому что мне не отдают авто 
   Ответ: Вы можете получить деньги на покупку автомобиля быстро, даже в тот момент, когда вам необходимо. Оформите кредит наличными через онлайн-заявку, и всего за 2 минуты получите предварительное решение. Если вам срочно нужны деньги, у вас есть возможность выбрать удобный способ получения, в том числе доставку средств на дом или в офис. Оставьте заявку, и вы сможете рефинансировать кредит в кратчайшие сроки.

6. Вопрос: Добрый день. У Вас можно оформить карту Юнион пэй?
   Ответ: Нет эталонного ответа
    
'''

USER_PROMPT = '''Вопрос клиента: {question} 

Релевантные документы по смыслу: 
{docs}
'''


async def generate_answer(prompt):
    completion = await client.chat.completions.create(
        n=1,
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,    # для уменьшения галлюцинаций
        max_tokens=300,     # ограничение длины ответа
        timeout=200,
        presence_penalty=1.1,
        frequency_penalty=1.1,
        extra_body={"top_k": 10, "top_p": 0.8,}
    )
    result = completion.choices[0].message.content
    return result


def create_prompt(row):
    return USER_PROMPT.format(question=row['query'], docs=('\n').join(row['best_candidates']))


def get_prompts(data):
    prompts = [create_prompt(row) for i, row in data[['query', 'best_candidates']].iterrows()]
    return prompts


async def run_requests(prompts):
    tasks = []
    for prompt in prompts:
        task = asyncio.create_task(generate_answer(prompt))
        tasks.append(task)
    answers = await tqdm.gather(*tasks)
    return answers



if __name__ == '__main__':
    embedder_name = 'ai-forever/FRIDA'
    questions_path = './questions.csv'
    documents_path = './websites.csv'
    best_docs_path = './best_candidates.csv'

    if os.path.exists(best_docs_path):          # если уже найдены лучшие документы
        retrieval_result = pd.read_csv(best_docs_path)
    else:
        retrieval_result = retrieve_docs(embedder_name, questions_path, documents_path)

    print('Генерация ответов...')
    prompts = get_prompts(retrieval_result)
    batch_size = 1000
    start = 0

    gen_result = []
    for i in range(start, len(prompts), batch_size):
        gen_result.extend(asyncio.run(run_requests(i, prompts[i: i + batch_size])))
    
    print('Сохранение ответов.')
    retrieval_result['answer'] = gen_result
    retrieval_result[['q_id', 'answer']].to_csv('submission.csv', index=False)
