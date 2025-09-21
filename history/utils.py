from django.db.models import Q

from history.models import QuoteSessionHistory, ContractHistory
from history.serializer import QuoteSessionHistorySerializer, ContractHistorySerializer


COMMAND_WORDS = {
    "найти",
    "поиск",
    "искать",
    "найди",
    "ищу",
    "создать",
    "добавить",
    "сделать",
    "создай",
    "показать",
    "вывести",
    "отобразить",
    "покажи",
    "удалить",
    "удали",
    "стереть",
    "очистить",
    "изменить",
    "редактировать",
    "обновить",
    "посмотреть",
    "гле",
    "какие",
    "что",
    "как",
    "мне",
    "можно",
    "хочу",
    "нужно",
    "надо",
    "где",
    "закупку",
    "закупка",
    "закупки",
}


def clean_search_query(search_query):
    words = search_query.split()
    words = [word for word in words if word.lower() not in COMMAND_WORDS]
    filtered_words = []
    for word in words:
        try:
            zhopa = int(word)
        except:
            filtered_words.append(word)
    return filtered_words


def find_any_history_model(id, inn, search_query):
    words = clean_search_query(search_query)

    # quotes

    if id:
        quotes_q_objects = Q(quote_id=id)
    elif inn:
        quotes_q_objects = Q(Q(customer_inn=inn) | Q(vendor_inn=inn))
    else:
        for word in words:
            if word:
                quotes_q_objects = Q(
                    Q(quote_name__icontains=word)
                    | Q(category_name__icontains=word)
                    # | Q(customer_name__icontains=word)
                    | Q(vendor_name__icontains=word)
                    | Q(fundamental_law__icontains=word)
                )

    # contracts
    contracts_q_objects = Q()

    if id:
        contracts_q_objects = Q(contract_id=id)
    elif inn:
        contracts_q_objects = Q(Q(customer_inn=inn) | Q(vendor_inn=inn))
    else:
        for word in words:
            if word:
                contracts_q_objects = Q(
                    Q(contract_name__icontains=word)
                    | Q(category_name__icontains=word)
                    # | Q(customer_name__icontains=word)
                    | Q(vendor_name__icontains=word)
                    | Q(fundamental_law__icontains=word)
                )

    quotes = QuoteSessionHistory.objects.filter(quotes_q_objects)
    contracts = ContractHistory.objects.filter(contracts_q_objects)
    return {
        "quotes": QuoteSessionHistorySerializer(quotes, many=True).data,
        "contracts": ContractHistorySerializer(contracts, many=True).data,
    }
