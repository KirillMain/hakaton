from django.db.models import Q

from history.models import QuoteSessionHistory, ContractHistory
from history.serializer import QuoteSessionHistorySerializer, ContractHistorySerializer


def find_any_history_model(id, inn, search_query):
    words = search_query.split()

    # quotes
    quotes_q_objects = Q()

    if id:
        quotes_q_objects |= Q(quote_id=id)
    elif inn:
        quotes_q_objects |= Q(customer_inn=inn) | Q(vendor_inn=inn)
    else:
        for word in words:
            if word:
                quotes_q_objects |= (
                    Q(quote_name__icontains=word)
                    | Q(category_name__icontains=word)
                    # | Q(customer_name__icontains=word)
                    | Q(vendor_name__icontains=word)
                    | Q(fundamental_law__icontains=word)
                )

    # contracts
    contracts_q_objects = Q()

    if id:
        contracts_q_objects |= Q(contract_id=id)
    elif inn:
        contracts_q_objects |= Q(customer_inn=inn) | Q(vendor_inn=inn)
    else:
        for word in words:
            if word:
                contracts_q_objects |= (
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
