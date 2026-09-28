"""One-line descriptions of the 77 intents for the prompting arms.

Written once from the label names and a few training examples per label (never
test), then frozen before any model saw them. They were not tuned on dev or
test: there were no live calls while building this repo. Kept short on purpose,
because every word is paid for on every one of 3,080 calls and counts against
the deployment's tokens-per-minute limit.
"""

from __future__ import annotations

from b77.data import LABEL_NAMES

DESCRIPTIONS: dict[str, str] = {
    "activate_my_card": "how to activate a card, or activation not working",
    "age_limit": "minimum age to open an account, accounts for children",
    "apple_pay_or_google_pay": "using or topping up with Apple Pay or Google Pay",
    "atm_support": "which ATMs accept the card, finding an ATM",
    "automatic_top_up": "setting up or using auto top-up",
    "balance_not_updated_after_bank_transfer": "a bank transfer in has not shown up in the balance",
    "balance_not_updated_after_cheque_or_cash_deposit": "a cheque or cash deposit is not in the "
    "balance yet",
    "beneficiary_not_allowed": "cannot add a beneficiary or send money to one",
    "cancel_transfer": "wants to cancel or reverse a transfer they made",
    "card_about_to_expire": "card is expiring or expired, getting a replacement",
    "card_acceptance": "where or with which merchants the card can be used",
    "card_arrival": "ordered card has not arrived yet, tracking it",
    "card_delivery_estimate": "how long card delivery takes, expedited delivery",
    "card_linking": "linking or adding a card in the app",
    "card_not_working": "physical card does not work",
    "card_payment_fee_charged": "unexpected fee on a card payment",
    "card_payment_not_recognised": "a card payment they did not make",
    "card_payment_wrong_exchange_rate": "wrong exchange rate on a card payment abroad",
    "card_swallowed": "an ATM kept the card",
    "cash_withdrawal_charge": "fee for withdrawing cash",
    "cash_withdrawal_not_recognised": "a cash withdrawal they did not make",
    "change_pin": "how to change the PIN",
    "compromised_card": "card details stolen or fraud suspected",
    "contactless_not_working": "contactless payments do not work",
    "country_support": "which countries the service or card is available in",
    "declined_card_payment": "a card payment was declined",
    "declined_cash_withdrawal": "an ATM cash withdrawal was declined",
    "declined_transfer": "a transfer was declined",
    "direct_debit_payment_not_recognised": "a direct debit they do not recognise",
    "disposable_card_limits": "limits or restrictions of disposable virtual cards",
    "edit_personal_details": "changing name, address, phone or other details",
    "exchange_charge": "fees for exchanging currency",
    "exchange_rate": "what exchange rate is used or how it is set",
    "exchange_via_app": "how to exchange currency in the app",
    "extra_charge_on_statement": "an unexplained extra charge on the statement",
    "failed_transfer": "a transfer failed or returned an error",
    "fiat_currency_support": "which currencies can be held or exchanged",
    "get_disposable_virtual_card": "getting or using a disposable virtual card",
    "get_physical_card": "getting the PIN or activating a physical card that arrived",
    "getting_spare_card": "ordering an additional or spare card",
    "getting_virtual_card": "getting a virtual card",
    "lost_or_stolen_card": "card lost or stolen",
    "lost_or_stolen_phone": "phone with the app lost or stolen",
    "order_physical_card": "ordering a physical card, delivery places or fees",
    "passcode_forgotten": "forgot the app passcode or password",
    "pending_card_payment": "a card payment is still pending",
    "pending_cash_withdrawal": "a cash withdrawal is still pending",
    "pending_top_up": "a top-up is still pending",
    "pending_transfer": "a transfer is still pending",
    "pin_blocked": "PIN blocked after wrong attempts",
    "receiving_money": "how to receive money or a salary into the account",
    "Refund_not_showing_up": "a refund has not appeared in the account",
    "request_refund": "wants a refund for a purchase",
    "reverted_card_payment?": "a card payment was reverted or came back",
    "supported_cards_and_currencies": "which cards and currencies are supported for top-ups "
    "and payments",
    "terminate_account": "close or delete the account",
    "top_up_by_bank_transfer_charge": "fee for topping up by bank transfer, including SEPA",
    "top_up_by_card_charge": "fee for topping up with a card",
    "top_up_by_cash_or_cheque": "topping up with cash or a cheque, accepted top-up methods",
    "top_up_failed": "a top-up failed or was declined",
    "top_up_limits": "limits on top-up amount or number",
    "top_up_reverted": "a top-up was reverted or cancelled after appearing",
    "topping_up_by_card": "how to top up with a card, card top-up not showing",
    "transaction_charged_twice": "charged twice for one transaction",
    "transfer_fee_charged": "fee charged on a transfer, recipient got less",
    "transfer_into_account": "how to transfer money into the account",
    "transfer_not_received_by_recipient": "the recipient has not received a transfer",
    "transfer_timing": "how long a transfer takes to arrive",
    "unable_to_verify_identity": "identity verification is failing",
    "verify_my_identity": "how to verify identity, documents needed",
    "verify_source_of_funds": "verifying or showing where funds came from",
    "verify_top_up": "verifying a card used for top-ups",
    "virtual_card_not_working": "a virtual card does not work",
    "visa_or_mastercard": "whether cards are Visa or Mastercard",
    "why_verify_identity": "why identity verification is needed",
    "wrong_amount_of_cash_received": "ATM gave less cash than requested",
    "wrong_exchange_rate_for_cash_withdrawal": "wrong exchange rate on a cash withdrawal abroad",
}

if set(DESCRIPTIONS) != set(LABEL_NAMES):  # pragma: no cover - checked at import
    raise RuntimeError("DESCRIPTIONS must cover exactly the 77 label names")


def label_block(with_descriptions: bool = True) -> str:
    """The label list for a prompt, one label per line, in label-id order."""
    if not with_descriptions:
        return "\n".join(LABEL_NAMES)
    return "\n".join(f"{name}: {DESCRIPTIONS[name]}" for name in LABEL_NAMES)
